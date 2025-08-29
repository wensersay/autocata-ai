from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math, time
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.0")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI base (para compatibilidad con tu despliegue actual)
FAST_DPI = int(os.getenv("FAST_DPI", "300").strip() or 300)
PDF_DPI  = int(os.getenv("PDF_DPI",  "400").strip() or 400)

# AutoDPI
AUTO_DPI = (os.getenv("AUTO_DPI", "1").strip() == "1")
DPI_LADDER_ENV = os.getenv("DPI_LADDER", "").strip()

# 2ª línea ruidosa
JUNK_2NDLINE = [x.strip().upper() for x in (os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",")) if x.strip()]

# Refuerzo opcional de nombres (apellidos/nombres frecuentes locales)
NAME_HINTS_EXTRA = [x.strip().upper() for x in os.getenv("NAME_HINTS_EXTRA", "").split(",") if x.strip()]

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ──────────────────────────────────────────────────────────────────────────────
# Modelos I/O
# ──────────────────────────────────────────────────────────────────────────────
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades comunes
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","CSV","DIRECCIÓN","DIRECCION"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

NAME_HINTS_BASE = {
    # Mini base; se refuerza con NAME_HINTS_EXTRA desde ENV
    "MARIA","JOSE","JOSÉ","LUIS","ANTONIO","MANUEL","FRANCISCO","JAVIER",
    "ANA","CARMEN","JUAN","RODRIGUEZ","RODRÍGUEZ","FERNANDEZ","FERNÁNDEZ",
    "GONZALEZ","GONZÁLEZ","LOPEZ","LÓPEZ","ALVAREZ","ALVÁREZ","VARELA",
    "MOSQUERA","VAZQUEZ","VÁZQUEZ","POMBO","DOSINDA","ROGELIO"
}
NAME_HINTS = NAME_HINTS_BASE.union(set(NAME_HINTS_EXTRA))

def parse_dpi_ladder() -> List[int]:
    if AUTO_DPI:
        if DPI_LADDER_ENV:
            ladder = []
            for tok in DPI_LADDER_ENV.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    ladder.append(int(tok))
            if ladder:
                return ladder
        # Fallback si no hay env: usa el DPI que usaríamos por modo + un refuerzo
        first = FAST_DPI if FAST_MODE else PDF_DPI
        second = 340 if first < 340 else first + 40
        return [first, second]
    # sin autodpi → usa el DPI por modo
    return [FAST_DPI if FAST_MODE else PDF_DPI]

def fetch_pdf_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el PDF (HTTP {r.status_code}).")
        ct = (r.headers.get("content-type") or "").lower()
        if "pdf" not in ct and not r.content.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="La URL no parece entregar un PDF válido.")
        return r.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error de red al descargar el PDF: {e}")

def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Raster (pág. 2)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes, dpi: int) -> np.ndarray:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

# ──────────────────────────────────────────────────────────────────────────────
# Detección colores y centros (puntos verde/rojo)
# ──────────────────────────────────────────────────────────────────────────────
def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160, 20, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int,int,int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Cálculo de lado (8 rumbos)
# ──────────────────────────────────────────────────────────────────────────────
OCTANTS = ["este","noreste","norte","noroeste","oeste","suroeste","sur","sureste"]
OCT_LABELS = {"norte":"N","noreste":"NE","este":"E","sureste":"SE","sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"}

def side8_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=este, 90=norte
    # Mapear a 8 sectores de 45°
    idx = int(round(((ang % 360) / 45.0))) % 8
    return OCTANTS[idx]

# ──────────────────────────────────────────────────────────────────────────────
# OCR utils
# ──────────────────────────────────────────────────────────────────────────────
def enhance_gray(g: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

def binarize(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
    flags_inv = THRESH_BINARY_INV | (THRESH_OTSU if THRESH_OTSU else 0)
    _, bw  = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_bin)
    _, bwi = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_inv)
    return bw, bwi

def ocr_text(img: np.ndarray, psm: int, whitelist: Optional[str] = None) -> str:
    cfg = f"--psm {psm} --oem 3"
    if whitelist is not None:
        safe = (whitelist or "").replace('"','')
        cfg += f' -c tessedit_char_whitelist="{safe}"'
    txt = pytesseract.image_to_string(img, config=cfg) or ""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def clean_owner_line(line: str) -> str:
    if not line: return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in GEO_TOKENS or "[" in t or "]" in t: break
        if t in BAD_TOKENS: continue
        # filtrar tokens sueltos típicos de encabezado OCR
        if t in {"A","N","R","RZ"}: 
            continue
        out.append(t)
        # parar si ya hay suficientes tokens
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:48]

def pick_owner_from_text(txt: str) -> str:
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):
            continue
        if sum(ch.isdigit() for ch in U) > 1:
            continue
        if not UPPER_NAME_RE.match(U):
            continue
        name = clean_owner_line(U)
        if len(name) >= 8:
            return name
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna “Apellidos…” y extraer línea 1 + línea 2 (cuando aplique)
# ──────────────────────────────────────────────────────────────────────────────
def find_header_window(bgr: np.ndarray) -> Tuple[int,int]:
    """
    Devuelve (x_left_header, x_nif) aproximados para recortar ROI de nombre.
    Si no detecta, retorna (int(w*0.33), int(w*0.62)) como fallback.
    """
    h, w = bgr.shape[:2]
    x0_fallback = int(w * 0.33)
    x1_fallback = int(w * 0.62)

    # Usar una banda amplia de la zona de texto (medio-superior)
    y0, y1 = int(h*0.08), int(h*0.24)
    band = bgr[y0:y1, int(w*0.28):int(w*0.95)]
    if band.size == 0:
        return x0_fallback, x1_fallback

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    best_left = None
    best_nifx = None

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", []); xs = data.get("left", []); ys = data.get("top", []); ws = data.get("width", []); hs = data.get("height", [])
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T or "APELLIDOS/NOMBRE" in T or "APELLIDOS" in T:
                x_abs = int(w*0.28) + lx
                best_left = x_abs if best_left is None else min(best_left, x_abs)
            if T == "NIF":
                x_abs = int(w*0.28) + lx
                best_nifx = x_abs if best_nifx is None else min(best_nifx, x_abs)

    if best_left is None and best_nifx is None:
        return x0_fallback, x1_fallback

    if best_left is None: best_left = x0_fallback
    if best_nifx is None: best_nifx = x1_fallback
    if best_nifx - best_left < int(w*0.18):
        best_nifx = best_left + int(w*0.24)
    return max(0, best_left-10), min(w-1, best_nifx-8)

def extract_name_lines(bgr: np.ndarray, row_y: int, header_x0: int, header_x1: int) -> Tuple[str, str, dict]:
    """
    Extrae línea 1 (nombre base) y posible continuación en línea 2.
    También captura el caso de salto de línea dentro de la primera línea (t1_raw con \n).
    """
    h, w = bgr.shape[:2]
    # Banda vertical alrededor de la fila
    band_h = int(h * 0.17)  # suficiente para captar L1 y L2
    y0 = max(0, row_y - band_h//2)
    y1 = min(h, row_y + band_h//2)
    x0 = max(0, header_x0)
    x1 = min(w, header_x1)

    band = bgr[y0:y1, x0:x1]
    dbg = {"band":[x0,y0,x1,y1]}
    if band.size == 0:
        return "", "", dbg

    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    g = enhance_gray(g)
    bw, bwi = binarize(g)

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    # OCR principal en modo línea(s) (psm 6/7) para capturar posibles saltos
    variants = [
        ocr_text(bw,  psm=6,  whitelist=WL),
        ocr_text(bwi, psm=6,  whitelist=WL),
        ocr_text(bw,  psm=7,  whitelist=WL),
        ocr_text(bwi, psm=7,  whitelist=WL),
    ]
    t1_raw = ""
    for txt in variants:
        if txt:
            t1_raw = txt
            break

    # Normalizar y detectar si la "línea 1" trae salto incluido (nombre+continuación)
    t1_lines = [ln.strip() for ln in (t1_raw.split("\n") if t1_raw else []) if ln.strip()]
    t1_extra = ""
    if len(t1_lines) >= 2:
        # Si la 2ª sublínea parece nombre, úsala como extra (continuación)
        cand = clean_owner_line(t1_lines[1])
        if len(cand) >= 2:
            t1_extra = cand

    # Intento de segunda línea explícita: recorta mitad inferior de la banda para "L2"
    h_band = band.shape[0]
    l2_roi = band[(h_band//2):, :] if h_band > 10 else None
    t2_raw = ""
    if l2_roi is not None and l2_roi.size > 0:
        gg = cv2.cvtColor(l2_roi, cv2.COLOR_BGR2GRAY)
        gg = cv2.resize(gg, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        gg = enhance_gray(gg)
        bw2, bwi2 = binarize(gg)
        for txt in (ocr_text(bw2, psm=6, whitelist=WL), ocr_text(bwi2, psm=6, whitelist=WL)):
            if txt:
                t2_raw = txt
                break

    dbg.update({"t1_raw": t1_raw, "t1_extra_raw": t1_extra, "t2_raw": t2_raw})
    # Limpiar nombres finales
    base = clean_owner_line(t1_lines[0] if t1_lines else t1_raw)
    add2 = ""
    # preferencia: si t1_extra trae un nombre razonable, úsalo
    if t1_extra and t1_extra not in JUNK_2NDLINE:
        add2 = t1_extra
    elif t2_raw:
        # tomar solo el primer renglón de t2_raw
        cand2 = (t2_raw.split("\n")[0] or "").strip().upper()
        # cortar en primer número/corchete
        cand2 = re.split(r"[\d\[\]:]", cand2)[0].strip()
        # recortar tokens de basura comunes
        if cand2 in JUNK_2NDLINE:
            cand2 = ""
        # reforzar si contiene nombre común
        if cand2 and (len(cand2) <= 26) and (len(cand2) >= 2):
            add2 = cand2

    return base, add2, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte de área de croquis a la izquierda
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.05); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    # Centrales (verde) y vecinos (magenta/rojo)
    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 160))
    if not mains:
        return (
            {k:"" for k in OCT_LABELS.keys()},
            {"rows": [], "note":"no_mains"},
            vis
        )

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # por Y
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    header_x0, header_x1 = find_header_window(bgr)

    rows_dbg = []
    linderos = {k:"" for k in OCT_LABELS.keys()}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:8]:
        # Vecino más próximo
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.30)**2:
            side = side8_of((mcx, mcy), best)

        base, add2, od = extract_name_lines(bgr, row_y=mcy, header_x0=header_x0, header_x1=header_x1)

        # Combinar
        owner = base
        if add2 and add2 not in owner:
            if len(owner) + 1 + len(add2) <= 48:
                owner = f"{owner} {add2}"

        # Aplicar
        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = OCT_LABELS.get(side, "")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": od,
            "header_left_abs": header_x0,
            "x_nif_abs": header_x1
        })

    # owners_detected
    owners_detected = [r["owner"] for r in rows_dbg if r.get("owner")]
    owners_detected = list(dict.fromkeys(owners_detected))[:12]

    dbg = {"rows": rows_dbg}
    return linderos, {"rows": rows_dbg, "owners_detected": owners_detected}, vis

# ──────────────────────────────────────────────────────────────────────────────
# Heurística de calidad simple para AutoDPI
# ──────────────────────────────────────────────────────────────────────────────
def is_noise_name(name: str) -> bool:
    if not name: return True
    U = name.upper()
    if any(tok in U for tok in BAD_TOKENS): return True
    if "[" in U or "]" in U: return True
    if len(U) < 4: return True
    return False

def evaluate_quality(linderos: Dict[str,str], owners_detected: List[str]) -> Tuple[bool, float, str]:
    sides_ok = sum(1 for v in linderos.values() if v)
    owners_ok = sum(1 for v in owners_detected if not is_noise_name(v))
    # criterio OK:
    #  - 3+ lados con nombre, o
    #  - 2 lados + 3 owners limpios
    if sides_ok >= 3:
        return True, sides_ok + owners_ok/10.0, "sides>=3"
    if sides_ok >= 2 and owners_ok >= 3:
        return True, sides_ok + owners_ok/10.0, "sides>=2_and_owners>=3"
    return False, sides_ok + owners_ok/10.0, "low_confidence"

# ──────────────────────────────────────────────────────────────────────────────
# AutoDPI wrapper
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(pdf_bytes: bytes, dpi: int, annotate: bool, names: bool):
    bgr = page2_bgr(pdf_bytes, dpi=dpi)
    linderos, vdbg, vis = detect_rows_and_extract(bgr, annotate=annotate, annotate_names=names)
    owners = vdbg.get("owners_detected", [r.get("owner","") for r in vdbg.get("rows",[]) if r.get("owner")])
    ok, score, reason = evaluate_quality(linderos, owners)
    return {
        "linderos": linderos,
        "owners_detected": owners,
        "dbg": vdbg,
        "vis": vis,
        "ok": ok,
        "score": score,
        "reason": reason,
        "dpi": dpi
    }

def autodpi_best_attempt(pdf_bytes: bytes, annotate: bool, names: bool):
    attempts_info = []
    ladder = parse_dpi_ladder()
    chosen = None

    for idx, dpi in enumerate(ladder):
        t0 = time.time()
        res = run_pipeline(pdf_bytes, dpi=dpi, annotate=annotate, names=names)
        ms = int((time.time() - t0) * 1000)
        attempts_info.append({"dpi": dpi, "ok": res["ok"], "score": res["score"], "reason": res["reason"], "ms": ms})
        if res["ok"]:
            chosen = res
            break
        if idx == 0 and AUTO_DPI:
            # no ok → probamos siguiente dpi (si existe)
            continue

    if chosen is None:
        # usar el mejor score si ninguno ok
        chosen = max(
            (run_pipeline(pdf_bytes, dpi=d, annotate=annotate, names=names) if a.get("dpi")!=d else None)
            if False else None  # (ya lo corrimos arriba; evitar rerun)
            for d, a in [(i["dpi"], i) for i in attempts_info]
        )
        # si no re-ejecutamos, el "chosen" será el último intento
        chosen = chosen if chosen is not None else run_pipeline(pdf_bytes, dpi=ladder[-1], annotate=annotate, names=names)

    chosen_dbg = chosen["dbg"]
    chosen_dbg["raster"] = {
        "dpi": chosen["dpi"],
        "ladder_used": ladder,
        "retry_used": (len(attempts_info) > 1 and attempts_info[0]["ok"] is False),
        "attempts": attempts_info
    }
    return chosen

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "FAST_DPI": FAST_DPI,
        "PDF_DPI": PDF_DPI,
        "AUTO_DPI": AUTO_DPI,
        "DPI_LADDER": parse_dpi_ladder(),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    if TEXT_ONLY:
        # Generar PNG mínimo con aviso
        blank = np.zeros((280, 900, 3), np.uint8)
        cv2.putText(blank, "TEXT_ONLY: preview desactivado", (20,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    try:
        res = autodpi_best_attempt(pdf_bytes, annotate=bool(labels), names=bool(names))
        vis = res["vis"]
    except Exception as e:
        err = str(e)
        blank = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:60]}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...),
                 labels: int = Query(0),
                 names: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        return ExtractOut(
            linderos={k:"" for k in OCT_LABELS.keys()},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        res = autodpi_best_attempt(pdf_bytes, annotate=False, names=False)
        linderos = res["linderos"]
        owners_detected = res["owners_detected"]
        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = res["dbg"] if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={k:"" for k in OCT_LABELS.keys()},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )







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
app = FastAPI(title="AutoCatastro AI", version="0.6.1")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad / rendimiento
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI por entorno (si no, 340 en FAST / 500 en normal)
def get_raster_dpi() -> int:
    env = os.getenv("RASTER_DPI", "").strip()
    if env.isdigit():
        return max(220, min(600, int(env)))
    return 340 if FAST_MODE else 500

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
# Constantes / utilidades de texto
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","CSV","DIRECCIÓN","DIRECCION",
    "HOJA","CATRASTRAL","CATASTRO","GENERAL","SELLO"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","O","DE","DEL","LA","EL","AL"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Hints de nombres (banco corto + extra por ENV)
DEFAULT_NAME_HINTS = {
    "MARIA","JOSE","LUIS","ANTONIO","JUAN","MANUEL","FRANCISCO","DAVID","JAVIER",
    "CARLOS","ALEJANDRO","DANIEL","MIGUEL","RAFAEL","PEDRO","FERNANDO","PABLO",
    "ANDRES","ALVARO","ANGEL","SERGIO","ROBERTO","RODRIGUEZ","GARCIA","LOPEZ",
    "PEREZ","MARTINEZ","SANCHEZ","GOMEZ","FERNANDEZ","ALVAREZ","VARELA","VARGAS",
    "RODRIGUEZ","DOMINGUEZ","DIEZ","SUAREZ","IGLESIAS"
}
EXTRA_HINTS = set([s.strip().upper() for s in os.getenv("NAME_HINTS_EXTRA","").split(",") if s.strip()])

def collapse_spaces(s: str) -> str:
    return re.sub(r"\s{2,}", " ", (s or "").strip())

def strip_leading_singletons(s: str) -> str:
    toks = [t for t in s.split() if t]
    while toks and len(toks[0]) == 1:
        toks.pop(0)
    return " ".join(toks)

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
# Raster (pág. 2) y masks
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = get_raster_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def crop_map(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    h, w = bgr.shape[:2]
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.42)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    return bgr[top:bottom, left:right], (left, top)

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
# Geometría: octantes
# ──────────────────────────────────────────────────────────────────────────────
SIDE_LABELS = ["este","noreste","norte","noroeste","oeste","suroeste","sur","sureste"]

def side_of_octant(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0°=Este, 90°=Norte
    if ang < 0: ang += 360.0
    # octantes de 45°
    idx = int(((ang + 22.5) % 360) // 45)
    return SIDE_LABELS[idx]

# ──────────────────────────────────────────────────────────────────────────────
# OCR utils y saneo
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

def sanitize_l1(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ '\-]", " ", s)
    s = collapse_spaces(s)
    if any(ch.isdigit() for ch in s):
        s = re.split(r"[\[\]:0-9]", s)[0]
    toks = [t for t in s.split() if t]
    toks2 = []
    for t in toks:
        if t in BAD_TOKENS: continue
        if t in GEO_TOKENS: continue
        toks2.append(t)
    s = " ".join(toks2)
    s = strip_leading_singletons(s)
    return s[:48]

def sanitize_l2(s: str) -> str:
    s = (s or "").upper()
    # corta al primer número / [ ] :
    s = re.split(r"[\[\]:0-9]", s)[0]
    s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ '\-]", " ", s)
    s = collapse_spaces(s)
    # eliminar tokens iniciales de 1 carácter
    s = strip_leading_singletons(s)
    # descartes por ruido (SS, KK, 1-2 letras, repetición)
    core = s.replace(" ", "")
    if len(s) <= 2 or (core and len(set(core)) == 1):
        return ""
    if core and (max(core.count(ch) for ch in set(core)) / len(core)) > 0.7:
        return ""
    return s[:26]

def combine_name(name1: str, extra_from_l1: str, name2: str) -> str:
    # preferir añadir el “extra_from_l1” si es token corto de hints (e.g., LUIS)
    parts = [p for p in [name1, extra_from_l1, name2] if p]
    s = " ".join(parts)
    return collapse_spaces(s)[:64]

# ──────────────────────────────────────────────────────────────────────────────
# Localización de columna “Apellidos/Nombre … NIF” y lectura de 2 líneas
# ──────────────────────────────────────────────────────────────────────────────
def find_header_band_and_cols(bgr: np.ndarray, row_y: int) -> Tuple[Tuple[int,int], Tuple[int,int,int,int], dict]:
    """
    Devuelve:
      (x0_text, x1_text), (x_band0, y_band0, x_band1, y_band1), debug
    Busca 'APELLIDOS' y 'NIF' mediante OCR en una banda vertical alrededor de row_y.
    """
    h, w = bgr.shape[:2]
    y0s = max(0, row_y - int(h*0.10))
    y1s = min(h, row_y + int(h*0.10))
    x0s = int(w * 0.30)
    x1s = int(w * 0.97)

    band = bgr[y0s:y1s, x0s:x1s]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    header_left = None
    x_nif = None
    header_bottom = None

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_left = x0s + lx if header_left is None else min(header_left, x0s + lx)
                header_bottom = max(header_bottom or 0, y0s + ty + hh)
            elif T == "NIF":
                x_nif = x0s + lx
                header_bottom = max(header_bottom or 0, y0s + ty + hh)

    # Fallback si no encontramos cabecera
    if header_bottom is None:
        header_left = x0s + int(0.02*(x1s-x0s))
        x_nif = x0s + int(0.60*(x1s-x0s))
        header_bottom = row_y - int(h*0.01)

    x0_text = max(0, (header_left or x0s) - int(0.02 * w))
    x1_text = min(w, (x_nif or x1s) - int(0.01 * w))

    band2_h = int(h * 0.20)  # banda generosa para buscar ambas líneas
    yb0 = max(0, header_bottom + 2)
    yb1 = min(h, yb0 + band2_h)

    dbg = {
        "header_left_abs": int(header_left) if header_left is not None else None,
        "x_nif_abs": int(x_nif) if x_nif is not None else None
    }
    return (int(x0_text), int(x1_text)), (int(x0s), int(y0s), int(x1s), int(y1s)), dbg

def read_two_lines(bgr: np.ndarray, row_y: int) -> Tuple[str, str, str, Tuple[int,int,int,int], Tuple[int,int]]:
    """
    Lee dos líneas bajo la cabecera:
      devuelve (t1_raw, t1_extra_raw, t2_raw, band_xyxy, y_line1, y_line2_hint)
    t1_extra_raw recoge un posible salto de línea partido en la propia línea1.
    """
    h, w = bgr.shape[:2]
    (x0_text, x1_text), (_bx0, _by0, _bx1, _by1), _dbg = find_header_band_and_cols(bgr, row_y)

    # Definir banda grande y 2 franjas (línea 1 y pista para línea 2)
    y_base = row_y  # aproximación vertical
    y1a = max(0, y_base - int(h*0.01))
    y1b = min(h, y1a + int(h*0.04))
    y2a = min(h, y1b + 2)
    y2b = min(h, y2a + int(h*0.04))

    band = bgr[min(y1a, y2a):max(y1b, y2b), x0_text:x1_text]
    band_xyxy = (x0_text, min(y1a, y2a), x1_text, max(y1b, y2b))

    # ROIs absolutas
    l1_xy = (x0_text, y1a, x1_text, y1b)
    l2_xy = (x0_text, y2a, x1_text, y2b)

    # OCR línea 1
    roi1 = bgr[y1a:y1b, x0_text:x1_text]
    t1_raw, t1_extra_raw = "", ""
    if roi1.size > 0:
        g = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
        v1 = [
            ocr_text(bw, psm=6, whitelist=WL),
            ocr_text(bwi,psm=6, whitelist=WL),
            ocr_text(bw, psm=7, whitelist=WL),
            ocr_text(bwi,psm=7, whitelist=WL)
        ]
        for v in v1:
            if "\n" in v:
                parts = [p.strip() for p in v.split("\n") if p.strip()]
                if parts:
                    t1_raw = parts[0]
                    if len(parts) >= 2:
                        t1_extra_raw = parts[1]
                    break
            elif v.strip():
                t1_raw = v.strip()
                break

    # OCR línea 2 (pista)
    roi2 = bgr[y2a:y2b, x0_text:x1_text]
    t2_raw = ""
    if roi2.size > 0:
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g2 = enhance_gray(g2)
        bw2, bwi2 = binarize(g2)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
        v2 = [
            ocr_text(bw2,  psm=6, whitelist=WL),
            ocr_text(bwi2, psm=6, whitelist=WL),
            ocr_text(bw2,  psm=7, whitelist=WL),
            ocr_text(bwi2, psm=7, whitelist=WL),
        ]
        for v in v2:
            if v.strip():
                t2_raw = v.strip()
                break

    return t1_raw, t1_extra_raw, t2_raw, band_xyxy, (y1a, y1b)

def extract_owner_for_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Devuelve (owner, dbg_dict) usando 2 líneas bajo la cabecera.
    - L1 se limpia estricta.
    - L1_extra (si hay salto) + L2 se tratan de forma permisiva (saneo nuevo).
    """
    t1_raw, t1_extra_raw, t2_raw, (bx0, by0, bx1, by1), (l1a, l1b) = read_two_lines(bgr, row_y)

    name1 = sanitize_l1(t1_raw)
    extra_from_l1 = sanitize_l2(t1_extra_raw)
    name2_clean = sanitize_l2(t2_raw)

    # Heurística de hints (por si L2 es un nombre típico corto)
    hints = DEFAULT_NAME_HINTS | EXTRA_HINTS
    if not name2_clean:
        t2_tokens = [t for t in (t2_raw or "").upper().split() if t]
        for t in t2_tokens:
            if t in hints:
                name2_clean = t
                break

    # oculta t2_raw en debug si es ruido evidente
    t2_show = t2_raw
    core = (t2_raw or "").replace(" ", "")
    if (not name2_clean) and (len(core) <= 2 or (core and len(set(core)) == 1)):
        t2_show = ""

    # Caso en el que el extra válido viene de la propia L1 partida
    picked_from = "strict"
    if extra_from_l1 and not name2_clean:
        name = combine_name(name1, extra_from_l1, "")
        picked_from = "from_l1_break"
    else:
        name = combine_name(name1, "", name2_clean)
        if name2_clean:
            picked_from = "strict"

    dbg = {
        "band": [bx0, by0, bx1, by1],
        "y_line1": [l1a, l1b],
        "y_line2_hint": [l1b+2, min(bgr.shape[0], l1b + 2 + int(bgr.shape[0]*0.04))],
        "x0": bx0,
        "x1": bx1,
        "t1_raw": t1_raw,
        "t1_extra_raw": t1_extra_raw if extra_from_l1 else "",
        "t2_raw": t2_show,
        "picked_from": picked_from
    }
    return name, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline: detectar filas, octante y extraer nombres
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False,
                            eight_dirs: bool = True) -> Tuple[Dict[str,str], dict, np.ndarray]:
    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(340 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 160))
    if not mains:
        return ({s:"" for s in SIDE_LABELS},
                {"rows": [], "timings_ms":{"rows_pipeline": int((time.time()-t0)*1000)}},
                vis)

    mains_abs  = [(cx+ox, cy+oy, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+ox, cy+oy, a) for (cx,cy,a) in neighs]

    linderos = {s: "" for s in SIDE_LABELS}
    used_sides = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecino más cercano
        best = None; best_d = 1e12
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.30)**2:
            side = side_of_octant((mcx, mcy), best)

        owner, ocr_dbg = extract_owner_for_row(bgr, row_y=mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl_map = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"
                }
                lbl = lbl_map.get(side, "")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if annotate_names and owner:
                cv2.putText(vis, owner[:28], (int(w*0.45), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.45), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
        })

    timings = {"rows_pipeline": int((time.time()-t0)*1000)}
    dbg = {"rows": rows_dbg, "timings_ms": timings, "raster": {"dpi": get_raster_dpi()}}

    return linderos, dbg, vis

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
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)},
        "raster_dpi": get_raster_dpi(),
        "name_hints_extra": sorted(list(EXTRA_HINTS)) if EXTRA_HINTS else []
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar rótulos N/NE/E/SE/S/SO/O/NO"),
    names: int = Query(0, description="1=mostrar nombre estimado al lado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names), eight_dirs=True
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((260, 880, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:70]}", (10,140),
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
            linderos={s:"" for s in SIDE_LABELS},
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False, eight_dirs=True)
        # owners_detected = únicos, en orden
        owners_detected = []
        for r in vdbg.get("rows", []):
            o = r.get("owner") or ""
            if o and o not in owners_detected:
                owners_detected.append(o)
        owners_detected = owners_detected[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={s:"" for s in SIDE_LABELS},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )




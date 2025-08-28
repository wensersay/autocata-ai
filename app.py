from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.2")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE", "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY", "0").strip() == "1")

# DPI (prioridad: FAST_DPI si FAST_MODE=1, si no PDF_DPI)
FAST_DPI = _get_env_int("FAST_DPI", 340)
PDF_DPI  = _get_env_int("PDF_DPI", 420)

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def current_dpi() -> int:
    dpi = FAST_DPI if FAST_MODE else PDF_DPI
    return _clamp(dpi, 220, 600)

# Ruido típico en 2ª línea (configurable)
JUNK_2NDLINE = {t.strip().upper() for t in (os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(","))}
# Pistas de nombres extra (configurable)
NAME_HINTS_EXTRA = [t.strip().upper() for t in os.getenv("NAME_HINTS_EXTRA", "").split(",") if t.strip()]

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
    "CATASTRAL","TITULARIDAD","PRINCIPAL"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

NAME_HINTS_BASE = [
    "JOSE","JOSÉ","LUIS","MARIA","MARÍA","ANTONIO","MANUEL","FRANCISCO","JAVIER",
    "ANA","RODRIGUEZ","RODRÍGUEZ","GARCIA","GARCÍA","FERNANDEZ","FERNÁNDEZ",
    "PEREZ","PÉREZ","LOPEZ","LÓPEZ","ALVAREZ","ALVÁREZ","VARELA","MOSQUERA",
    "POMBO","VAZQUEZ","VÁZQUEZ"
] + NAME_HINTS_EXTRA

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
    dpi = current_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

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
    for lo, hi in g_ranges:
        mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges:
        mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int,int,int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: 
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

# 8 rumbos
DIR_LABELS = {
    "norte":"N", "noreste":"NE", "este":"E", "sureste":"SE",
    "sur":"S", "suroeste":"SO", "oeste":"O", "noroeste":"NO"
}

def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=E, 90=N
    # sector de 45º
    if -22.5 <= ang < 22.5:    return "este"
    if 22.5 <= ang < 67.5:     return "noreste"
    if 67.5 <= ang < 112.5:    return "norte"
    if 112.5 <= ang < 157.5:   return "noroeste"
    if ang >= 157.5 or ang < -157.5: return "oeste"
    if -157.5 <= ang < -112.5: return "suroeste"
    if -112.5 <= ang < -67.5:  return "sur"
    return "sureste"

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
        out.append(t)
        # parar si ya tenemos 4–5 tokens útiles
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    # compactar conectores múltiples
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:64]

def sanitize_second_line(s2: str) -> str:
    if not s2: return ""
    U = s2.upper().strip()
    # cortar en dígitos o corchetes / dos puntos
    U = re.split(r"[\[\]:0-9]", U)[0].strip()
    U = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\s\.'\-]", " ", U)
    U = re.sub(r"\s{2,}", " ", U).strip()
    if not U:
        return ""
    if U in JUNK_2NDLINE:
        return ""
    return U[:26]

# ──────────────────────────────────────────────────────────────────────────────
# Localizar banda y leer L1/L2
# ──────────────────────────────────────────────────────────────────────────────
def find_columns_once(bgr: np.ndarray) -> Tuple[int,int]:
    """
    Busca 'APELLIDOS' y 'NIF' una sola vez en la página para fijar x0/x1 de la columna.
    Si no encuentra, usa heurístico por porcentaje.
    """
    h, w = bgr.shape[:2]
    page_g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(page_g)
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ws = data.get("width", [])
        found_ap = []
        x_nif = None
        for t, lx, ww in zip(words, xs, ws):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                found_ap.append(lx)
            if T == "NIF":
                x_nif = lx
        if (found_ap and x_nif is not None):
            x0 = max(0, min(found_ap) - int(0.01*w))
            x1 = max(x0+10, x_nif - int(0.01*w))
            return x0, x1
    # Fallback
    return int(0.31*w), int(0.61*w)

def read_owner_two_lines(bgr: np.ndarray, row_y: int, x0: int, x1: int) -> Tuple[str, dict]:
    """
    Extrae L1 y L2 de la columna x0..x1 alrededor de row_y.
    Aplica:
      - Si L1 contiene salto con un segundo renglón "limpio", se usa (from_l1_break).
      - Si L2 viene limpia y no es ruido, se concatena (2ndline_ok).
      - Si L2 es ruido (en JUNK_2NDLINE), se ignora.
    """
    h, w = bgr.shape[:2]
    line_h = int(h * 0.045)               # altura aproximada por línea
    band_top = max(0, row_y - int(h*0.055))
    band_bot = min(h, row_y + int(h*0.055))
    band = bgr[band_top:band_bot, x0:x1]
    dbg = {"band":[x0, band_top, x1, band_bot]}
    if band.size == 0:
        return "", {"band": dbg["band"], "t1_raw":"", "t2_raw":""}

    # L1 (justo alrededor de row_y)
    l1_top = max(band_top, row_y - line_h//2)
    l1_bot = min(band_bot, l1_top + line_h)
    l1 = bgr[l1_top:l1_bot, x0:x1]
    # L2 (debajo de L1)
    l2_top = min(band_bot, l1_bot + 2)
    l2_bot = min(band_bot, l2_top + line_h)
    l2 = bgr[l2_top:l2_bot, x0:x1]

    dbg.update({"y_line1":[l1_top, l1_bot], "y_line2_hint":[l2_top, l2_bot], "x0":x0, "x1":x1})

    def _ocr_clean(img: np.ndarray) -> str:
        if img is None or img.size == 0:
            return ""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        variants = [
            ocr_text(bw,  psm=6,  whitelist=WL),
            ocr_text(bwi, psm=6,  whitelist=WL),
            ocr_text(bw,  psm=7,  whitelist=WL),
            ocr_text(bwi, psm=7,  whitelist=WL),
            ocr_text(bw,  psm=13, whitelist=WL),
        ]
        # mejor línea (más tokens de nombre o hints)
        best = ""
        best_score = -1
        for t in variants:
            U = t.strip().upper()
            if not U: 
                continue
            toks = [x for x in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", U) if x]
            score = sum(1 for x in toks if (x in NAME_HINTS_BASE)) + len([x for x in toks if x not in NAME_CONNECTORS])
            if score > best_score:
                best = U
                best_score = score
        return best

    t1_raw = _ocr_clean(l1)
    t2_raw = _ocr_clean(l2)
    dbg["t1_raw"] = t1_raw
    dbg["t2_raw"] = t2_raw

    # ¿L1 trae salto interno?
    if "\n" in t1_raw:
        parts = [p.strip() for p in t1_raw.split("\n") if p.strip()]
        if len(parts) >= 2:
            base = clean_owner_line(parts[0])
            cont = sanitize_second_line(parts[1])
            if base and cont:
                dbg["picked_from"] = "from_l1_break"
                return (f"{base} {cont}").strip(), dbg
            elif base:
                dbg["picked_from"] = "from_l1_first_only"
                return base, dbg

    base = clean_owner_line(t1_raw)
    if not base:
        # intenta con L2 sola (raro)
        alt = clean_owner_line(t2_raw)
        if alt:
            dbg["picked_from"] = "l2_only"
            return alt, dbg
        dbg["picked_from"] = "empty"
        return "", dbg

    # concatenar 2ª línea si es válida y no ruido
    second = sanitize_second_line(t2_raw)
    if second and second not in JUNK_2NDLINE:
        dbg["picked_from"] = "strict"
        return f"{base} {second}".strip(), dbg

    dbg["picked_from"] = "strict_no_l2"
    return base, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:

    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Región del croquis (izquierda)
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    if not mains:
        return {k:"" for k in DIR_LABELS.keys()}, {"rows": [], "raster":{"dpi":current_dpi()}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    x0_col, x1_col = find_columns_once(bgr)

    rows_dbg = []
    linderos = {k:"" for k in DIR_LABELS.keys()}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano
        best = None; best_d = 1e12
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None:
            side = side_of8((mcx, mcy), best)

        # OCR L1/L2 de la columna de texto
        owner, ocr_dbg = read_owner_two_lines(bgr, row_y=mcy, x0=x0_col, x1=x1_col)

        # rellenar linderos (no machacar si ya está ese rumbo)
        if side and owner and (side not in used_sides):
            linderos[side] = owner
            used_sides.add(side)

        # anotaciones
        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = DIR_LABELS.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
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
            "ocr": ocr_dbg
        })

    dbg = {"rows": rows_dbg, "raster": {"dpi": current_dpi()}}
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
        "pdf_dpi": PDF_DPI,
        "fast_dpi": FAST_DPI,
        "effective_dpi": current_dpi(),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar rumbos N/NE/E/..."),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((260, 900, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:120]}", (10,130),
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
            linderos={k:"" for k in DIR_LABELS.keys()},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        # quitar duplicados pero mantener orden
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={k:"" for k in DIR_LABELS.keys()},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )



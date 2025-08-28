from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple, Set
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
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

def get_env_set(name: str, default: List[str]) -> Set[str]:
    raw = os.getenv(name, "")
    if not raw:
        return set(default)
    return set([s.strip().upper() for s in raw.split(",") if s.strip()])

# Ruido típico de 2ª línea (configurable por ENV)
JUNK_2NDLINE = get_env_set("JUNK_2NDLINE", ["Z", "VA", "EO", "SS", "KO", "KR", "LN"])

# DPI por modo (ENV: FAST_DPI, PDF_DPI)
def get_raster_dpi() -> int:
    if FAST_MODE:
        try:
            return int(os.getenv("FAST_DPI", "340"))
        except:
            return 340
    try:
        return int(os.getenv("PDF_DPI", "400"))
    except:
        return 400

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
    "GALICIA","[LUGO]","[BARCELONA]","CSV","DIRECCIÓN","DIRECCION"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

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
def page2_bgr(pdf_bytes: bytes) -> Tuple[np.ndarray, int]:
    dpi = get_raster_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1], dpi

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
# Ocho vientos
# ──────────────────────────────────────────────────────────────────────────────
def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # Norte arriba
    # Octantes de 45°
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

# ──────────────────────────────────────────────────────────────────────────────
# PARCHE: detección robusta de cabecera + limpieza + unión de segunda línea
# ──────────────────────────────────────────────────────────────────────────────
def looks_like_header_line(s: str) -> bool:
    """
    Detecta si un renglón se parece al encabezado 'Apellidos Nombre / Razón social'
    incluso con OCR roto (trozos tipo 'A LEV', 'NOM', 'RAZ', 'SOC', 'NIF').
    """
    u = (s or "").upper()
    u = re.sub(r"[^A-ZÁÉÍÓÚÜÑ ]+", " ", u)
    u = re.sub(r"(.)\1{2,}", r"\1\1", u)

    needles = ("APELL", "APEL", "AP ", " A P ", "NOM", "NOMB", "RAZ", "RAZON", "RAZÓN", "SOC", "SOCIAL", "NIF")
    hits = sum(1 for pat in needles if pat in u)

    toks = [t for t in u.split() if t]
    short = [t for t in toks if len(t) <= 2]
    tiny_ratio = (len(short) / max(1, len(toks))) if toks else 0.0
    short_garbage = {"A","R","D","N","RO","RZ","RN","AN","AP","NO","SO"}

    looks_broken_header = tiny_ratio >= 0.5 or any(t in short_garbage for t in toks[:4])
    return hits >= 2 or looks_broken_header

def clean_owner_line(line: str) -> str:
    if not line:
        return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t):
            break
        if t in GEO_TOKENS or "[" in t or "]" in t:
            break
        if t in BAD_TOKENS:
            continue
        if len(t) <= 2 and t not in NAME_CONNECTORS:
            continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:48]

def pick_owner_from_l1(raw: str) -> Tuple[str, str]:
    """
    Devuelve (owner, extra_from_break) usando SOLO la banda de la 1ª línea (L1).
    - Si el primer renglón parece encabezado (incluso roto), lo saltamos.
    - Si el primer renglón parece nombre y hay un 2º renglón dentro de L1 que
      también parece nombre (≤26 chars, sin dígitos, no geo/ruido), lo devolvemos como extra.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", ""

    parts = [p.strip() for p in raw.split("\n") if p.strip()]
    if not parts:
        return "", ""

    if looks_like_header_line(parts[0]):
        for p in parts[1:]:
            cand = clean_owner_line(p.upper())
            if len(cand) >= 8:
                return cand, ""
        return "", ""

    first = parts[0]
    owner = clean_owner_line(first.upper())

    extra = ""
    if len(parts) >= 2:
        s2 = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", parts[1].upper()).strip()
        if s2 and len(s2) <= 26 and not any(ch.isdigit() for ch in s2):
            if s2 not in JUNK_2NDLINE and s2 not in GEO_TOKENS and s2 not in BAD_TOKENS:
                extra = s2

    return owner, extra

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna “Apellidos…” y extraer nombre (L1 + posible extra)
# ──────────────────────────────────────────────────────────────────────────────
def find_header_window(bgr: np.ndarray, row_y: int) -> Tuple[int,int,int,int]:
    """ Devuelve ventana amplia donde buscar cabecera y nombres (x0,y0,x1,y1). """
    h, w = bgr.shape[:2]
    x0 = int(w * 0.52)
    x1 = int(w * 0.90)
    pad = int(h * 0.14)
    y0 = max(0, row_y - pad)
    y1 = min(h, row_y + pad)
    return x0, y0, x1, y1

def detect_header_and_lines(bgr: np.ndarray, row_y: int) -> Tuple[int,int,int,int,List[int],List[int]]:
    """
    Busca cabecera cerca de row_y y devuelve:
    (x0,x1,y0_band,y1_band, [y0_l1,y1_l1], [y0_l2,y1_l2])
    """
    x0, y0, x1, y1 = find_header_window(bgr, row_y)
    region = bgr[y0:y1, x0:x1]
    h, w = bgr.shape[:2]

    if region.size == 0:
        # Fallback: banda estrecha
        bl = int(h * 0.03)
        return x0, max(0, row_y - bl), x1, min(h, row_y + bl), [row_y - bl, row_y], [row_y, row_y + bl]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    def parse_data(img):
        return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")

    header_bottom = None
    x_nif = None
    header_left = None

    for im in (bw, bwi):
        d = parse_data(im)
        words = d.get("text", [])
        xs = d.get("left", [])
        ys = d.get("top", [])
        ws = d.get("width", [])
        hs = d.get("height", [])

        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            s = t.strip()
            U = s.upper()
            glob_x = x0 + int(lx)
            glob_y = y0 + int(ty)
            if looks_like_header_line(s):
                header_bottom = max(header_bottom or 0, glob_y + int(hh))
                header_left = min(header_left or (x0 + w), glob_x)
            if U == "NIF":
                x_nif = glob_x
                header_bottom = max(header_bottom or 0, glob_y + int(hh))

    if header_bottom is None:
        # Fallback sin cabecera clara: banda fija alrededor de row_y
        band_h = int(bgr.shape[0] * 0.06)
        y0b = max(0, row_y - band_h//2)
        y1b = min(bgr.shape[0], y0b + band_h)
        # dos renglones partiendo la banda a la mitad
        mid = (y0b + y1b) // 2
        return x0, y0b, x1, y1b, [y0b, mid], [mid, y1b]

    # Definir banda entre header_bottom y ~una altura corta
    band_top = min(bgr.shape[0]-1, header_bottom + 6)
    band_h  = int(bgr.shape[0] * 0.06)
    band_bot = min(bgr.shape[0], band_top + band_h)
    # Cortes de L1 y L2
    l1_top = band_top
    l1_bot = min(bgr.shape[0], band_top + band_h//2)
    l2_top = l1_bot
    l2_bot = band_bot

    return x0, band_top, (x_nif - 8) if x_nif else x1, band_bot, [l1_top, l1_bot], [l2_top, l2_bot]

def ocr_owner_for_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Extrae el titular para una fila usando L1 (con posible salto de línea interno)
    y, si procede, 2ª línea (L2) limpia.
    """
    x0, y0b, x1, y1b, l1, l2 = detect_header_and_lines(bgr, row_y)
    band = bgr[y0b:y1b, x0:x1]
    dbg = {"band": [x0, y0b, x1, y1b], "y_line1": l1, "y_line2_hint": l2, "x0": x0, "x1": x1}

    if band.size == 0:
        return "", dbg

    # OCR L1 (y L2 si hace falta)
    def read_box(y0, y1):
        box = bgr[y0:y1, x0:x1]
        if box.size == 0:
            return ""
        g = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
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
        # elegir la variante con más letras
        return max(variants, key=lambda t: sum(ch.isalpha() for ch in t or "")) or ""

    t1_raw = read_box(l1[0], l1[1])
    t2_raw = read_box(l2[0], l2[1])

    dbg["t1_raw"] = t1_raw
    dbg["t2_raw"] = t2_raw

    # Primero intenta desde L1 (incluido salto de línea dentro de L1)
    owner, extra = pick_owner_from_l1(t1_raw)
    picked_from = "strict"

    if not owner:
        # Si L1 no dio nada, intenta con L2 (como apoyo muy laxo)
        cand2 = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", (t2_raw or "").upper()).strip()
        if cand2 and len(cand2) <= 26 and not any(ch.isdigit() for ch in cand2):
            if cand2 not in JUNK_2NDLINE and cand2 not in GEO_TOKENS and cand2 not in BAD_TOKENS:
                owner = cand2
                extra = ""
                picked_from = "fallback"

    if owner and extra:
        owner = f"{owner} {extra}"

    return owner.strip(), {**dbg, "picked_from": picked_from}

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas + vecinos + 8 vientos
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    H, W = bgr.shape[:2]

    # Croquis (columna izquierda)
    top = int(H * 0.10); bottom = int(H * 0.92)
    left = int(W * 0.06); right = int(W * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return (
            {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            {"rows": [], "note": "no_main_polygons"},
            vis
        )

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # por fila
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecino más cercano en esa fila
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (W*0.30)**2:
            side = side_of8((mcx, mcy), best)

        owner, ocr_dbg = ocr_owner_for_row(bgr, mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"
                }.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:24], (int(W*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:24], (int(W*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
        })

    dbg = {"rows": rows_dbg}
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
        "FAST_DPI": os.getenv("FAST_DPI", ""),
        "PDF_DPI": os.getenv("PDF_DPI", ""),
        "dpi_effective": get_raster_dpi(),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)},
        "junk_2ndline": sorted(list(JUNK_2NDLINE)),
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr, _dpi = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
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
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True, "dpi_effective": get_raster_dpi()} if debug else None
        )

    try:
        bgr, dpi_eff = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:12]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = {**vdbg, "raster": {"dpi": dpi_eff}} if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )


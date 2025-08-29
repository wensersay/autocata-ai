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
app = FastAPI(title="AutoCatastro AI", version="0.6.1")

# ──────────────────────────────────────────────────────────────────────────────
# Flags / entorno
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE", "1").strip() == "1")

# DPI: PDF_DPI > FAST_DPI > 340
def get_raster_dpi() -> int:
    pdf_dpi  = os.getenv("PDF_DPI")
    fast_dpi = os.getenv("FAST_DPI")
    if pdf_dpi and pdf_dpi.isdigit():
        return int(pdf_dpi)
    if fast_dpi and fast_dpi.isdigit():
        return int(fast_dpi)
    return 340

# Orientación: snap a cardinales y modo estricto
ANGLE_SNAP_DEG    = float(os.getenv("ANGLE_SNAP_DEG", "20"))  # ±20° a cardinal => cardinal
STRICT_CARDINALS  = os.getenv("STRICT_CARDINALS", "0").strip() == "1"

# Filtros de 2ª línea (ruido OCR)
JUNK_2NDLINE = [t.strip().upper() for t in os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",") if t.strip()]

# Seguridad
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
# Utilidades texto
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
    return np.array(pil)[:, :, ::-1]  # BGR

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde (parcela principal)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # Rosa (colindantes)
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

# ──────────────────────────────────────────────────────────────────────────────
# Orientación (8 rumbos + snap a cardinal)
# ──────────────────────────────────────────────────────────────────────────────
def angle_deg(from_xy: Tuple[int,int], to_xy: Tuple[int,int]) -> float:
    (x0, y0), (x1, y1) = from_xy, to_xy
    dx, dy = x1 - x0, y1 - y0
    # Convenio: 0° = Este, 90° = Norte (eje Y hacia abajo -> signo -)
    ang = math.degrees(math.atan2(-(dy), dx))
    return ang

def snap_to_cardinal(ang: float) -> Optional[str]:
    """
    Si el ángulo está cerca (±ANGLE_SNAP_DEG) de un cardinal, devuelvo ese cardinal.
    """
    cand = [
        (0.0,  "este"),
        (90.0, "norte"),
        (180.0,"oeste"),
        (-180.0,"oeste"),  # mismo que 180
        (-90.0,"sur")
    ]
    for a, name in cand:
        if abs((ang - a + 180) % 360 - 180) <= ANGLE_SNAP_DEG:
            return name
    return None

def angle_to_compass(ang: float) -> str:
    """
    8 rumbos: N, NE, E, SE, S, SW, W, NW, con snap previo a cardinales.
    """
    # Snap a cardinal si está cerca
    snapped = snap_to_cardinal(ang)
    if snapped:
        return snapped

    # Bins de 45°
    # Centro de cada bin: E(0), NE(45), N(90), NW(135), W(180/-180), SW(-135), S(-90), SE(-45)
    if -22.5 <= ang <= 22.5:   d = "este"
    elif 22.5 < ang <= 67.5:   d = "noreste"
    elif 67.5 < ang <= 112.5:  d = "norte"
    elif 112.5 < ang <= 157.5: d = "noroeste"
    elif ang > 157.5 or ang < -157.5: d = "oeste"
    elif -157.5 <= ang < -112.5: d = "suroeste"
    elif -112.5 <= ang < -67.5:  d = "sur"
    else:                        d = "sureste"

    if STRICT_CARDINALS:
        # convertir a cardinal más cercano
        return {
            "noreste":"este" if abs(ang) < 67.5 else "norte",
            "noroeste":"oeste" if abs(ang) > 112.5 else "norte",
            "sureste":"este" if abs(ang) < 22.5 or abs(ang) < 67.5 else "sur",
            "suroeste":"oeste" if abs(ang) > 157.5 or abs(ang) > 112.5 else "sur"
        }.get(d, d)
    return d

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    ang = angle_deg(main_xy, pt_xy)
    return angle_to_compass(ang)

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
    # eliminar basura inicial tipo A N R / cabeceras
    junk_heads = {"A","AN","AP","N","R","AR","ANR","APELLIDOS","NOMBRE","RAZON","RAZÓN","SOCIAL"}
    out = []
    started = False
    for t in toks:
        if not started:
            if t in junk_heads: 
                continue
            if any(ch.isdigit() for ch in t): 
                continue
            started = True
        if t in GEO_TOKENS or "[" in t or "]" in t: 
            break
        if t in BAD_TOKENS: 
            continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    # compactar conectores repetidos
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
# Localizar columna “Apellidos…” y extraer 1ª y 2ª línea
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_band(bgr: np.ndarray, row_y: int,
                               x_text0: int, x_text1: int) -> Tuple[int,int,int,int,int,int]:
    """
    Devuelve (x0, x1, y1_top, y1_bot, y2_top, y2_bot) para L1 y L2 del NOMBRE.
    Se busca 'APELLIDOS'/'NIF' en una banda alrededor de row_y. Fallback si no hay.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        lh = int(h * 0.035)
        y1_top = max(0, row_y - int(h*0.01))
        y1_bot = min(h, y1_top + lh)
        y2_top = min(h, y1_bot + 2)
        y2_bot = min(h, y2_top + lh)
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))
        return x0, x1, y1_top, y1_bot, y2_top, y2_bot

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    header_bottom = None
    x_nif_local = None
    header_left_local = None

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: 
                continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
                if header_left_local is None:
                    header_left_local = lx
            if T == "NIF":
                x_nif_local = lx
                header_bottom = max(header_bottom or 0, ty + hh)

    # fallback conservador si no detectó cabecera
    lh = int(h * 0.035)
    if header_bottom is None:
        y1_top = max(0, row_y - int(h*0.01))
        y1_bot = min(h, y1_top + lh)
        y2_top = min(h, y1_bot + 2)
        y2_bot = min(h, y2_top + lh)
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))
        return x0, x1, y1_top, y1_bot, y2_top, y2_bot

    y_abs = y0s + header_bottom + 6
    y1_top = y_abs
    y1_bot = min(h, y1_top + lh)
    y2_top = min(h, y1_bot + 2)
    y2_bot = min(h, y2_top + lh)

    if x_nif_local is not None:
        x0 = x_text0
        x1 = min(x_text1, x_text0 + x_nif_local - 8)
    else:
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))
    return x0, x1, y1_top, y1_bot, y2_top, y2_bot

def ocr_two_lines(bgr: np.ndarray, row_y: int) -> Tuple[str,str,dict]:
    """
    Devuelve (line1_clean, line2_clean, debug_dict)
    - L1: nombre principal (si tesseract mete salto '\n' dentro de L1, usamos la parte posterior como extra).
    - L2: segunda línea, filtrada por JUNK_2NDLINE y sin números/corchetes/':'.
    """
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.29)
    x_text1 = int(w * 0.96)

    x0, x1, y1_top, y1_bot, y2_top, y2_bot = find_header_and_owner_band(bgr, row_y, x_text0, x_text1)

    # OCR L1
    roi1 = bgr[y1_top:y1_bot, x0:x1]
    t1_raw = ""
    t1_extra = ""
    if roi1.size > 0:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g1 = enhance_gray(g1)
        bw1, bwi1 = binarize(g1)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        v1 = [
            ocr_text(bw1, 6, WL),
            ocr_text(bwi1,6, WL),
            ocr_text(bw1, 7, WL),
            ocr_text(bwi1,7, WL),
            ocr_text(bw1, 13, WL),
        ]
        # el primero con contenido
        for txt in v1:
            if txt:
                t1_raw = txt
                break
        # si L1 trae salto, intentar quedarse con la parte trasera (nombre) como extra
        if "\n" in t1_raw:
            parts = [p.strip() for p in t1_raw.split("\n") if p.strip()]
            if len(parts) >= 2:
                # suelen venir primero restos tipo "A N R..." y luego el nombre
                t1_raw, t1_extra = parts[0], parts[-1]

    name1 = clean_owner_line(t1_extra or t1_raw)

    # OCR L2
    roi2 = bgr[y2_top:y2_bot, x0:x1]
    t2_raw = ""
    if roi2.size > 0:
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g2 = enhance_gray(g2)
        bw2, bwi2 = binarize(g2)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        v2 = [
            ocr_text(bw2, 6, WL),
            ocr_text(bwi2,6, WL),
            ocr_text(bw2, 7, WL),
            ocr_text(bwi2,7, WL),
            ocr_text(bw2, 13, WL),
        ]
        for txt in v2:
            if txt:
                t2_raw = txt
                break

    # Limpiar 2ª línea: cortar por dígito, [, ], :, y filtrar junk cortos
    def sanitize_second_line(s: str) -> str:
        s = s.upper().strip()
        s = re.split(r"[\[\]:0-9]", s)[0].strip()
        s = re.sub(r"\s{2,}", " ", s)
        if len(s) <= 3 and s in JUNK_2NDLINE:
            return ""
        if not s or not re.search(r"[A-ZÁÉÍÓÚÜÑ]", s):
            return ""
        return s[:26]

    name2 = sanitize_second_line(t2_raw)

    # Componer final
    owner = name1
    if not owner and name2:
        owner = name2
    elif owner and name2:
        # Añadir si parece continuación del nombre (no duplicar)
        if name2 not in owner and len(owner) + 1 + len(name2) <= 48:
            owner = f"{owner} {name2}"

    dbg = {
        "band":[x0, y1_top - (y1_top - y2_top), x1, y2_bot],  # aprox banda usada
        "y_line1":[y1_top, y1_bot],
        "y_line2_hint":[y2_top, y2_bot],
        "x0":x0, "x1":x1,
        "t1_raw":t1_raw,
        "t1_extra_raw":t1_extra,
        "t2_raw":t2_raw
    }
    return owner, name2, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Zona izquierda (croquis)
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return (
            {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            {"rows": [], "raster":{"dpi":get_raster_dpi()}},
            vis
        )

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # de arriba a abajo
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_keys = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino rosa más cercano
        best = None; best_d = 1e18
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.30)**2:
            side = side_of((mcx, mcy), best)

        owner, name2, odbg = ocr_two_lines(bgr, mcy)

        if side:
            # Si ese lado ya está ocupado y este nombre parece más largo/útil, podemos reemplazar
            if side not in used_keys or (owner and len(owner) > len(linderos.get(side,""))):
                linderos[side] = owner
                used_keys.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl_map = {"norte":"N","noreste":"NE","este":"E","sureste":"SE","sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"}
                lbl = lbl_map.get(side,"")
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
            "ocr": odbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi":get_raster_dpi()}}
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
        "PDF_DPI": os.getenv("PDF_DPI"),
        "FAST_DPI": os.getenv("FAST_DPI"),
        "raster_dpi_effective": get_raster_dpi(),
        "ANGLE_SNAP_DEG": ANGLE_SNAP_DEG,
        "STRICT_CARDINALS": STRICT_CARDINALS
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar puntos cardinales"),
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
        cv2.putText(blank, f"ERR: {err[:80]}", (10,140),
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
    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )






from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import numpy as np
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.4.1")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# Knobs de layout (ajustables sin redeploy)
OWNER_X0_FRAC = float(os.getenv("OWNER_X0_FRAC", "0.38"))  # inicio columna "Apellidos…"
OWNER_X1_FRAC = float(os.getenv("OWNER_X1_FRAC", "0.70"))  # fin columna "Apellidos…"
ROW_GAP_PX    = int(os.getenv("ROW_GAP_PX", "260"))        # separador entre filas (img px @ ~400dpi)
HEADER_WINDOW = int(os.getenv("HEADER_WINDOW", "12"))      # ventana de líneas para fallback textual

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
# Utilidades PDF / Texto
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
STOP_IN_NAME = (
    "POLÍGONO","POLIGONO","PARCELA","COORDENADAS","ETRS","HUSO","ESCALA",
    "TITULARIDAD","VALOR CATASTRAL","LOCALIZACIÓN","LOCALIZACION","REFERENCIA CATASTRAL",
    "APELLIDOS NOMBRE","RAZON SOCIAL","NIF","DOMICILIO","[","]","(",")"
)

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

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def is_upper_name(line: str, maxlen: int = 26) -> bool:
    line = line.strip()
    if not line or len(line) > maxlen:
        return False
    U = line.upper()
    for bad in STOP_IN_NAME:
        if bad in U:
            return False
    if any(ch.isdigit() for ch in line):
        return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner(lines: List[str]) -> str:
    """Une hasta 2 líneas en mayúsculas en un nombre razonable."""
    picked = [re.sub(r"\s+"," ",ln.strip()) for ln in lines if ln.strip()]
    name = " ".join(picked)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

# ──────────────────────────────────────────────────────────────────────────────
# OpenCV helpers (con fallbacks)
# ──────────────────────────────────────────────────────────────────────────────
def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))
THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Rasterización / color
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 400 if FAST_MODE else 550
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    return np.array(pages[0].convert("RGB"))[:, :, ::-1]  # RGB→BGR

def crop_map(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    h, w = bgr.shape[:2]
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.08); right = int(w * 0.92)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    if bottom - top < 100 or right - left < 100:
        return bgr, (0, 0)
    return bgr[top:bottom, left:right], (left, top)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [ (np.array([35,20,50],np.uint8), np.array([85,255,255],np.uint8)),
                 (np.array([86,15,50],np.uint8), np.array([100,255,255],np.uint8)) ]
    p_ranges = [ (np.array([160,20,80],np.uint8), np.array([179,255,255],np.uint8)),
                 (np.array([  0,20,80],np.uint8), np.array([ 10,255,255],np.uint8)) ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int = 200) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy; x, y = pt_xy
    ang = math.degrees(math.atan2(-(y-cy), x-cx))  # Norte arriba
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

def ocr_digits(img: np.ndarray, psm: int = 7) -> str:
    cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_string(img, config=cfg) or ""
    return re.sub(r"\D+", "", data)

def read_parcel_number_at(bgr: np.ndarray, center: Tuple[int,int], box: int = 110) -> str:
    x, y = center; h, w = bgr.shape[:2]; half = box // 2
    x0, y0 = max(0, x-half), max(0, y-half)
    x1, y1 = min(w, x+half), min(h, y+half)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0: return ""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
    flags_inv = THRESH_BINARY_INV | (THRESH_OTSU if THRESH_OTSU else 0)
    _, bw  = cv2.threshold(g, 0 if THRESH_OTSU else 127, 255, flags_bin)
    _, bwi = cv2.threshold(g, 0 if THRESH_OTSU else 127, 255, flags_inv)
    for p in (7,6):
        for var in (bw,bwi):
            txt = ocr_digits(var, psm=p)
            if txt: return txt
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Segmentación por filas (imagen) + ROI de texto por fila
# ──────────────────────────────────────────────────────────────────────────────
def segment_rows_and_sides(bgr: np.ndarray) -> Tuple[List[dict], np.ndarray]:
    """Devuelve filas con centros (verde/rojo), lado, y ROI texto en coords IMG."""
    vis = bgr.copy()
    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    greens = [(x+ox,y+oy,a) for (x,y,a) in contours_centroids(mg, 200)]
    reds   = [(x+ox,y+oy,a) for (x,y,a) in contours_centroids(mp, 180)]

    # Agrupar rojos por Y en filas
    reds_sorted = sorted(reds, key=lambda t: t[1])
    rows: List[List[Tuple[int,int,int]]] = []
    for pt in reds_sorted:
        if not rows or abs(pt[1]-rows[-1][-1][1]) > ROW_GAP_PX:
            rows.append([pt])
        else:
            rows[-1].append(pt)

    H, W = bgr.shape[:2]
    x0_owner = int(W * OWNER_X0_FRAC)
    x1_owner = int(W * OWNER_X1_FRAC)

    rows_info: List[dict] = []
    for idx, cluster in enumerate(rows):
        ys = [p[1] for p in cluster]; y_min, y_max = min(ys), max(ys)
        pad = 120
        r_y0 = max(0, y_min - pad); r_y1 = min(H, y_max + pad)

        # main = verde más grande próximo a esta banda vertical
        g_band = [g for g in greens if r_y0 <= g[1] <= r_y1]
        main_pt = g_band[0] if g_band else None

        # vecino = rojo del cluster más cercano al main; si no hay main, coge el central del cluster
        if main_pt:
            neigh_pt = min(cluster, key=lambda p: (p[0]-main_pt[0])**2 + (p[1]-main_pt[1])**2)
            side = side_of((main_pt[0],main_pt[1]), (neigh_pt[0],neigh_pt[1]))
        else:
            neigh_pt = cluster[len(cluster)//2]
            side = ""

        rows_info.append({
            "row_index": idx,
            "main_center": (int(main_pt[0]), int(main_pt[1])) if main_pt else None,
            "neigh_center": (int(neigh_pt[0]), int(neigh_pt[1])) if neigh_pt else None,
            "side": side,
            "img_owner_roi": [x0_owner, int(r_y0), x1_owner, int(r_y1)],  # [x0,y0,x1,y1] en IMG
        })

        # Visual
        if main_pt:
            cv2.circle(vis, (main_pt[0],main_pt[1]), 10, (0,255,0), -1)
        if neigh_pt:
            cv2.circle(vis, (neigh_pt[0],neigh_pt[1]), 8, (0,0,255), -1)
        if side:
            cv2.putText(vis, side[:1].upper(), (neigh_pt[0]+8,neigh_pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    return rows_info, vis

# ──────────────────────────────────────────────────────────────────────────────
# Texto posicional por ROI (pdfplumber.words → líneas → nombre)
# ──────────────────────────────────────────────────────────────────────────────
def words_in_bbox(page: pdfplumber.page.Page, bbox_pdf: Tuple[float,float,float,float]) -> List[dict]:
    x0,y0,x1,y1 = bbox_pdf
    words = page.extract_words(keep_blank_chars=False, use_text_flow=True)
    out = []
    for w in words:
        if (w["x0"] >= x0 and w["x1"] <= x1 and w["top"] >= y0 and w["bottom"] <= y1):
            out.append(w)
    return out

def lines_from_words(words: List[dict], y_tol: float = 3.0) -> List[str]:
    if not words: return []
    # agrupar por bandas horizontales
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[List[dict]] = []
    for w in words:
        if not lines or abs(w["top"] - lines[-1][-1]["top"]) > y_tol:
            lines.append([w])
        else:
            lines[-1].append(w)
    out = []
    for ln in lines:
        txt = " ".join([w["text"] for w in ln])
        out.append(txt.strip())
    return out

def owner_from_roi(page: pdfplumber.page.Page,
                   img_roi: List[int],
                   img_size: Tuple[int,int]) -> str:
    """Convierte ROI de IMG → PDF, extrae líneas y aplica reglas ≤26, MAYÚSC., sin dígitos."""
    W_img, H_img = img_size
    x0_i, y0_i, x1_i, y1_i = img_roi

    # Map a coords PDF (escala lineal)
    W_pdf, H_pdf = float(page.width), float(page.height)
    sx, sy = W_pdf / float(W_img), H_pdf / float(H_img)
    x0 = x0_i * sx; x1 = x1_i * sx; y0 = y0_i * sy; y1 = y1_i * sy
    bbox_pdf = (x0, y0, x1, y1)

    # Palabras → líneas
    words = words_in_bbox(page, bbox_pdf)
    lines = lines_from_words(words, y_tol=3.5)

    # Filtrado por reglas
    candidates = [ln for ln in lines if is_upper_name(ln, maxlen=26)]
    if not candidates:
        # Fallback muy corto: busca dos seguidas válidas dentro de HEADER_WINDOW líneas
        short = []
        for ln in lines[:HEADER_WINDOW]:
            if is_upper_name(ln, maxlen=26):
                short.append(ln)
                if len(short) == 2: break
        candidates = [" ".join(short)] if short else []

    if candidates:
        # preferir la más larga pero ≤26
        best = sorted(candidates, key=lambda s: len(s), reverse=True)[0]
        # segunda línea opcional si cabe y es válida
        extra = ""
        for ln in lines:
            if ln == best: continue
            if is_upper_name(ln, maxlen=26) and len(best) + 1 + len(ln) <= 26:
                extra = ln; break
        return reconstruct_owner([best] + ([extra] if extra else []))
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Detección completa por filas (imagen + texto)
# ──────────────────────────────────────────────────────────────────────────────
def detect_by_rows_and_text(pdf_bytes: bytes) -> Tuple[Dict[str,str], dict, np.ndarray]:
    bgr = page2_bgr(pdf_bytes)
    rows_info, vis = segment_rows_and_sides(bgr)

    # Abrir página 2 para texto posicional
    page2 = None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) >= 2:
            page2 = pdf.pages[1]

        owners_rows = []
        linderos = {"norte":"","sur":"","este":"","oeste":""}

        for row in rows_info:
            owner = ""
            if page2:
                owner = owner_from_roi(page2, row["img_owner_roi"], (bgr.shape[1], bgr.shape[0]))
            row["owner"] = owner
            owners_rows.append(owner)
            if row.get("side") and owner:
                # Asignar



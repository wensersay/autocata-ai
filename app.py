from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse, JSONResponse
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
app = FastAPI(title="AutoCatastro AI", version="0.4.3")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

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
    "POLÍGONO", "POLIGONO", "PARCELA", "LOCALIZACIÓN", "LOCALIZACION",
    "REFERENCIA CATASTRAL", "COORDENADAS", "ETRS", "HUSO", "ESCALA",
    "TITULARIDAD", "VALOR CATASTRAL", "NIF", "DOMICILIO", "APELLIDOS NOMBRE",
    "RAZON SOCIAL", "RAZÓN SOCIAL"
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

def is_upper_name(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    U = line.upper()
    if any(bad in U for bad in STOP_IN_NAME):
        return False
    # demasiados números → probablemente NIF/dirección
    if sum(ch.isdigit() for ch in line) >= 2:
        return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner(lines: List[str]) -> str:
    """Une 1–2 líneas MAYÚSCULAS en un nombre. Limita a ≈ 32–36 chars."""
    joined = " ".join([re.sub(r"\s+", " ", L.strip()) for L in lines if L.strip()])
    joined = re.sub(r"\s{2,}", " ", joined).strip()
    # regla de 26–36 caracteres (evita domicilios largos)
    if len(joined) > 36:
        # si es muy largo, quédate con primeras ~32 sin cortar palabra
        parts = joined.split(" ")
        acc = []
        for p in parts:
            if len(" ".join(acc + [p])) > 32:
                break
            acc.append(p)
        joined = " ".join(acc) if acc else joined[:32]
    return joined

# ──────────────────────────────────────────────────────────────────────────────
# OpenCV helpers y constantes seguras (evita crash si falta OTSU en build)
# ──────────────────────────────────────────────────────────────────────────────
def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Visión por computador (página 2): filas + N/S/E/O
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 400 if FAST_MODE else 550
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def crop_map_band(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Recorta banda donde viven croquis y columnas (salta cabecera/pie)."""
    h, w = bgr.shape[:2]
    top = int(h * 0.16); bottom = int(h * 0.96)
    left = int(w * 0.05); right = int(w * 0.95)
    return bgr[top:bottom, left:right], (left, top)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde de parcela propia
    g_ranges = [
        (np.array([35,  20, 40], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 40], np.uint8), np.array([100,255, 255], np.uint8)),
    ]
    # Rosa de vecino
    p_ranges = [
        (np.array([160, 20, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
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
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda t: t[1])  # de arriba a abajo
    return out

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # norte=arriba
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

def detect_rows_with_sides(bgr: np.ndarray) -> List[dict]:
    """
    Devuelve lista de filas (ordenadas de arriba a abajo):
    { main:(x,y), neigh:(x,y)|None, side:'norte'|'sur'|'este'|'oeste'|'' }
    """
    band, (ox, oy) = crop_map_band(bgr)
    mg, mp = color_masks(band)
    greens = contours_centroids(mg, min_area=(260 if FAST_MODE else 200))
    pinks  = contours_centroids(mp, min_area=(220 if FAST_MODE else 180))

    rows: List[dict] = []
    used = set()

    for gx, gy, _ in greens:
        gabs = (gx+ox, gy+oy)
        # vecino rosa más cercano por distancia vertical
        best = None; bestd = 1e9; idx = -1
        for i, (px, py, _a) in enumerate(pinks):
            if i in used: continue
            d = abs(py - gy) + 0.3*abs(px - gx)
            if d < bestd:
                bestd = d; best = (px+ox, py+oy); idx = i
        side = ""
        if best is not None:
            side = side_of(gabs, best)
            used.add(idx)
        rows.append({"main": gabs, "neigh": best, "side": side})

    # si hay más rosas sueltas, añádelas como filas huérfanas (raro)
    for i, (px, py, _a) in enumerate(pinks):
        if i in used: continue
        rows.append({"main": None, "neigh": (px+ox, py+oy), "side": ""})

    rows.sort(key=lambda r: (r["main"][1] if r["main"] else r["neigh"][1]))
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Extracción por COLUMNA (debajo de “Apellidos Nombre / Razón social”)
# ──────────────────────────────────────────────────────────────────────────────
def owner_from_column(page: pdfplumber.page.Page,
                      center_y_pdf: float,
                      window_pdf: float,
                      left_ratio: float = 0.34,
                      right_ratio: float = 0.96) -> str:
    """
    Recorta una franja horizontal alrededor de center_y_pdf en la columna derecha
    y extrae el titular justo debajo del encabezado “APELLIDOS NOMBRE / …”.
    """
    W, H = page.width, page.height
    y0 = max(0, center_y_pdf - window_pdf)
    y1 = min(H, center_y_pdf + window_pdf)
    x0 = W * left_ratio
    x1 = W * right_ratio

    crop = page.crop((x0, y0, x1, y1))
    text = crop.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
    lines = normalize_text(text).split("\n")
    if not lines:
        return ""

    # 1) Localiza cabecera “APELLIDOS NOMBRE …”
    hdr_idx = None
    for i, ln in enumerate(lines):
        U = ln.upper()
        if "APELLIDOS NOMBRE" in U and ("RAZON" in U or "RAZÓN" in U):
            hdr_idx = i
            break

    start = (hdr_idx + 1) if hdr_idx is not None else 0
    # 2) Toma 1–2 líneas válidas en mayúsculas (sin números) como “owner”
    picked: List[str] = []
    steps = 0
    for j in range(start, min(len(lines), start + 8)):
        cand = lines[j].strip()
        U = cand.upper()
        if not cand:
            if picked: break
            continue
        if "NIF" in U or "DOMICILIO" in U:
            break
        if is_upper_name(cand):
            picked.append(U)
            if len(picked) >= 2:
                break
        steps += 1

    owner = reconstruct_owner(picked) if picked else ""
    return owner

# ──────────────────────────────────────────────────────────────────────────────
# OCR auxiliar para números de parcela (si hiciera falta en el futuro)
# ──────────────────────────────────────────────────────────────────────────────
def ocr_digits(img: np.ndarray, psm: int = 7) -> str:
    cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_string(img, config=cfg) or ""
    digits = re.sub(r"\D+", "", data)
    return digits

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True, "version": app.version,
        "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY,
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...), labels: int = Query(0)):
    """
    labels=0 → solo puntos; 1 → N/S/E/O; 2 → N/S/E/O + owner abreviado (si hay)
    """
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    bgr = page2_bgr(pdf_bytes)
    rows = detect_rows_with_sides(bgr)

    owners_by_row: Dict[int, str] = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[1]  # pág.2 (index 1)
            Himg = bgr.shape[0]
            Hpdf = page.height
            band = Hpdf * 0.06  # ventana vertical ≈ 6% de la página

            for idx, r in enumerate(rows):
                cy_img = (r["main"][1] if r["main"] else (r["neigh"][1] if r["neigh"] else None))
                if cy_img is None: 
                    owners_by_row[idx] = ""
                    continue
                cy_pdf = (cy_img / Himg) * Hpdf
                owner = owner_from_column(page, cy_pdf, band)
                owners_by_row[idx] = owner
    except Exception:
        owners_by_row = {}

    vis = bgr.copy()
    for i, r in enumerate(rows):
        if r["main"]:
            cv2.circle(vis, r["main"], 12, (0,255,0), -1)
        if r["neigh"]:
            cv2.circle(vis, r["neigh"], 10, (0,0,255), -1)
            if labels >= 1 and r["side"]:
                label = {"norte":"N","sur":"S","este":"E","oeste":"O"}[r["side"]]
                cv2.putText(vis, label, (r["neigh"][0]+6, r["neigh"][1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
            if labels >= 2:
                short = (owners_by_row.get(i) or "")[:18]
                if short:
                    cv2.putText(vis, short, (r["neigh"][0]+6, r["neigh"][1]+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...), labels: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))
    if TEXT_ONLY:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = normalize_text(pdf.pages[1].extract_text() or "")
        owners_detected = []
        for line in text.split("\n"):
            if is_upper_name(line):
                owners_detected.append(line.strip().upper())
        owners_detected = list(dict.fromkeys(owners_detected))[:8]
        return ExtractOut(linderos={"norte":"","sur":"","este":"","oeste":""},
                          owners_detected=owners_detected,
                          note="Modo TEXT_ONLY activo.",
                          debug={"TEXT_ONLY": True} if debug else None)

    # 1) Detectar filas y lados en la imagen
    bgr = page2_bgr(pdf_bytes)
    rows = detect_rows_with_sides(bgr)

    # 2) Extraer por COLUMNA el titular para cada fila
    owners_by_row: Dict[int, str] = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[1]
            Himg = bgr.shape[0]
            Hpdf = page.height
            band = Hpdf * 0.06  # ventana ≈ 6% de la altura
            for idx, r in enumerate(rows):
                cy_img = (r["main"][1] if r["main"] else (r["neigh"][1] if r["neigh"] else None))
                if cy_img is None:
                    owners_by_row[idx] = ""
                    continue
                cy_pdf = (cy_img / Himg) * Hpdf
                owners_by_row[idx] = owner_from_column(page, cy_pdf, band)
    except Exception as e:
        owners_by_row = {}

    # 3) Armar linderos (lado -> titular) a partir de filas
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    for idx, r in enumerate(rows):
        sd = r.get("side") or ""
        if not sd or linderos.get(sd): 
            continue
        owner = owners_by_row.get(idx, "")
        if





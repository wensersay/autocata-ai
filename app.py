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
app = FastAPI(title="AutoCatastro AI", version="0.3.3")

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
DNI_RE        = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE= re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)
POLY_PARC_RE  = re.compile(r"POL[ÍI]GONO\s+\d+.*?PARCELA\s+(\d{1,5})", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION",
    "REFERENCIA CATASTRAL", "RELACIÓN DE PARCELAS", "RELACION DE PARCELAS"
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
    for bad in STOP_IN_NAME:
        if bad in U:
            return False
    if sum(ch.isdigit() for ch in line) >= 3:
        return False
    return bool(UPPER_NAME_RE.match(line))

def clean_tokens_to_name(text: str) -> str:
    # Mantiene solo tokens alfabéticos en mayúsculas razonables
    toks = [t for t in re.findall(r"[A-ZÁÉÍÓÚÜÑ']{2,}", text.upper())]
    # Evita conectores sueltos
    bad = {"DE","DEL","LA","LOS","LAS","Y"}
    toks = [t for t in toks if t not in bad]
    name = " ".join(toks)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def probable_second_name_line(s: str) -> bool:
    # Una línea de 1 palabra (o 2 muy cortas) tipo "LUIS", sin dígitos ni metadatos
    if not s or any(ch.isdigit() for ch in s):
        return False
    U = s.upper().strip()
    for bad in STOP_IN_NAME:
        if bad in U:
            return False
    toks = [t for t in re.findall(r"[A-ZÁÉÍÓÚÜÑ']+", U)]
    if len(toks) == 1 and 3 <= len(toks[0]) <= 6:
        return True
    if len(toks) == 2 and all(2 <= len(t) <= 5 for t in toks):
        return True
    return False

def extract_owners_map(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Construye dict {parcela: titular} desde páginas ≥2.

    Estrategia híbrida:
    A) Método tabla (si existiera "TITULARIDAD PRINCIPAL" / cabecera)
       → toma 1–3 líneas en mayúsculas saltando NIF/DOMICILIO.
    B) Fallback robusto sin cabeceras:
       tras 'Polígono ... Parcela N' escanea una ventana (10–12 líneas):
       - Si una línea tiene dígitos, toma el tramo anterior a los dígitos como nombre.
       - Si la línea anterior a los dígitos era nombre puro (sin dígitos), úsala y,
         si la siguiente es una sola palabra tipo "LUIS", la añade.
    """
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, page in enumerate(pdf.pages):
            if pi == 0:
                continue  # saltar portada
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")

            i = 0
            while i < len(lines):
                raw = lines[i].strip()
                up  = raw.upper()

                # Detecta "Polígono ... Parcela N" o "Parcela N"
                m_poly = re.search(POLY_PARC_RE, up)
                m_only = re.search(PARCEL_ONLY_RE, up)
                parcel = None
                if m_poly:
                    parcel = m_poly.group(1)
                elif m_only:
                    parcel = m_only.group(1)

                if not parcel:
                    i += 1
                    continue

                # ---- Método A: si vemos cabeceras de tabla cerca, aplicarlas
                found_owner = ""
                for k in range(i+1, min(i+12, len(lines))):
                    U = lines[k].upper()
                    if ("TITULARIDAD PRINCIPAL" in U) or ("APELLIDOS NOMBRE" in U and "RAZON" in U):
                        # recoger 1–3 líneas en mayúsculas, saltando NIF/DOMICILIO/números
                        j = k + 1
                        picked: List[str] = []
                        steps = 0
                        while j < len(lines) and steps < 10 and len(picked) < 3:
                            cand = lines[j].strip()
                            UU   = cand.upper()
                            if any(b in UU for b in ("APELLIDOS NOMBRE","RAZON SOCIAL","NIF","DOMICILIO")):
                                j += 1; steps += 1; continue
                            if is_upper_name(cand):
                                picked.append(cand)
                            else:
                                # si ya llevamos algo y esta no cuadra, cerramos
                                if picked:
                                    break
                            j += 1; steps += 1
                        found_owner = clean_tokens_to_name(" ".join(picked)) if picked else ""
                        break

                # ---- Método B: sin cabecera → ventana y recorte antes de dígitos
                if not found_owner:
                    j = i + 1
                    end = min(i + (12 if FAST_MODE else 16), len(lines))
                    while j < end:
                        s = lines[j].strip()
                        if not s:
                            j += 1; continue
                        # Si la línea contiene dígitos, cortamos en el primer dígito
                        if any(ch.isdigit() for ch in s):
                            head = re.split(r"\d", s, 1)[0]
                            cand = clean_tokens_to_name(head)
                            # caso “RODRIGUEZ ALVAREZ JOSE” (sin dígitos) justo antes
                            prev = lines[j-1].strip() if j-1 >= 0 else ""
                            if not cand and prev and is_upper_name(prev):
                                cand = clean_tokens_to_name(prev)
                                # posible "LUIS" en la siguiente
                                nxt = lines[j+1].strip() if j+1 < len(lines) else ""
                                if probable_second_name_line(nxt):
                                    cand = (cand + " " + clean_tokens_to_name(nxt)).strip()
                            # posible segunda línea “LUIS” si la inmediatamente siguiente es muy corta
                            else:
                                nxt = lines[j+1].strip() if j+1 < len(lines) else ""
                                if probable_second_name_line(nxt):
                                    cand = (cand + " " + clean_tokens_to_name(nxt)).strip()

                            found_owner = cand
                            break
                        j += 1

                if parcel and found_owner and parcel not in mapping:
                    mapping[parcel] = found_owner

                i += 1

    return mapping

# ──────────────────────────────────────────────────────────────────────────────
# OpenCV helpers (fallbacks seguros)
# ──────────────────────────────────────────────────────────────────────────────
def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Visión por computador (página 2)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 400 if FAST_MODE else 550
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

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
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160, 20, 80], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 80], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int = 250) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # Norte arriba
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

def ocr_digits(img: np.ndarray, psm: int = 7) -> str:
    cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_string(img, config=cfg) or ""
    return re.sub(r"\D+", "", data)

def read_parcel_number_at(bgr: np.ndarray, center: Tuple[int,int], box: int = 110) -> str:
    x, y = center
    h, w = bgr.shape[:2]
    half = box // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(w, x + half), min(h, y + half)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    flags_bin = (cv2.THRESH_BINARY if hasattr(cv2, "THRESH_BINARY") else 0) | (cv2.THRESH_OTSU if hasattr(cv2, "THRESH_OTSU") else 0)
    flags_inv = (cv2.THRESH_BINARY_INV if hasattr(cv2, "THRESH_BINARY_INV") else 0) | (cv2.THRESH_OTSU if hasattr(cv2, "THRESH_OTSU") else 0)
    _, bw  = cv2.threshold(g, 0 if hasattr(cv2, "THRESH_OTSU") else 127, 255, flags_bin)
    _, bwi = cv2.threshold(g, 0 if hasattr(cv2, "THRESH_OTSU") else 127, 255, flags_inv)
    for p in (7, 6):
        for var in (bw, bwi):
            txt = ocr_digits(var, psm=p)
            if txt: return txt
    return ""

def detect_neighbors_and_assign(bgr: np.ndarray,
                                parcel2owner: Dict[str,str]) -> Tuple[Dict[str,str], dict, np.ndarray]:
    """Devuelve (linderos, debug, annotated_png_bgr)."""
    vis = bgr.copy()
    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    mains = contours_centroids(mg, min_area=(400 if FAST_MODE else 250))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"reason":"no_main_green"}, vis
    main_cx, main_cy, _ = mains[0]
    main_abs = (main_cx + ox, main_cy + oy)
    cv2.circle(vis, main_abs, 10, (0,255,0), -1)

    neighs = contours_centroids(mp, min_area=(280 if FAST_MODE else 180))
    side2parcel: Dict[str, str] = {}
    max_neigh = 24 if FAST_MODE else 48

    for (cx, cy, _a) in neighs[:max_neigh]:
        abs_pt = (cx + ox, cy + oy)
        cv2.circle(vis, abs_pt, 8, (0,0,255), -1)
        sd = side_of(main_abs, abs_pt)
        num = read_parcel_number_at(bgr, abs_pt, box=(90 if FAST_MODE else 120))
        if num and sd not in side2parcel:
            side2parcel[sd] = num
            cv2.putText(vis, f"{sd[:1].upper()}:{num}", (abs_pt[0]+6, abs_pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    linderos = {"norte":"","sur":"","este":"","oeste":""}
    for sd, num in side2parcel.items():
        owner = parcel2owner.get(num, "")
        if owner:
            linderos[sd] = owner

    dbg = {
        "main_center": main_abs,
        "neighbors": len(neighs),
        "side2parcel": side2parcel,
    }
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "version": app.version, "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY}

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...)):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        parcel2owner = extract_owners_map(pdf_bytes)
        bgr = page2_bgr(pdf_bytes)
        linderos, dbg, vis = detect_neighbors_and_assign(bgr, parcel2owner)
    except Exception as e:
        # Devolver mini-imagen con el error (evita 5xx)
        err = str(e)
        blank = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:60]}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...)):
    return preview_get(pdf_url=data.pdf_url)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto (páginas ≥2): parcela → titular
    parcel2owner = extract_owners_map(pdf_bytes)

    if TEXT_ONLY:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = "Modo TEXT_ONLY activo: mapa desactivado."
        dbg = {"TEXT_ONLY": True, "owners_by_parcel_sample": dict(list(parcel2owner.items())[:6])} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected, note=note, debug=dbg)

    # 2) Visión (pág. 2)
    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_neighbors_and_assign(bgr, parcel2owner)
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = None if any(linderos.values()) else "OCR sin coincidencias claras; afinaremos ROI y mapeo de titulares."
        dbg = {"owners_by_parcel_sample": dict(list(parcel2owner.items())[:8]), **vdbg} if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)
    except Exception as e:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = f"Excepción visión/OCR: {e}"
        dbg = {"exception": str(e)} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected, note=note, debug=dbg)



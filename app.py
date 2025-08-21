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
UPPER_NAME_RE  = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
ONLY_LETTERS_RE= re.compile(r"^[A-ZÁÉÍÓÚÜÑ\-\.'´`]+$", re.UNICODE)
DNI_RE         = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE = re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION",
    "RELACIÓN DE PARCELAS", "RELACION DE PARCELAS",
    "REFERENCIA CATASTRAL"
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
    # si hay demasiados dígitos, probablemente es dirección
    if sum(ch.isdigit() for ch in line) >= 3:
        return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner_from_tokens(tokens: List[str]) -> str:
    """Limpia y une tokens (permite compuestos: DE/DEL/DA/DO/Y/SAN/SANTA)."""
    ALLOW = {"DE","DEL","DA","DO","DOS","DAS","Y","E","SAN","SANTA","D'","DI","DU"}
    clean: List[str] = []
    for t in tokens:
        t = t.strip(".´`'")
        if not t: 
            continue
        if ONLY_LETTERS_RE.match(t):
            clean.append(t)
        else:
            # ejemplo: "D'ALMEIDA" -> mantener letras y apóstrofo inicial
            t2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'´`]", "", t)
            if t2 and ONLY_LETTERS_RE.match(t2):
                clean.append(t2)
    # compacta conectores permitidos si aparecen aislados
    out: List[str] = []
    for t in clean:
        if t in ALLOW or len(t) >= 2:
            out.append(t)
    name = " ".join(out)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_name_window(lines: List[str], start_idx: int) -> str:
    """
    Desde la cabecera de titularidad/tabla, recorre hasta 12 líneas,
    extrayendo *del comienzo de cada línea* los tokens alfabéticos (A–Z con acentos)
    hasta el primer token 'no-letras' (con dígitos, ':', '[', etc.).
    Esto capta casos como:
        L1: RODRIGUEZ ALVAREZ JOSE 38526627V AV ...
        L2: LUIS Pl:04 Pt:02 ...
    → 'RODRIGUEZ ALVAREZ JOSE LUIS'
    """
    tokens: List[str] = []
    steps = 0
    i = start_idx + 1
    while i < len(lines) and steps < 12 and len(tokens) < 8:
        raw = lines[i].strip()
        U = raw.upper()

        # Cortes duros de sección / fin de bloque nombres
        if any(stop in U for stop in STOP_IN_NAME) and ("TITULARIDAD" not in U and "APELLIDOS NOMBRE" not in U and "RAZON SOCIAL" not in U):
            break

        # Salta líneas claramente de cabecera de tabla o metadatos
        if ("APELLIDOS NOMBRE" in U) or ("RAZON SOCIAL" in U):
            i += 1; steps += 1; continue

        # Toma tokens alfabeticos desde el inicio de la línea hasta el primer token "no-letras"
        line_tokens = []
        for tok in raw.split():
            if ONLY_LETTERS_RE.match(tok.upper()):
                line_tokens.append(tok.upper())
                continue
            # si el primer token no-letras llega, paramos esta línea
            break

        if line_tokens:
            tokens.extend(line_tokens)
        else:
            # si ya veníamos acumulando y esta línea no aporta, cortamos
            if tokens:
                break

        i += 1
        steps += 1

    name = reconstruct_owner_from_tokens(tokens)
    return name

def extract_owners_map(pdf_bytes: bytes) -> Dict[str, str]:
    """
    {parcela: titular} desde páginas ≥ 2.
    • Detecta '... PARCELA N' (suele ir junto a Localización).
    • Al ver 'TITULARIDAD PRINCIPAL' o cabecera de tabla ('APELLIDOS NOMBRE'/'RAZON SOCIAL'),
      abre una ventana de 12 líneas y captura tokens alfabéticos del inicio de cada línea
      hasta tropezar con el primer token no-alfabético (dígitos, 'Pl:', etc.).
    """
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, page in enumerate(pdf.pages):
            if pi == 0:
                continue  # saltar portada
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")

            curr_parcel: Optional[str] = None
            i = 0
            while i < len(lines):
                raw = lines[i].strip()
                U = raw.upper()

                # (1) PARCELA N
                if "PARCELA" in U:
                    nums = [t for t in U.replace(",", " ").split() if t.isdigit()]
                    if nums:
                        curr_parcel = nums[-1]

                # (2) Cabecera de titulares
                if ("TITULARIDAD PRINCIPAL" in U) or (("APELLIDOS NOMBRE" in U) and ("RAZON" in U)):
                    owner = extract_name_window(lines, i)
                    if curr_parcel and owner and curr_parcel not in mapping:
                        mapping[curr_parcel] = owner
                    # avanzar un poco para no re-parsar lo mismo
                    i += 4
                    continue

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
    # Soporta builds sin OTSU
    def thr(img, inv=False):
        flags = (THRESH_BINARY_INV if inv else THRESH_BINARY) | (THRESH_OTSU if THRESH_OTSU else 0)
        _, bw = cv2.threshold(img, 0 if THRESH_OTSU else 127, 255, flags)
        return bw
    variants = [thr(g, inv=False), thr(g, inv=True)]
    for p in (7, 6):
        for var in variants:
            txt = ocr_digits(var, psm=p)
            if txt:
                return txt
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
    return {
        "ok": True, 
        "version": app.version, 
        "FAST_MODE": FAST_MODE, 
        "TEXT_ONLY": TEXT_ONLY,
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...)):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        parcel2owner = extract_owners_map(pdf_bytes)
        bgr = page2_bgr(pdf_bytes)
        linderos, dbg, vis = detect_neighbors_and_assign(bgr, parcel2owner)
    except Exception as e:
        # mini-imagen con error (evita 5xx)
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
def preview_post(data: ExtractIn = Body(...)):
    return preview_get(pdf_url=data.pdf_url)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto (páginas ≥2): parcela → titular (robusto a nombres multilínea)
    parcel2owner = extract_owners_map(pdf_bytes)

    if TEXT_ONLY:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = "Modo TEXT_ONLY activo: mapa desactivado."
        dbg = {"TEXT_ONLY": True, "owners_by_parcel_sample": dict(list(parcel2owner.items())[:8])} if debug else None
        return ExtractOut(
            linderos={"norte":"","sur":"","oeste":"","este":""},
            owners_detected=owners_detected,
            note=note,
            debug=dbg
        )

    # 2) Visión (pág. 2): localizar principal/vecinos y asignar lados → titulares
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
        return ExtractOut(
            linderos={"norte":"","sur":"","oeste":"","este":""},
            owners_detected=owners_detected,
            note=note,
            debug=dbg
        )



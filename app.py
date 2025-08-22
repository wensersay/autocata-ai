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
POLY_PARC_RE  = re.compile(r"POL[ÍI]GONO\s+(\d+).*?PARCELA\s+(\d+)", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION",
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
    if sum(ch.isdigit() for ch in line) >= 3:
        return False
    return bool(UPPER_NAME_RE.match(line))

def take_leading_upper_before_digits(raw: str) -> str:
    """
    Devuelve el tramo INICIAL en mayúsculas ANTES del primer token con dígitos.
    Ej: 'LUIS Pl:04 Pt:02 ...' -> 'LUIS'
    """
    parts = []
    for tok in raw.strip().split():
        if any(ch.isdigit() for ch in tok):
            break
        T = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\.\-']", "", tok.upper())
        if T:
            parts.append(T)
        else:
            # si se coló algo muy raro, paramos el bloque
            break
    return " ".join(parts).strip()

def reconstruct_owner(lines: List[str]) -> str:
    """Une 1–4 líneas en mayúsculas en un nombre razonable."""
    toks: List[str] = []
    for ln in lines:
        ln = take_leading_upper_before_digits(ln)
        ln = re.sub(r"\s+", " ", ln.strip())
        if ln:
            toks.extend(ln.split(" "))
    clean = []
    i = 0
    while i < len(toks):
        tok = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", toks[i])
        if not tok:
            i += 1
            continue
        # pegados típicos (A L VARELA → ALVARELA)
        if len(tok) <= 2 and i + 1 < len(toks):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", toks[i+1])
            if nxt and len(nxt) >= 3:
                clean.append(tok + nxt)
                i += 2
                continue
        clean.append(tok)
        i += 1
    name = " ".join(clean)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_owners_map(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Construye dict {parcela: titular} desde páginas ≥2.
    • Detecta '... Parcela N' (en la línea de Localización).
    • Tras 'Titularidad principal' o la cabecera de tabla, abre una ventana
      de hasta 12 líneas y recoge hasta 4 líneas en MAYÚSCULAS, tomando
      sólo el tramo inicial antes de aparecer dígitos (captura 'LUIS' aunque
      luego vengan 'Pl:04 Pt:02 ...').
    """
    mapping: Dict[str, str] = {}
    META_STOP_PREFIX = ("REFERENCIA CATASTRAL", "RELACIÓN DE PARCELAS", "RELACION DE PARCELAS")
    TABLE_META = ("APELLIDOS NOMBRE", "RAZON SOCIAL", "NIF", "DOMICILIO")

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
                up  = raw.upper()

                # (1) "PARCELA N"
                if "PARCELA" in up:
                    tokens = [t for t in up.replace(",", " ").split() if t.isdigit()]
                    if tokens:
                        curr_parcel = tokens[-1]

                # (2) Cabeceras que anteceden al titular
                if ("TITULARIDAD PRINCIPAL" in up) or ("APELLIDOS NOMBRE" in up and "RAZON" in up):
                    j = i + 1
                    picked: List[str] = []
                    steps = 0
                    while j < len(lines) and steps < 12 and len(picked) < 4:
                        cand = lines[j].strip()
                        U = cand.upper()

                        # Cortes duros si empieza nueva sección
                        if any(U.startswith(k) for k in META_STOP_PREFIX):
                            break

                        # Saltar filas informativas, pero NO romper la ventana
                        if any(k in U for k in TABLE_META):
                            j += 1; steps += 1; continue

                        # Tomar tramo inicial mayúsculas sin dígitos
                        lead = take_leading_upper_before_digits(cand)
                        if lead and is_upper_name(lead):
                            picked.append(lead)

                        j += 1; steps += 1

                    owner = reconstruct_owner(picked) if picked else ""
                    if curr_parcel and owner and curr_parcel not in mapping:
                        mapping[curr_parcel] = owner
                    i = j
                    continue

                i += 1

    return mapping

def guess_self_parcel(pdf_bytes: bytes) -> Optional[str]:
    """Busca en pág. 1 'Polígono ... Parcela N' para deducir tu parcela."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return None
            t = (pdf.pages[0].extract_text(x_tolerance=2, y_tolerance=2) or "")
            m = re.search(r"POL[ÍI]GONO\s+\d+.*?PARCELA\s+(\d+)", t, flags=re.I | re.S)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None

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

def contours_centroids(mask: np.ndarray, min_area: int = 250) -> List[Tuple[int,int,int,Tuple[int,int,int,int]]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a), (x,y,w,h)))
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
    flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
    flags_inv = THRESH_BINARY_INV | (THRESH_OTSU if THRESH_OTSU else 0)
    _, bw  = cv2.threshold(g, 0 if THRESH_OTSU else 127, 255, flags_bin)
    _, bwi = cv2.threshold(g, 0 if THRESH_OTSU else 127, 255, flags_inv)
    for p in (7, 6):
        for var in (bw, bwi):
            txt = ocr_digits(var, psm=p)
            if txt: return txt
    return ""

def find_self_center_by_digits(bgr: np.ndarray, self_num: str) -> Optional[Tuple[int,int]]:
    if not self_num:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    cfg = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    best = None
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if re.sub(r"\D+","", t) == self_num:
            x = int(data["left"][i]); y = int(data["top"][i])
            w = int(data["width"][i]); h = int(data["height"][i])
            cx, cy = x + w//2, y + h//2
            best = (cx, cy)
            break
    return best

def detect_neighbors_and_assign(
    bgr: np.ndarray,
    parcel2owner: Dict[str,str],
    self_num: Optional[str] = None
) -> Tuple[Dict[str,str], dict, np.ndarray]:
    """Devuelve (linderos, debug, annotated_png_bgr)."""
    vis = bgr.copy()
    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    # Centro principal: 1) por número propio si lo vemos; 2) mayor componente verde
    mains = contours_centroids(mg, min_area=(400 if FAST_MODE else 250))
    main_abs = None
    center_mode = "none"
    if self_num:
        c = find_self_center_by_digits(bgr, self_num)
        if c:
            main_abs = c
            center_mode = "self_ocr"
    if main_abs is None and mains:
        main_cx, main_cy, _, _rect = mains[0]
        main_abs = (main_cx + ox, main_cy + oy)
        center_mode = "green_centroid"

    if main_abs is None:
        return {"norte":"","sur":"","este":"","oeste":""}, {"reason":"no_main"}, vis

    cv2.circle(vis, main_abs, 10, (0,255,0), -1)

    # Vecinos rosas
    neighs = contours_centroids(mp, min_area=(280 if FAST_MODE else 180))
    side2parcel: Dict[str, str] = {}

    max_neigh = 24 if FAST_MODE else 48
    for (cx, cy, _a, _rect) in neighs[:max_neigh]:
        abs_pt = (cx + ox, cy + oy)
        cv2.circle(vis, abs_pt, 8, (0,0,255), -1)
        sd = side_of(main_abs, abs_pt)
        num = read_parcel_number_at(bgr, abs_pt, box=(90 if FAST_MODE else 120))
        if num and sd not in side2parcel:
            side2parcel[sd] = num
            cv2.putText(vis, f"{sd[:1].upper()}:{num}", (abs_pt[0]+6, abs_pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    # Mapear a titulares
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    for sd, num in side2parcel.items():
        owner = parcel2owner.get(num, "")
        if owner:
            linderos[sd] = owner

    dbg = {
        "main_center": main_abs,
        "center_mode": center_mode,
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
        self_num = guess_self_parcel(pdf_bytes)
        bgr = page2_bgr(pdf_bytes)
        linderos, dbg, vis = detect_neighbors_and_assign(bgr, parcel2owner, self_num=self_num)
    except Exception as e:
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

    # 2) Visión (pág. 2) + ancla por número propio si lo tenemos
    try:
        self_num = guess_self_parcel(pdf_bytes)
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_neighbors_and_assign(bgr, parcel2owner, self_num=self_num)
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






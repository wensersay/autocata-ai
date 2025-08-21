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
app = FastAPI(title="AutoCatastro AI", version="0.3.4")

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

def leading_upper_segment(line: str) -> str:
    """
    Extrae el segmento inicial en MAYÚSCULAS aun si después hay números/dirección.
    Ej.: 'LUIS Pl:04 Pt:02 ...' -> 'LUIS'
    Devuelve '' si no hay un inicio razonable en mayúsculas.
    """
    s = line.strip()
    if not s:
        return ""
    U = s.upper()
    # Evita líneas de metadatos evidentes
    if any(tag in U for tag in ("APELLIDOS NOMBRE", "RAZON SOCIAL", "NIF", "DOMICILIO")):
        return ""
    # Segmento inicial de letras/espacios/puntos/apóstrofes/guiones
    m = re.match(r"^([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\.\-\' ]{1,})", U)
    if not m:
        return ""
    seg = m.group(1).strip()
    # Cortar si el segmento capta bloques raros
    seg = re.sub(r"\s{2,}", " ", seg).strip()
    # Si el segmento es demasiado corto o parece topónimo puro, descártalo
    if len(seg) < 3:
        return ""
    return seg

def reconstruct_owner(lines: List[str]) -> str:
    """Une 1–4 líneas/segmentos en mayúsculas en un nombre razonable."""
    toks: List[str] = []
    for ln in lines:
        # Preferimos el segmento inicial en mayúsculas aunque haya números después.
        seg = leading_upper_segment(ln)
        if not seg:
            # como respaldo, si toda la línea parece nombre en mayúsculas sin números
            if is_upper_name(ln):
                seg = ln.strip()
        if seg:
            toks.extend(seg.split(" "))
    clean: List[str] = []
    i = 0
    while i < len(toks):
        tok = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", toks[i])
        if not tok:
            i += 1
            continue
        # unir pegados típicos (A L VARELA → ALVARELA)
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
    # Limitar longitud para no comernos la dirección
    return name[:80].strip()

def extract_owners_map(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Construye dict {parcela: titular} desde páginas ≥2.
    • Detecta '... Parcela N' (en la línea de Localización).
    • Tras 'Titularidad principal' o cabecera de tabla, escanea una ventana
      de hasta 12 líneas y recolecta hasta 4 segmentos en MAYÚSCULAS
      (aunque haya NIF/Domicilio intermedios), con cortes por metacabeceras.
    """
    mapping: Dict[str, str] = {}
    META_STOPS = (
        "REFERENCIA CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION",
        "RELACIÓN DE PARCELAS", "RELACION DE PARCELAS", "TITULARIDAD PRINCIPAL"
    )

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

                # (1) Detectar "PARCELA N"
                if "PARCELA" in up:
                    tokens = [t for t in up.replace(",", " ").split() if t.isdigit()]
                    if tokens:
                        curr_parcel = tokens[-1]

                # (2) Cabecera de titularidad / tabla
                if ("TITULARIDAD PRINCIPAL" in up) or ("APELLIDOS NOMBRE" in up and "RAZON" in up):
                    j = i + 1
                    picked: List[str] = []
                    steps = 0
                    while j < len(lines) and steps < 12 and len(picked) < 4:
                        cand = lines[j].strip()
                        U = cand.upper()

                        # Cortes duros si aparecen otras secciones
                        if any(s in U for s in META_STOPS):
                            break
                        # Saltar cabeceras de tabla, pero NO cortar el bucle
                        if ("APELLIDOS NOMBRE" in U) or ("RAZON SOCIAL" in U):
                            j += 1; steps += 1; continue

                        seg = leading_upper_segment(cand)
                        if seg:
                            picked.append(seg)

                        j += 1; steps += 1

                    owner = reconstruct_owner(picked) if picked else ""
                    # sanity: evita meter topónimos típicos
                    if owner and not re.search(r"\b(LUGO|MADRID|BARCELONA|CORUÑA|SEVILLA|VALENCIA|BIZKAIA|ALAVA|GIPUZKOA|ZARAGOZA|MALAGA)\b", owner):
                        if curr_parcel and curr_parcel not in mapping:
                            mapping[curr_parcel] = owner

                    i = j
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
    # Fallback si OTSU no existe en la build
    TH_BIN = int(getattr(cv2, "THRESH_BINARY", 0))
    TH_INV = int(getattr(cv2, "THRESH_BINARY_INV", 0))
    TH_OT  = int(getattr(cv2, "THRESH_OTSU", 0))
    _, bw  = cv2.threshold(g, 0 if TH_OT else 127, 255, TH_BIN | (TH_OT if TH_OT else 0))
    _, bwi = cv2.threshold(g, 0 if TH_OT else 127, 255, TH_INV | (TH_OT if TH_OT else 0))
    # Muestrear también con pequeños desplazamientos para capturar números fuera del centroide del polígono
    variants = [bw, bwi]
    for dx in (0, 20, -20, 35, -35):
        for dy in (0, 20, -20, 35, -35):
            xr0, yr0 = max(0, x0+dx), max(0, y0+dy)
            xr1, yr1 = min(w, x1+dx), min(h, y1+dy)
            if xr1 <= xr0 or yr1 <= yr0: continue
            sub = g[(yr0- y0):(yr1- y0), (xr0- x0):(xr1- x0)]
            if sub.size:
                _, sbw  = cv2.threshold(sub, 0 if TH_OT else 127, 255, TH_BIN | (TH_OT if TH_OT else 0))
                _, sbwi = cv2.threshold(sub, 0 if TH_OT else 127, 255, TH_INV | (TH_OT if TH_OT else 0))
                variants.extend([sbw, sbwi])
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
        "cv2_flags": {"OTSU": bool(getattr(cv2, "THRESH_OTSU", 0))}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...)):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        parcel2owner = extract_owners_map(pdf_bytes)
        bgr = page2_bgr(pdf_bytes)
        linderos, dbg, vis = detect_neighbors_and_assign(bgr, parcel2owner)
    except Exception as e:
        # Devolver mini-imagen con el error (evita 5xx en healthcheck)
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
                          owners_detected=owners_detected,
                          note=note,
                          debug=dbg)

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




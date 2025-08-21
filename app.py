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
app = FastAPI(title="AutoCatastro AI", version="0.3.2")

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
UPPER_NAME_RE   = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE          = re.compile(r"\b\d{8}[A-Z]\b")
POLY_PARC_RE    = re.compile(r"POL[ÍI]GONO\s+(\d+).*?PARCELA\s+(\d+)", re.IGNORECASE)
PARCEL_ONLY_RE  = re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)
NAME_HEAD_RE    = re.compile(r"^([A-ZÁÉÍÓÚÜÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ]{2,}){1,4})\b")
ONE_UPPER_WORD  = re.compile(r"^([A-ZÁÉÍÓÚÜÑ]{2,12})(?:\s|$)")

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION"
)

ADDR_HINTS = (" NIF", "NIF ", " DOMICILIO", " CL ", " AV ", " LG ", " PL:", " PT:", " ESC ", " ES:", " KM ", " C/")

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

def is_meta_line(s: str) -> bool:
    U = s.upper().strip()
    if U == "": return True
    if "APELLIDOS NOMBRE" in U or "RAZON SOCIAL" in U: return True
    if "NIF" in U or "DOMICILIO" in U: return True
    return False

def looks_like_location_line(s: str) -> bool:
    U = s.upper()
    # Localizaciones suelen traer corchetes o puntos entre palabras
    if "[" in U or "]" in U: return True
    if "." in U: return True
    return False

def extract_name_head(line: str) -> Optional[str]:
    """
    Devuelve el 'cabezado' de nombre (2–5 palabras en mayúsculas) al inicio de la línea.
    Ej: 'RODRIGUEZ ALVAREZ JOSE 38526627V ...' -> 'RODRIGUEZ ALVAREZ JOSE'
    """
    m = NAME_HEAD_RE.search(line.upper().strip())
    if not m:
        return None
    head = m.group(1).strip()
    # Evitar que el 'cabezado' sea una localización accidental
    if looks_like_location_line(head):
        return None
    return head

def maybe_append_split_second_name(next_line: str) -> Optional[str]:
    """
    Si la siguiente línea empieza con UNA palabra en mayúsculas (p.ej. 'LUIS')
    y luego ya aparecen dígitos o pinta de dirección, devolvemos esa palabra
    para pegarla al final del nombre.
    """
    U = next_line.upper().strip()
    # No aceptamos si trae corchetes/puntos (suele ser localización)
    if looks_like_location_line(U):
        return None
    m = ONE_UPPER_WORD.match(U)
    if not m:
        return None
    word = m.group(1)
    # ¿El resto de la línea parece dirección o DNI?
    rest = U[len(word):]
    if any(h in rest for h in ADDR_HINTS) or any(ch.isdigit() for ch in rest):
        return word
    # También si la línea es MUY corta (solo esa palabra)
    if rest.strip() == "":
        return word
    return None

def extract_owners_map(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Construye dict {parcela: titular} leyendo las páginas ≥ 2.
    Estrategia robusta frente a localizaciones:
      - Tras 'PARCELA N', inspecciona ~12 líneas.
      - Ignora cabeceras/meta y líneas con '[' ']' o '.' (localización).
      - Extrae el 'cabezado de nombre' (2–5 palabras mayúsculas) y
        si la línea siguiente empieza con UNA palabra en mayúsculas y
        luego ya hay números/dirección, la añade (caso '... JOSE' + 'LUIS Pl:...').
    """
    mapping: Dict[str, str] = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, page in enumerate(pdf.pages):
            if pi == 0:
                continue  # saltamos portada

            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")

            curr_parcel: Optional[str] = None
            i = 0
            while i < len(lines):
                up = lines[i].upper().strip()

                # Detectar PARCELA
                if "PARCELA" in up:
                    # Guardar número de parcela "actual"
                    tokens = [t for t in up.replace(",", " ").split() if t.isdigit()]
                    if tokens:
                        curr_parcel = tokens[-1]

                    # Buscar titular en las ~12 siguientes líneas o hasta próxima PARCELA
                    j = i + 1
                    while j < len(lines) and j <= i + 12:
                        raw = lines[j].strip()
                        U   = raw.upper()

                        if "PARCELA" in U:
                            break  # empieza el siguiente bloque

                        if is_meta_line(raw):
                            j += 1
                            continue

                        if looks_like_location_line(raw):
                            j += 1
                            continue

                        # Intentar extraer el cabezado del nombre en esta línea
                        head = extract_name_head(raw)
                        if head:
                            # ¿La siguiente línea añade un segundo nombre suelto?
                            tail = ""
                            if j + 1 < len(lines):
                                add = maybe_append_split_second_name(lines[j+1])
                                if add:
                                    tail = " " + add
                                    j += 1  # consumimos esa línea

                            owner = (head + tail).strip()
                            if curr_parcel and owner and curr_parcel not in mapping:
                                mapping[curr_parcel] = owner
                            break

                        j += 1

                    i = j
                    continue

                i += 1

    return mapping

# ──────────────────────────────────────────────────────────────────────────────
# Visión por computador (página 2)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 400 if FAST_MODE else 550
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    arr = np.array(pil)[:, :, ::-1]  # RGB→BGR
    return arr

def crop_map(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Recorta márgenes: cabecera/pie y laterales. Devuelve (crop, (ox,oy))."""
    h, w = bgr.shape[:2]
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.08); right = int(w * 0.92)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    if bottom - top < 100 or right - left < 100:
        return bgr, (0, 0)
    return bgr[top:bottom, left:right], (left, top)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (mask_verde_principal, mask_rosa_vecinos)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde agua (parcela propia) – rangos amplios
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255, 255], np.uint8)),
    ]
    # Rosa palo (vecinos) – h≈0 o h≈160
    p_ranges = [
        (np.array([160, 20, 80], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 80], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))

    k3 = np.ones((3,3), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3)
    mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3)
    mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int = 250) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]) ; cy = int(M["m01"] / M["m00"]) ;
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    # Norte arriba, Sur abajo, Este derecha, Oeste izquierda
    ang = math.degrees(math.atan2(-(sy), sx))
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

def ocr_digits(img: np.ndarray, psm: int = 7) -> str:
    cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_string(img, config=cfg) or ""
    digits = re.sub(r"\D+", "", data)
    return digits

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
    _, bw  = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.OTSU)
    _, bwi = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.OTSU)
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

    # Centro principal: mayor componente verde
    mains = contours_centroids(mg, min_area=(400 if FAST_MODE else 250))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"reason":"no_main_green"}, vis
    main_cx, main_cy, _ = mains[0]
    main_abs = (main_cx + ox, main_cy + oy)
    cv2.circle(vis, main_abs, 10, (0,255,0), -1)

    # Vecinos rosas
    neighs = contours_centroids(mp, min_area=(280 if FAST_MODE else 180))
    side2parcel: Dict[str, str] = {}

    for (cx, cy, _a) in neighs[:24 if FAST_MODE else 48]:
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

    # 1) Texto (páginas ≥2): parcela → titular (nuevo parser)
    parcel2owner = extract_owners_map(pdf_bytes)

    if TEXT_ONLY:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = "Modo TEXT_ONLY activo: mapa desactivado."
        dbg = {"TEXT_ONLY": True, "owners_by_parcel_sample": dict(list(parcel2owner.items())[:6])} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected,
                          note=note,
                          debug=dbg)

    # 2) Visión (página 2)
    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_neighbors_and_assign(bgr, parcel2owner)
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = None if any(linderos.values()) else "OCR sin coincidencias claras; afinaremos ROI y mapeo de titulares."
        dbg = ({"owners_by_parcel_sample": dict(list(parcel2owner.items())[:8]), **vdbg} if debug else None)
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)
    except Exception as e:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = f"Excepción visión/OCR: {e}"
        dbg = {"exception": str(e)} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected,
                          note=note,
                          debug=dbg)


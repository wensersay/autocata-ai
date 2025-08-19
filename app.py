from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional, Tuple
import requests, io, re, math, os, subprocess
import pdfplumber
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import pytesseract
from PIL import Image

app = FastAPI(title="AutoCatastro AI", version="0.2.5-RUN1")

# -------- Flags/entorno --------
def env_flag(name: str, default: bool=False) -> bool:
    val = os.getenv(name, "")
    if not val:
        return default
    return val.strip().lower() in {"1","true","yes","on"}

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
FAST_MODE  = env_flag("FAST_MODE",  True)
TEXT_ONLY  = env_flag("TEXT_ONLY",  False)

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ----------------- Modelos -----------------
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None  # si ?debug=1

# ----------------- Utilidades texto PDF -----------------
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE = re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)
POLY_PARC_RE = re.compile(r"POL[ÍI]GONO\s+(\d+).*?PARCELA\s+(\d+)", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION", "REFERENCIA CATASTRAL",
    "USO", "CLASE", "NIF", "DERECHO"
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
    for bad in STOP_IN_NAME:
        if bad in line.upper():
            return False
    if sum(ch.isdigit() for ch in line) >= 3:
        return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner_from_block(lines: List[str]) -> str:
    tokens = []
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln.strip())
        if ln:
            tokens.extend(ln.split(" "))
    clean = []
    i = 0
    while i < len(tokens):
        tok = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", tokens[i])
        if not tok:
            i += 1; continue
        if len(tok) == 1 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 2:
                clean.append(tok + nxt); i += 2; continue
        if len(tok) <= 2 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 3:
                clean.append(tok + nxt); i += 2; continue
        if i + 2 < len(tokens):
            nxt1 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            nxt2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 2])
            if nxt1 and len(nxt1) == 1 and nxt2 and len(nxt2) >= 3:
                clean.append(tok + nxt1 + nxt2); i += 3; continue
        clean.append(tok); i += 1
    name = " ".join(clean)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_owners_by_parcel(pdf_bytes: bytes) -> Dict[str, str]:
    """Construye dict: parcela -> titular (toma 1–4 líneas en mayúsculas tras 'Parcela N' o 'Polígono ... Parcela ...')."""
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            i = 0
            while i < len(lines):
                line_up = lines[i].upper()

                m_full = POLY_PARC_RE.search(line_up)
                if m_full:
                    parcel = m_full.group(2)
                    block, j = [], i + 1
                    while j < len(lines) and len(block) < 4:
                        raw = lines[j].strip()
                        if not raw:
                            if block: break
                            j += 1; continue
                        if is_upper_name(raw):
                            block.append(raw); j += 1
                        else:
                            break
                    name = reconstruct_owner_from_block(block) if block else ""
                    if parcel and name and parcel not in mapping:
                        mapping[parcel] = name
                    i = j; continue

                m_parc = PARCEL_ONLY_RE.search(line_up)
                if m_parc:
                    parcel = m_parc.group(1)
                    block, j = [], i + 1
                    while j < len(lines) and len(block) < 4:
                        raw = lines[j].strip()
                        if not raw:
                            if block: break
                            j += 1; continue
                        if is_upper_name(raw):
                            block.append(raw); j += 1
                        else:
                            break
                    name = reconstruct_owner_from_block(block) if block else ""
                    if parcel and name and parcel not in mapping:
                        mapping[parcel] = name
                    i = j; continue

                i += 1
    return mapping

def extract_parcels_text(pdf_bytes: bytes) -> List[str]:
    """Lista de parcelas detectadas por texto (para buscar sus números en la pág. 1)."""
    found, seen = [], set()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = normalize_text(page.extract_text(x_tolerance=2, y_tolerance=2) or "")
            for m in re.finditer(PARCEL_ONLY_RE, text):
                num = m.group(1)
                if num not in seen:
                    seen.add(num)
                    found.append(num)
    return found

def extract_owners_fallback_list(pdf_bytes: bytes) -> List[str]:
    """Fallback: hasta 4 nombres cerca de DNIs si faltan números de parcela."""
    owners: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            for i, ln in enumerate(lines):
                if DNI_RE.search(ln):
                    prev_block = []
                    for j in range(i - 1, max(0, i - 14) - 1, -1):
                        raw = lines[j].strip()
                        if not raw: continue
                        if ("POLÍGONO" in raw.upper() or "POLIGONO" in raw.upper() or "PARCELA" in raw.upper() or "[" in raw):
                            break
                        if not is_upper_name(raw): break
                        prev_block.append(raw)
                    prev_block.reverse()
                    name = reconstruct_owner_from_block(prev_block) if prev_block else ""
                    if name and name not in owners:
                        owners.append(name)
                        if len(owners) >= 4: return owners
    return owners

# ----------------- Visión/OCR página 1 -----------------
def page1_to_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 600 if FAST_MODE else 800
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 1.")
    pil_img: Image.Image = pages[0].convert("RGB")
    arr = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
    return arr

def crop_map_region_with_offset(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Recorta cabecera/pie/bordes y devuelve (crop, (offset_x, offset_y))."""
    h, w = bgr.shape[:2]
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.07); right  = int(w * 0.93)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    if bottom - top < 100 or right - left < 100:
        return bgr, (0, 0)
    return bgr[top:bottom, left:right], (left, top)

def as_int_conf(val) -> int:
    try:
        return int(float(val))
    except Exception:
        return -1

def tesseract_boxes(img: np.ndarray, psm: int) -> List[dict]:
    cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data.get("text", []))
    for i in range(n):
        txt_raw = data["text"][i]
        txt = str(txt_raw or "").strip()
        conf = as_int_conf(data["conf"][i])
        if not txt:
            continue
        x = int(data["left"][i]);  y = int(data["top"][i])
        w = int(data["width"][i]); h = int(data["height"][i])
        out.append({"text": txt, "conf": conf, "box": (x, y, w, h)})
    return out

def preprocess_variants(bgr: np.ndarray) -> List[np.ndarray]:
    """Variantes binarizadas; añade CLAHE y dilataciones para engordar trazos finos."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out: List[np.ndarray] = []

    def clahe(g):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return c.apply(g)

    scales = [1.4] if FAST_MODE else [1.6, 2.0]
    bases = [gray, cv2.medianBlur(gray, 3), cv2.GaussianBlur(gray, (5,5), 0), clahe(gray)]
    for g in bases:
        for sc in scales:
            g2 = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            _, bw  = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); out.append(bw)
            _, bwi = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); out.append(bwi)
            ada_g = cv2.adaptiveThreshold(g2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9); out.append(ada_g)
            ada_m = cv2.adaptiveThreshold(g2,255,cv2.ADAPTIVE_THRESH_MEAN_C,     cv2.THRESH_BINARY, 31, 7); out.append(ada_m)
            kern = np.ones((2,2), np.uint8)
            out.append(cv2.dilate(bw,  kern, iterations=1))
            out.append(cv2.dilate(ada_g, kern, iterations=1))
    return out

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(mask_main_green, mask_neighbors_pink) – rangos ampliados para tonos claros."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [
        (np.array([30,  5, 40], np.uint8), np.array([90, 255, 255], np.uint8)),  # más ancho y con S baja
        (np.array([86, 15, 40], np.uint8), np.array([100,255, 255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160,  5, 50], np.uint8), np.array([179,255,255], np.uint8)),  # acepta rosas muy pálidos
        (np.array([  0,  5, 50], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mask_g = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in g_ranges:
        mask_g = cv2.bitwise_or(mask_g, cv2.inRange(hsv, lo, hi))
    mask_p = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in p_ranges:
        mask_p = cv2.bitwise_or(mask_p, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3, 3), np.uint8); k5 = np.ones((5, 5), np.uint8)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, k3)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, k5)
    mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_OPEN, k3)
    mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, k5)
    return mask_g, mask_p

def contours_centroids(mask: np.ndarray, min_area: int = 200) -> List[Tuple[int, int, int]]:
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

def center_of(box):
    x,y,w,h,conf = box
    return (x + w//2, y + h//2)

def side_of(main: Tuple[int,int], pt: Tuple[int,int]) -> str:
    cx, cy = main
    sx, sy = pt[0]-cx, pt[1]-cy
    ang = math.degrees(math.atan2(-(sy), sx))  # norte=+90
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

def read_digits(img: np.ndarray, psms: List[int]) -> str:
    best = ""
    for p in psms:
        cfg = f"--psm {p} --oem 3 -c tessedit_char_whitelist=0123456789"
        txt = pytesseract.image_to_string(img, config=cfg)
        txt = re.sub(r"\D+", "", txt)
        if len(txt) >= 2:
            best = txt
            break
    return best

def find_number_positions(bgr: np.ndarray, targets: List[str]) -> Dict[str, List[Tuple[int,int,int,int,int]]]:
    """Busca posiciones de números exactos en el recorte y devuelve dict num -> cajas (abs)."""
    results: Dict[str, List[Tuple[int,int,int,int,int]]] = {t: [] for t in targets}
    crop, (ox, oy) = crop_map_region_with_offset(bgr)
    variants = preprocess_variants(crop)
    psms = ([6, 7, 11, 13] if FAST_MODE else [6, 7, 8, 10, 11, 12, 13])
    for var in variants:
        for p in psms:
            boxes = tesseract_boxes(var, psm=p)
            for b in boxes:
                txt = re.sub(r"\D+", "", b["text"])
                if not txt:
                    continue
                if txt in results:
                    x,y,w,h = b["box"]
                    abs_box = (x + ox, y + oy, w, h, b["conf"])
                    results[txt].append(abs_box)
    return results

def crop_around_point(img: np.ndarray, pt: Tuple[int,int], w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    x = max(0, pt[0] - w//2)
    y = max(0, pt[1] - h//2)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    return img[y:y2, x:x2]

def local_number_ocr(bgr: np.ndarray, centers: Dict[str, Tuple[int,int]]) -> Dict[str, str]:
    """
    Si el OCR global no encontró números, probar OCR local cerca de cada lado (norte/sur/este/oeste).
    Devuelve dict: lado -> numero_detectado (string de dígitos) si lo hay.
    """
    out: Dict[str, str] = {}
    crop, (ox, oy) = crop_map_region_with_offset(bgr)
    # convert back to absolute: provided centers are absolute; convert to crop coords:
    def to_crop(pt):
        return (max(0, pt[0]-ox), max(0, pt[1]-oy))
    for side, cabs in centers.items():
        cx, cy = to_crop(cabs)
        # recorte algo ancho (números suelen estar pegados a la parcela)
        roi = crop_around_point(crop, (cx, cy), 260 if not FAST_MODE else 200, 160 if not FAST_MODE else 120)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # variantes locales: OTSU, invertido y CLAHE
        locs = []
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); locs.append(bw)
        _, bwi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); locs.append(bwi)
        cla = cv2.createCLAHE(2.0,(8,8)).apply(gray)
        _, cl = cv2.threshold(cla, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); locs.append(cl)
        psms = [7, 6, 11, 13, 10]
        for L in locs:
            num = read_digits(L, psms)
            if num:
                out[side] = num
                break
    return out

# ----------------- Endpoint principal -----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(
    data: ExtractIn = Body(...),
    debug: bool = Query(False)
) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto: mapeo parcela->titular y lista de parcelas por texto
    parcel2owner   = extract_owners_by_parcel(pdf_bytes)
    parcels_text   = extract_parcels_text(pdf_bytes)
    fallback_owners= extract_owners_fallback_list(pdf_bytes)

    # Intentar deducir tu parcela (self) en texto de páginas 1–2
    guessed_self = None
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            t0 = (pdf.pages[0].extract_text() or "") + "\n" + (pdf.pages[1].extract_text() or "")
            mself = re.search(r"PARCELA\s+(\d+)", t0, flags=re.I)
            if mself:
                guessed_self = mself.group(1)
    except Exception:
        pass

    # Si TEXT_ONLY, salimos rápido
    if TEXT_ONLY:
        linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}
        used = set()
        idx = 0
        order = ["norte","sur","oeste","este"]
        for side in order:
            while idx < len(fallback_owners) and fallback_owners[idx] in used:
                idx += 1
            if idx < len(fallback_owners):
                linderos[side] = fallback_owners[idx]
                used.add(fallback_owners[idx]); idx += 1
        owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8]
        note = "Modo TEXT_ONLY activo: OCR desactivado para respuesta rápida."
        dbg = {"TEXT_ONLY": True, "guessed_self": guessed_self, "parcels_text": parcels_text[:20]} if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    # Conjunto de objetivos para OCR
    target_parcels = set(parcels_text) | set(parcel2owner.keys())
    if guessed_self:
        target_parcels.add(guessed_self)
    targets = sorted(list(target_parcels))[: (12 if FAST_MODE else 24) ]

    note_parts = []
    dbg: dict = {}
    linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}

    try:
        bgr = page1_to_bgr(pdf_bytes)
        # 2) Máscara colores
        mask_g, mask_p = color_masks(bgr)

        # Centro principal
        mains = contours_centroids(mask_g, min_area=(400 if FAST_MODE else 250))
        main_center = (mains[0][0], mains[0][1]) if mains else None
        center_mode = "green_centroid" if mains else "none"

        # 3) OCR global de números (si tenemos objetivos)
        num_boxes = {}
        if targets:
            num_boxes = find_number_positions(bgr, targets=targets)

        # 4) Escoger por lado el número más cercano al centro principal (si existe)
        best_per_side: Dict[str, Tuple[str, Tuple[int,int], int]] = {}
        if main_center:
            # Centroides de vecinos (rosas)
            neighbors = contours_centroids(mask_p, min_area=(220 if FAST_MODE else 150))
            # Si OCR global devuelve ubicaciones, combínalo; si no, luego haremos OCR local
            for num, boxes in num_boxes.items():
                if not boxes: continue
                c = min([center_of(b) for b in boxes], key=lambda p: (p[0]-main_center[0])**2 + (p[1]-main_center[1])**2)
                sd = side_of(main_center, c)
                d2 = (c[0]-main_center[0])**2 + (c[1]-main_center[1])**2
                if sd not in best_per_side or d2 < best_per_side[sd][2]:
                    best_per_side[sd] = (num, c, d2)

            # Si no hay números aún, usa vecinos y OCR local
            if not best_per_side and neighbors:
                # Candidato por lado = vecino rosa más cercano al centro
                chosen: Dict[str, Tuple[int,int]] = {}
                for cx, cy, _ in neighbors:
                    sd = side_of(main_center, (cx, cy))
                    d2 = (cx-main_center[0])**2 + (cy-main_center[1])**2
                    if sd not in chosen or d2 < (chosen[sd][0]-main_center[0])**2 + (chosen[sd][1]-main_center[1])**2:
                        chosen[sd] = (cx, cy)
                # OCR local alrededor de cada candidato
                local_nums = local_number_ocr(bgr, chosen)
                for sd, num in local_nums.items():
                    best_per_side[sd] = (num, chosen[sd], 0)

        # 5) Construir linderos con nombres si hay mapeo
        for sd, (num, c, _) in best_per_side.items():
            owner = parcel2owner.get(num, "")
            if owner:
                linderos[sd] = owner

        # 6) Completar con fallback textual si faltan lados
        used = {v for v in linderos.values() if v}
        if fallback_owners:
            idx = 0
            for side in ["norte","sur","oeste","este"]:
                if not linderos[side]:
                    while idx < len(fallback_owners) and fallback_owners[idx] in used:
                        idx += 1
                    if idx < len(fallback_owners):
                        linderos[side] = fallback_owners[idx]
                        used.add(fallback_owners[idx]); idx += 1

        if not any(linderos.values()):
            note_parts.append("OCR sin coincidencias claras; afinaremos binarización/PSM y extracción de titulares.")
        else:
            missing = [k for k,v in linderos.items() if not v]
            if missing:
                note_parts.append(f"Faltan lados: {', '.join(missing)} (sin titular asociado).")

        owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8]
        note = " ".join(note_parts) if note_parts else None

        if debug:
            dbg = {
                "FAST_MODE": FAST_MODE,
                "guessed_self": guessed_self,
                "parcels_text": parcels_text[:20],
                "targets": targets,
                "owners_by_parcel_sample": dict(list(parcel2owner.items())[:8]),
                "ocr_counts": {k: len(v) for k, v in num_boxes.items()},
                "main_center": main_center,
                "center_mode": center_mode,
                "best_per_side": {s: {"parcel": n, "center": c} for s,(n,c,_) in best_per_side.items()}
            }

        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=(dbg or None))

    except Exception as e:
        owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8]
        dbg = {"exception": str(e), "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY} if debug else None
        return ExtractOut(
            linderos={"norte":"","sur":"","oeste":"","este":""},
            owners_detected=owners_detected,
            note=f"Excepción visión/OCR: {e}",
            debug=dbg
        )

@app.get("/health")
def health():
    return {"ok": True, "version": app.version if hasattr(app, "version") else "unknown", "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY}

@app.get("/diag")
def diag():
    """Diagnóstico rápido dentro del contenedor: versiones de Tesseract/Poppler/cv2."""
    def cmdout(cmd: List[str]) -> str:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5).decode(errors="ignore").strip()
            return out.splitlines()[0]
        except Exception as e:
            return f"err: {e}"
    try:
        tess_v = cmdout(["tesseract", "--version"])
    except Exception as e:
        tess_v = f"err: {e}"
    try:
        pdfinfo_v = cmdout(["pdfinfo", "-v"])
    except Exception as e:
        pdfinfo_v = f"err: {e}"
    try:
        cv2_v = cv2.__version__
    except Exception as e:
        cv2_v = f"err: {e}"
    try:
        pytess_v = str(pytesseract.get_tesseract_version()) if hasattr(pytesseract,"get_tesseract_version") else "unknown"
    except Exception as e:
        pytess_v = f"err: {e}"
    return {
        "ok": True,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "tesseract": tess_v,
        "pdfinfo": pdfinfo_v,
        "cv2": cv2_v,
        "pytesseract": pytess_v
    }

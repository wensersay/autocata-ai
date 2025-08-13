from fastapi import FastAPI, HTTPException, Body, Depends, Header
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional, Tuple
import requests, io, re, math
import pdfplumber
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import pytesseract
from PIL import Image

app = FastAPI(title="AutoCatastro AI", version="0.2.0")

# ----------------- (Opcional) Seguridad simple por token -----------------
import os
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
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

# ----------------- Utilidades texto PDF -----------------
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")
POLY_PARC_RE = re.compile(r"POL[ÍI]GONO\s+(\d+).*?PARCELA\s+(\d+)", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION"
)

def fetch_pdf_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el PDF (HTTP {r.status_code}).")
        ct = (r.headers.get("content-type") or "").lower()
        if "pdf" not in ct:
            if not r.content.startswith(b"%PDF"):
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
        if not tok: i += 1; continue
        if len(tok) == 1 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 2: clean.append(tok+nxt); i += 2; continue
        if len(tok) <= 2 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 3: clean.append(tok+nxt); i += 2; continue
        if i + 2 < len(tokens):
            nxt1 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            nxt2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 2])
            if nxt1 and len(nxt1) == 1 and nxt2 and len(nxt2) >= 3:
                clean.append(tok+nxt1+nxt2); i += 3; continue
        clean.append(tok); i += 1
    name = " ".join(clean)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_owners_by_parcel(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Lee páginas (sobre todo la 2) para construir un dict: parcela -> titular (nombre en MAYÚSCULAS).
    """
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            i = 0
            while i < len(lines):
                m = POLY_PARC_RE.search(lines[i].upper())
                if m:
                    # tras esta línea, vienen 1-3 líneas de nombre en MAYÚSCULAS
                    parcel = m.group(2)
                    block = []
                    j = i + 1
                    while j < len(lines) and len(block) < 3:
                        raw = lines[j].strip()
                        if not raw:
                            if block: break
                            j += 1; continue
                        if is_upper_name(raw):
                            block.append(raw)
                            j += 1
                        else:
                            break
                    name = reconstruct_owner_from_block(block) if block else ""
                    if parcel and name and parcel not in mapping:
                        mapping[parcel] = name
                    i = j
                else:
                    i += 1
    return mapping

def extract_owners_fallback_list(pdf_bytes: bytes) -> List[str]:
    """
    Heurística lista (por si faltan números de parcela): hasta 4 nombres en orden.
    """
    owners: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            for i, ln in enumerate(lines):
                if DNI_RE.search(ln):
                    prev_block = []
                    for j in range(i - 1, max(0, i - 12) - 1, -1):
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

# ----------------- Visión por computador -----------------
def page1_to_bgr(pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 1.")
    pil_img: Image.Image = pages[0].convert("RGB")
    arr = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
    return arr

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (mask_main_green, mask_neighbors_pink)
    Ajustado a colores típicos Catastro: verde/aguamarina para la parcela y rosa para colindantes.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Verde/aguamarina (amplio)
    lower_g = np.array([70, 30, 70], dtype=np.uint8)
    upper_g = np.array([110, 255, 255], dtype=np.uint8)
    mask_g = cv2.inRange(hsv, lower_g, upper_g)

    # Rosa: dos rangos (rojo bajo y alto) + saturación baja (pastel)
    lower_p1 = np.array([160, 30, 70], dtype=np.uint8)
    upper_p1 = np.array([179, 255, 255], dtype=np.uint8)
    lower_p2 = np.array([0, 30, 70], dtype=np.uint8)
    upper_p2 = np.array([10, 255, 255], dtype=np.uint8)
    mask_p = cv2.bitwise_or(cv2.inRange(hsv, lower_p1, upper_p1),
                            cv2.inRange(hsv, lower_p2, upper_p2))

    # Limpiar ruido
    k = np.ones((5, 5), np.uint8)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, k, iterations=1)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, k, iterations=1)
    mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_OPEN, k, iterations=1)
    mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, k, iterations=1)

    return mask_g, mask_p

def contours_centroids(mask: np.ndarray, min_area: int = 300) -> List[Tuple[int, int, int]]:
    """
    Devuelve lista de (cx, cy, area) para contornos con área mínima.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    # ordenar por área desc
    out.sort(key=lambda x: -x[2])
    return out

def assign_cardinals(main: Tuple[int,int], neighbors: List[Tuple[int,int]]) -> Dict[str, Tuple[int,int]]:
    """
    Asigna para cada cuadrante el vecino más cercano: N, S, E, O.
    """
    if not neighbors: return {}
    cx, cy = main
    side_best: Dict[str, Tuple[int,int,int]] = {}  # side -> (nx,ny,dist2)

    def side_of(vx, vy) -> str:
        # y crece hacia abajo; norte es arriba
        ang = math.degrees(math.atan2(-(vy), vx))  # eje y invertido para norte=+90
        # Mapear por sectores
        if -45 <= ang <= 45: return "este"
        if 45 < ang <= 135:  return "norte"
        if -135 <= ang < -45:return "sur"
        return "oeste"

    for nx, ny in neighbors:
        sx, sy = nx - cx, ny - cy
        s = side_of(sx, sy)
        d2 = sx*sx + sy*sy
        if s not in side_best or d2 < side_best[s][2]:
            side_best[s] = (nx, ny, d2)

    return { k: (v[0], v[1]) for k, v in side_best.items() }

def ocr_digits_around(bgr: np.ndarray, pt: Tuple[int,int], box: int = 140) -> Optional[str]:
    """
    OCR de dígitos de parcela cerca del punto dado.
    """
    h, w = bgr.shape[:2]
    x0 = max(0, pt[0]-box//2); y0 = max(0, pt[1]-box//2)
    x1 = min(w, pt[0]+box//2); y1 = min(h, pt[1]+box//2)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0: return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cfg = "--psm 6 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(bw, config=cfg) or ""
    txt = re.sub(r"\D+", " ", txt)
    # Escoge el número más largo (3-5 dígitos típico)
    cands = sorted([t for t in txt.split() if 2 <= len(t) <= 6], key=len, reverse=True)
    return cands[0] if cands else None

# ----------------- Endpoint principal -----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto: mapeo parcela->titular y fallback lista
    parcel2owner = extract_owners_by_parcel(pdf_bytes)
    fallback_owners = extract_owners_fallback_list(pdf_bytes)

    # 2) Visión: localizar parcela principal (verde) y colindantes (rosa)
    try:
        bgr = page1_to_bgr(pdf_bytes, dpi=300)
        mask_main, mask_nei = color_masks(bgr)
        mains = contours_centroids(mask_main, min_area=500)
        if not mains:
            raise RuntimeError("No se detectó la parcela principal en verde.")
        main_cx, main_cy, _ = mains[0]
        neis = [(x,y) for (x,y,area) in contours_centroids(mask_nei, min_area=200)]
        side_pts = assign_cardinals((main_cx, main_cy), neis)
    except Exception as e:
        # Si algo falla en visión, seguimos con linderos vacíos; el texto/heurística actúa como fallback
        side_pts = {}

    # 3) Intentar OCR de número de parcela en cada lado → cruzar con parcel2owner
    linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}
    used = set()
    if side_pts:
        for side in ["norte","sur","oeste","este"]:
            pt = side_pts.get(side)
            if not pt: continue
            num = ocr_digits_around(bgr, pt, box=150)
            if num and num in parcel2owner:
                linderos[side] = parcel2owner[num]
                used.add(parcel2owner[num])

    # 4) Completar con fallback si faltan lados
    if fallback_owners:
        idx = 0
        for side in ["norte","sur","oeste","este"]:
            if not linderos[side]:
                # coge el primer no usado
                while idx < len(fallback_owners) and fallback_owners[idx] in used:
                    idx += 1
                if idx < len(fallback_owners):
                    linderos[side] = fallback_owners[idx]
                    used.add(fallback_owners[idx])
                    idx += 1

    note = None
    if not any(linderos.values()):
        note = ("Visión activa pero sin coincidencias robustas. "
                "Añadiremos ajustes de color/umbral u OCR ampliado si este PDF trae tono atípico.")

    # owners_detected: une los del mapping y del fallback (máx 8 para inspección)
    owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8]

    return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note)

@app.get("/health")
def health():
    return {"ok": True}


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
import os

app = FastAPI(title="AutoCatastro AI", version="0.2.1")

# -------- Seguridad opcional por token (deja AUTH_TOKEN vacío si no lo usas) --------
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
PARCEL_ONLY_RE = re.compile(r"PARCELA\s+(\d+)", re.IGNORECASE)
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
    Construye dict: parcela -> titular, tolerando “Polígono” y “Parcela” en líneas separadas.
    Regla: detecta “Parcela (\d+)” y toma 1–3 líneas MAYÚSCULAS siguientes como nombre.
    """
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            i = 0
            while i < len(lines):
                line_up = lines[i].upper()

                # Caso 1: ambos en la misma línea
                m_full = POLY_PARC_RE.search(line_up)
                if m_full:
                    parcel = m_full.group(2)
                    block, j = [], i + 1
                    while j < len(lines) and len(block) < 3:
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
                    i = j
                    continue

                # Caso 2: “Parcela (\d+)” sola, con nombre a continuación
                m_parc = PARCEL_ONLY_RE.search(line_up)
                if m_parc:
                    parcel = m_parc.group(1)
                    block, j = [], i + 1
                    while j < len(lines) and len(block) < 3:
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
                    i = j
                    continue

                i += 1
    return mapping

def extract_owners_fallback_list(pdf_bytes: bytes) -> List[str]:
    """
    Por si faltan números de parcela: hasta 4 nombres en orden detectados cerca de DNIs.
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
def page1_to_bgr(pdf_bytes: bytes, dpi: int = 400) -> np.ndarray:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 1.")
    pil_img: Image.Image = pages[0].convert("RGB")
    arr = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
    return arr

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (mask_main_green, mask_neighbors_pink) con rangos más tolerantes.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Verde/agua-marina (amplio)
    g_ranges = [
        (np.array([35, 20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),   # verde
        (np.array([86, 15, 50], np.uint8), np.array([100, 255, 255], np.uint8)),  # cian
    ]
    mask_g = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in g_ranges:
        mask_g = cv2.bitwise_or(mask_g, cv2.inRange(hsv, lo, hi))

    # Rosa (rojo bajo y alto, SAT/VAL bajos para pastel)
    p_ranges = [
        (np.array([160, 20, 60], np.uint8), np.array([179, 255, 255], np.uint8)),
        (np.array([0,   20, 60], np.uint8), np.array([10,  255, 255], np.uint8)),
    ]
    mask_p = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in p_ranges:
        mask_p = cv2.bitwise_or(mask_p, cv2.inRange(hsv, lo, hi))

    # Limpieza suave
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    for _ in range(1):
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

def assign_cardinals(main: Tuple[int,int], neighbors: List[Tuple[int,int]]) -> Dict[str, Tuple[int,int]]:
    if not neighbors: return {}
    cx, cy = main
    side_best: Dict[str, Tuple[int,int,int]] = {}

    def side_of(vx, vy) -> str:
        ang = math.degrees(math.atan2(-(vy), vx))  # norte=+90
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

def ocr_best_digits(bgr: np.ndarray, pt: Tuple[int,int]) -> Optional[str]:
    """
    Prueba varias ventanas y PSM; devuelve el número de parcela más probable.
    """
    h, w = bgr.shape[:2]
    sizes = [150, 220, 280]
    psms  = [6, 7, 11]
    best = None

    for box in sizes:
        x0 = max(0, pt[0]-box//2); y0 = max(0, pt[1]-box//2)
        x1 = min(w, pt[0]+box//2); y1 = min(h, pt[1]+box//2)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0: continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        for p in psms:
            cfg = f"--psm {p} -c tessedit_char_whitelist=0123456789"
            txt = pytesseract.image_to_string(bw, config=cfg) or ""
            txt = re.sub(r"\D+", " ", txt)
            cands = sorted([t for t in txt.split() if 2 <= len(t) <= 6], key=len, reverse=True)
            if cands:
                best = cands[0]
                return best  # primer buen match
    return best

# ----------------- Endpoint principal -----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto: mapeo parcela->titular y fallback
    parcel2owner = extract_owners_by_parcel(pdf_bytes)
    fallback_owners = extract_owners_fallback_list(pdf_bytes)

    # 2) Visión: localizar principal y vecinos
    linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}
    used = set()
    note = None
    try:
        bgr = page1_to_bgr(pdf_bytes, dpi=400)
        mask_main, mask_nei = color_masks(bgr)
        mains = contours_centroids(mask_main, min_area=400)
        neis  = [(x,y) for (x,y,area) in contours_centroids(mask_nei, min_area=180)]
        if not mains:
            raise RuntimeError("No se detectó la parcela principal en verde/aguamarina.")
        main_cx, main_cy, _ = mains[0]
        side_pts = assign_cardinals((main_cx, main_cy), neis)

        # 3) OCR → titular por parcela
        for side in ["norte","sur","oeste","este"]:
            pt = side_pts.get(side)
            if not pt: continue
            num = ocr_best_digits(bgr, pt)
            if num and num in parcel2owner:
                linderos[side] = parcel2owner[num]
                used.add(parcel2owner[num])

    except Exception as e:
        note = f"Visión activa pero sin coincidencias robustas ({e})."

    # 4) Completar con fallback si faltan lados
    if fallback_owners:
        idx = 0
        for side in ["norte","sur","oeste","este"]:
            if not linderos[side]:
                while idx < len(fallback_owners) and fallback_owners[idx] in used:
                    idx += 1
                if idx < len(fallback_owners):
                    linderos[side] = fallback_owners[idx]
                    used.add(fallback_owners[idx])
                    idx += 1

    if not any(linderos.values()):
        note = (note or "") + " Ajustaremos rangos de color/umbral u OCR si este PDF trae tonalidades atípicas."

    owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8]
    return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note)

@app.get("/health")
def health():
    return {"ok": True}

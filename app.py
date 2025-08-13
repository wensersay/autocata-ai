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

app = FastAPI(title="AutoCatastro AI", version="0.2.2")

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

# ----------------- Visión/OCR página 1 -----------------
def page1_to_bgr(pdf_bytes: bytes, dpi: int = 450) -> np.ndarray:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 1.")
    pil_img: Image.Image = pages[0].convert("RGB")
    arr = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
    return arr

def tesseract_boxes(img: np.ndarray, psm: int) -> List[dict]:
    """Devuelve las cajas OCR con conf y texto usando image_to_data."""
    cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = int(data["conf"][i]) if data["conf"][i].isdigit() else -1
        if not txt: continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"text": txt, "conf": conf, "box": (x,y,w,h)})
    return out

def preprocess_variants(bgr: np.ndarray) -> List[np.ndarray]:
    """Genera variantes binarizadas para mejorar OCR de dígitos."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    # Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(bw)
    # Otsu invertido
    _, bwi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(bwi)
    # Adaptativa
    ada = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    out.append(ada)
    # Suave + Otsu
    blur = cv2.medianBlur(gray, 3)
    _, bw2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(bw2)
    return out

def find_number_positions(bgr: np.ndarray, targets: List[str]) -> Dict[str, List[Tuple[int,int,int,int,int]]]:
    """
    Busca posiciones de números exactos en página completa.
    Devuelve dict num -> lista de cajas (x,y,w,h,conf).
    """
    results: Dict[str, List[Tuple[int,int,int,int,int]]] = {t: [] for t in targets}
    variants = preprocess_variants(bgr)
    psms = [6, 7, 11]
    for var in variants:
        for p in psms:
            boxes = tesseract_boxes(var, psm=p)
            for b in boxes:
                txt = re.sub(r"\D+", "", b["text"])
                if not txt: continue
                if txt in results:
                    x,y,w,h = b["box"]
                    results[txt].append( (x,y,w,h,b["conf"]) )
    return results

def center_of(box):
    x,y,w,h,conf = box
    return (x + w//2, y + h//2)

def choose_main_center(cands_72: List[Tuple[int,int,int,int,int]], neighbors_centers: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    if not cands_72:
        return None
    if not neighbors_centers:
        # elige el 72 más centrado en página
        return min([center_of(b) for b in cands_72], key=lambda c: c[0]*c[0]+c[1]*c[1])
    # elige el 72 más cercano al centroide de vecinos
    cx = int(sum(p[0] for p in neighbors_centers) / len(neighbors_centers))
    cy = int(sum(p[1] for p in neighbors_centers) / len(neighbors_centers))
    return min([center_of(b) for b in cands_72], key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)

def side_of(main: Tuple[int,int], pt: Tuple[int,int]) -> str:
    cx, cy = main
    sx, sy = pt[0]-cx, pt[1]-cy
    ang = math.degrees(math.atan2(-(sy), sx))  # norte=+90
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

# ----------------- Endpoint principal -----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto: mapeo parcela->titular (pág. 2) y fallback lista
    parcel2owner = extract_owners_by_parcel(pdf_bytes)
    fallback_owners = extract_owners_fallback_list(pdf_bytes)

    # Qué parcelas buscar: la propia + las colindantes detectadas en texto
    target_parcels = set(parcel2owner.keys())
    # si deducimos la tuya de la RC (últimos dígitos) mejor, pero mín. intenta "72"
    guessed_self = None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        t0 = (pdf.pages[0].extract_text() or "") + "\n" + (pdf.pages[1].extract_text() or "")
        mself = re.search(r"PARCELA\s+(\d+)", t0, flags=re.I)
        if mself:
            guessed_self = mself.group(1)
    if guessed_self:
        target_parcels.add(guessed_self)
    else:
        # último recurso: si hay exactamente 4 colindantes, intenta la que falta respecto a 71-74-... (no siempre sirve)
        pass

    # 2) OCR de números en página 1
    note_parts = []
    try:
        bgr = page1_to_bgr(pdf_bytes, dpi=450)
        targets = sorted(target_parcels)[:8]  # seguridad
        num_boxes = find_number_positions(bgr, targets=targets)

        # centros vecinos (todas menos la tuya)
        neighbors_centers: List[Tuple[int,int]] = []
        self_centers: List[Tuple[int,int]] = []

        for num, boxes in num_boxes.items():
            centers = [center_of(b) for b in boxes]
            if not centers: continue
            # heurística de cuál es "tu" parcela: guessed_self si existe
            if guessed_self and num == guessed_self:
                self_centers.extend(centers)
            else:
                neighbors_centers.extend(centers)

        main_center = choose_main_center(num_boxes.get(guessed_self, []), neighbors_centers) if guessed_self else None
        if not main_center and self_centers:
            # elige la más centrada
            main_center = min(self_centers, key=lambda c: c[0]*c[0]+c[1]*c[1])

        # Asignación cardinal → número de parcela vecino más cercano en cada cuadrante
        linderos_by_num: Dict[str, str] = {}
        if main_center and neighbors_centers:
            # para cada número detectado, coge el centro más cercano a main_center
            best_per_side: Dict[str, Tuple[str, Tuple[int,int], int]] = {}
            for num, boxes in num_boxes.items():
                if guessed_self and num == guessed_self:  # saltar la tuya
                    continue
                if not boxes: continue
                # mejor caja por proximidad al main
                c = min([center_of(b) for b in boxes], key=lambda p: (p[0]-main_center[0])**2 + (p[1]-main_center[1])**2)
                sd = side_of(main_center, c)
                d2 = (c[0]-main_center[0])**2 + (c[1]-main_center[1])**2
                if sd not in best_per_side or d2 < best_per_side[sd][2]:
                    best_per_side[sd] = (num, c, d2)

            for sd, (num, c, _) in best_per_side.items():
                if num in parcel2owner:
                    linderos_by_num[sd] = parcel2owner[num]

        # 3) Montar respuesta final
        linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}
        linderos.update(linderos_by_num)

        # 4) Completar con fallback si faltan lados
        used = set(linderos_by_num.values())
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

        # Nota de depuración (útil mientras afinamos)
        if not any(linderos.values()):
            note_parts.append("OCR sin coincidencias claras; afinaremos binarización/PSM.")
        else:
            missing = [k for k,v in linderos.items() if not v]
            if missing:
                note_parts.append(f"Faltan lados: {', '.join(missing)}. Se completaron con fallback si era posible.")

        owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8]
        note = " ".join(note_parts) if note_parts else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note)

    except Exception as e:
        # Si algo revienta, devolvemos al menos los owners detectados
        return ExtractOut(
            linderos={"norte":"","sur":"","oeste":"","este":""},
            owners_detected=list(dict.fromkeys(list(parcel2owner.values()) + fallback_owners))[:8],
            note=f"Excepción visión/OCR: {e}"
        )

@app.get("/health")
def health():
    return {"ok": True}


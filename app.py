from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional, Tuple
import requests, io, re, math, os, json, subprocess
import pdfplumber
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageDraw

app = FastAPI(title="AutoCatastro AI", version="0.3.0")

# -------- Flags/entorno --------
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")  # por defecto ON
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")  # por defecto OFF

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

# ----------------- Utilidades de red/Text -----------------
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE = re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)
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
    """Devuelve dict: nº parcela -> titular (leyendo texto en páginas 2+)."""
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[1:]:  # desde página 2 en adelante
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

# ----------------- Visión en páginas 2–3 -----------------
def pages2_to_bgrs(pdf_bytes: bytes, max_pages: int = 2) -> List[np.ndarray]:
    dpi = 400 if FAST_MODE else 550
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page= 1 + max_pages)
    out = []
    for p in pages:
        arr = np.array(p.convert("RGB"))[:, :, ::-1]  # RGB->BGR
        out.append(arr)
    return out

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (mask_green, mask_pink). Rango amplio y robusto."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Verde agua (parcela objetivo)
    green_ranges = [
        (np.array([35, 20, 60], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86, 15, 60], np.uint8), np.array([100,255, 255], np.uint8)),
    ]
    mask_g = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in green_ranges:
        mask_g |= cv2.inRange(hsv, lo, hi)

    # Rosa (adyacentes). Combina rojos bajo y alto.
    pink_ranges = [
        (np.array([160, 30, 60], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([0,   30, 60], np.uint8), np.array([10, 255,255], np.uint8)),
    ]
    mask_p = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in pink_ranges:
        mask_p |= cv2.inRange(hsv, lo, hi)

    k3 = np.ones((3,3), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    for _ in range(2):
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, k3)
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, k5)
        mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_OPEN, k3)
        mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, k5)
    return mask_g, mask_p

def find_main_green(mask_g: np.ndarray) -> Optional[np.ndarray]:
    """Devuelve el contorno de la mancha verde más grande."""
    cnts, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts[0]

def centroid_of(cnt: np.ndarray) -> Tuple[int,int]:
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        x,y,w,h = cv2.boundingRect(cnt)
        return x + w//2, y + h//2
    return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

def neighbors_touching(mask_p: np.ndarray, cnt_green: np.ndarray) -> List[np.ndarray]:
    """Filtra contornos rosas que tocan/rozan al verde (dilatamos verde y cruzamos)."""
    dil = cv2.dilate(cv2.drawContours(np.zeros_like(mask_p), [cnt_green], -1, 255, thickness=cv2.FILLED),
                     np.ones((7,7),np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 150:  # ruido pequeño
            continue
        # ¿toca al dilatado?
        mask_c = np.zeros_like(mask_p)
        cv2.drawContours(mask_c, [c], -1, 255, thickness=cv2.FILLED)
        if cv2.countNonZero(cv2.bitwise_and(mask_c, dil)) > 0:
            out.append(c)
    return out

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y = pt_xy
    # N arriba (y menor), E derecha (x mayor), S abajo (y mayor), O izquierda (x menor)
    dx, dy = (x - cx), (y - cy)
    ang = math.degrees(math.atan2(-dy, dx))
    if -45 <= ang <= 45:   return "este"
    if 45 < ang <= 135:    return "norte"
    if -135 <= ang < -45:  return "sur"
    return "oeste"

def ocr_parcel_in_bbox(bgr: np.ndarray, cnt: np.ndarray) -> str:
    """Recorta el bbox del contorno rosa y hace OCR de dígitos (robusto y barato)."""
    x,y,w,h = cv2.boundingRect(cnt)
    crop = bgr[max(0,y-4):y+h+4, max(0,x-4):x+w+4].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # realzar texto negro
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
    d = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)
    best = ""
    best_conf = -1
    for i, txt in enumerate(d.get("text", [])):
        t = re.sub(r"\D+", "", (txt or ""))
        if not t:
            continue
        try:
            conf = int(float(d["conf"][i]))
        except Exception:
            conf = 0
        if conf > best_conf and 1 <= len(t) <= 5:
            best, best_conf = t, conf
    return best

def build_preview_png(bgr: np.ndarray, cnt_g: Optional[np.ndarray], cols: List[Tuple[str, Tuple[int,int], str]]) -> bytes:
    """Dibuja verde/rosa detectados y etiquetas en PNG."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    if cnt_g is not None:
        xs = cnt_g[:,0,0].tolist(); ys = cnt_g[:,0,1].tolist()
        poly = list(zip(xs, ys))
        draw.line(poly + [poly[0]], fill=(0,200,0), width=4)
    for side, (x,y), label in cols:
        draw.ellipse((x-6,y-6,x+6,y+6), outline=(255,0,0), width=3)
        draw.text((x+8, y-10), f"{side.upper()}:{label}", fill=(255,0,0))
    out = io.BytesIO()
    pil.save(out, format="PNG")
    return out.getvalue()

# ----------------- Endpoint principal -----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(
    data: ExtractIn = Body(...),
    debug: bool = Query(False)
) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 0) Texto: titulares por parcela (pág. 2+)
    parcel2owner = extract_owners_by_parcel(pdf_bytes)

    # Modo solo texto (si se fuerza)
    if TEXT_ONLY:
        owners = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = "Modo TEXT_ONLY activo: visión desactivada."
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners, note=note,
                          debug={"TEXT_ONLY":True} if debug else None)

    # 1) Rasteriza páginas 2–3 (suelen contener el croquis)
    bgr_pages = pages2_to_bgrs(pdf_bytes, max_pages=2)
    if not bgr_pages:
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=list(parcel2owner.values())[:8],
                          note="No se pudieron rasterizar páginas 2–3.")

    linderos = {"norte":"","sur":"","oeste":"","este":""}
    debug_info = {}
    note_parts = []

    for idx, bgr in enumerate(bgr_pages, start=2):
        mask_g, mask_p = color_masks(bgr)
        cnt_g = find_main_green(mask_g)
        if cnt_g is None:
            note_parts.append(f"p{idx}: sin verde.")
            continue
        c_main = centroid_of(cnt_g)

        # 2) Colindantes rosas que tocan a la verde
        neigh = neighbors_touching(mask_p, cnt_g)
        if not neigh:
            note_parts.append(f"p{idx}: sin colindantes tocando.")
            continue

        # 3) Para cada colindante: lado + nº parcela (OCR local) + titular si existe
        side2best = {}  # side -> (parcel_num, conf_dist, center)
        for c in neigh:
            cx, cy = centroid_of(c)
            sd = side_of(c_main, (cx, cy))
            parcel_num = ocr_parcel_in_bbox(bgr, c)
            # aproximar “distancia” para elegir el más cercano por lado
            d2 = (cx - c_main[0])**2 + (cy - c_main[1])**2
            key = sd
            # preferimos el más cercano (d2 pequeño)
            prev = side2best.get(key)
            if prev is None or d2 < prev[1]:
                side2best[key] = (parcel_num, d2, (cx, cy))

        # 4) Mapear a titulares si hay nº de parcela reconocido
        for sd, (num, _, center) in side2best.items():
            if not linderos.get(sd):  # no sobreescribir si ya lo rellenamos con otra página
                owner = parcel2owner.get(num, "")
                linderos[sd] = owner or ""  # si no hay titular, lo dejamos vacío, ya rellenaremos fallback

        if debug:
            debug_info[f"p{idx}"] = {
                "main_center": c_main,
                "sides_found": {sd: {"parcel": num, "center": center} for sd,(num,_,center) in side2best.items()}
            }

        # Si ya tenemos los 4 lados, podemos salir
        if all(linderos.values()):
            break

    # 5) Fallback: si faltan lados, rellena con los primeros titulares “plausibles”
    missing = [k for k,v in linderos.items() if not v]
    if missing:
        owners_ordered = list(dict.fromkeys(parcel2owner.values()))
        i = 0
        for sd in missing:
            if i < len(owners_ordered):
                linderos[sd] = owners_ordered[i]
                i += 1

    owners_detected = list(dict.fromkeys(parcel2owner.values()))[:10]
    note = " ".join(note_parts) if note_parts else None
    dbg = (debug_info if debug else None)
    return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

# ----------------- Health & Preview -----------------
@app.get("/health")
def health():
    return {"ok": True, "version": "0.3.0", "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY}

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl):
    """Devuelve PNG con detección (útil para depurar en /page2)."""
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    bgr_pages = pages2_to_bgrs(pdf_bytes, max_pages=1)
    if not bgr_pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    bgr = bgr_pages[0]
    mask_g, mask_p = color_masks(bgr)
    cnt_g = find_main_green(mask_g)
    cols = []
    if cnt_g is not None:
        c_main = centroid_of(cnt_g)
        neigh = neighbors_touching(mask_p, cnt_g)
        for c in neigh:
            cx, cy = centroid_of(c)
            sd = side_of(c_main, (cx, cy))
            num = ocr_parcel_in_bbox(bgr, c)
            cols.append((sd, (cx, cy), num or "?"))
    png = build_preview_png(bgr, cnt_g, cols)
    return Response(content=png, media_type="image/png")

from fastapi.responses import Response


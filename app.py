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
app = FastAPI(title="AutoCatastro AI", version="0.4.4")

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
    "POLÍGONO","POLIGONO","PARCELA","[","]","(",")",
    "COORDENADAS","ETRS","HUSO","ESCALA","TITULARIDAD",
    "VALOR CATASTRAL","LOCALIZACIÓN","LOCALIZACION",
    "REFERENCIA CATASTRAL","NIF","DOMICILIO","PL:","PT:"
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
    if sum(ch.isdigit() for ch in line) >= 1:
        return False
    return bool(UPPER_NAME_RE.match(line))

def clean_name_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\[[^\]]*\]", "", s)          # quita [LUGO], etc.
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\d+", "", s)                  # quita números
    s = re.sub(r"\s{2,}", " ", s).strip()
    # corta a 26 caracteres (regla empírica del proyecto)
    if len(s) > 26:
        s = s[:26].rstrip()
    return s

def merge_second_token_if_short(first: str, second: str) -> str:
    """
    Si la segunda línea es una palabra corta (p.ej., 'LUIS', 'JOSE'),
    fusiónala al final del primer nombre.
    """
    t = second.strip()
    if 1 <= len(t) <= 10 and re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ]+", t):
        # evita duplicar si ya termina en esa palabra
        if not first.endswith(" " + t):
            return (first + " " + t).strip()
    return first

def extract_owners_map_text(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Construye dict {parcela: titular} desde páginas ≥2 usando SOLO TEXTO.
    Heurística: tras '... Parcela N' busca la primera línea en MAYÚSCULAS
    'tipo nombre' y opcionalmente fusiona una segunda palabra corta.
    """
    mapping: Dict[str, str] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, page in enumerate(pdf.pages):
            if pi == 0:
                continue
            text = normalize_text(page.extract_text(x_tolerance=2, y_tolerance=2) or "")
            lines = text.split("\n")
            curr_parcel: Optional[str] = None
            i = 0
            while i < len(lines):
                raw = lines[i].strip()
                up  = raw.upper()
                # Detectar '... Parcela N'
                if "PARCELA" in up:
                    tokens = [t for t in up.replace(",", " ").split() if t.isdigit()]
                    if tokens:
                        curr_parcel = tokens[-1]
                # Tras cabecera de tabla o inmediatamente después de 'PARCELA',
                # buscamos la primera línea candidata a titular
                if ("APELLIDOS NOMBRE" in up and "RAZON" in up) or ("TITULARIDAD PRINCIPAL" in up) or ("PARCELA" in up):
                    j = i + 1
                    cand1 = ""
                    cand2 = ""
                    steps = 0
                    while j < len(lines) and steps < 12 and (not cand1 or not cand2):
                        s = lines[j].strip()
                        U = s.upper()
                        if any(k in U for k in ("APELLIDOS NOMBRE","RAZON SOCIAL","NIF","DOMICILIO","REFERENCIA CATASTRAL")):
                            j += 1; steps += 1; continue
                        s_clean = clean_name_line(s.upper())
                        if s_clean and is_upper_name(s_clean):
                            if not cand1:
                                cand1 = s_clean
                            elif not cand2:
                                cand2 = s_clean
                        j += 1; steps += 1
                    if curr_parcel and cand1:
                        owner = cand1
                        if cand2:
                            owner = merge_second_token_if_short(owner, cand2)
                        if curr_parcel not in mapping:
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
    # recorte conservador que viene funcionando bien
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.08); right = int(w * 0.92)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    if bottom - top < 100 or right - left < 100:
        return bgr, (0, 0)
    return bgr[top:bottom, left:right], (left, top)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # verde (parcela actual)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # rosa (vecinos)
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
    ang = math.degrees(math.atan2(-(sy), sx))  # Norte ↑
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

# ──────────────────────────────────────────────────────────────────────────────
# Segmentación por filas + OCR de columna “Apellidos Nombre/Razón social”
# ──────────────────────────────────────────────────────────────────────────────
def split_rows_and_centers(bgr: np.ndarray) -> List[dict]:
    """
    Divide la zona recortada en 4 bandas verticales (filas). Para cada banda:
      - centro verde principal (si aparece)
      - centro rosa vecino más cercano (si aparece)
      - lado por vector (verde→rosa)
    """
    crop, (ox, oy) = crop_map(bgr)
    h, w = crop.shape[:2]
    band_h = h // 4
    mg, mp = color_masks(crop)
    g_pts = contours_centroids(mg, min_area=(400 if FAST_MODE else 250))
    p_pts = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))

    rows: List[dict] = []
    for ri in range(4):
        y0 = ri * band_h
        y1 = (ri + 1) * band_h if ri < 3 else h
        # centro verde dentro de la banda (coge el más grande en banda)
        g_in = [(x,y,a) for (x,y,a) in g_pts if y0 <= y < y1]
        main_center = None
        if g_in:
            main = max(g_in, key=lambda t: t[2])
            main_center = (main[0] + ox, main[1] + oy)
        # vecino rosa más cercano a main
        neigh_center = None
        side = ""
        if main_center:
            cand = [(x+ox, y+oy) for (x,y,_a) in p_pts if y0 <= y < y1]
            if cand:
                # vecino más próximo en euclídea
                best = min(cand, key=lambda p: (p[0]-main_center[0])**2 + (p[1]-main_center[1])**2)
                neigh_center = best
                side = side_of(main_center, neigh_center)
        rows.append({
            "row_y": (y0 + y1)//2 + oy,
            "main_center": main_center,
            "neigh_center": neigh_center,
            "side": side
        })
    return rows

def ocr_text(img: np.ndarray, psm: int = 6) -> str:
    cfg = f"--psm {psm} --oem 3"
    return pytesseract.image_to_string(img, config=cfg) or ""

def extract_owner_from_column(bgr: np.ndarray, row_band: Tuple[int,int]) -> Tuple[str, Tuple[int,int,int,int], int]:
    """
    Extrae el titular por OCR en la columna derecha del croquis, dentro de
    la banda 'row_band' (y0,y1) en coordenadas ABSOLUTAS del bgr completo.
    Devuelve (owner, roi_abs, attempts).
    """
    h, w = bgr.shape[:2]
    # ventana columna derecha (55%..97% del ancho)
    x0 = int(w * 0.55)
    x1 = int(w * 0.97)

    y0_abs, y1_abs = row_band
    # Zona superior de la banda, donde suelen ir “Apellidos Nombre/Razón social”
    y0 = max(0, y0_abs + int((y1_abs - y0_abs) * 0.15))
    y1 = max(y0 + 60, y0_abs + int((y1_abs - y0_abs) * 0.45))

    attempts = 0
    for (xx0, xx1) in [(x0, x1), (int(w*0.50), int(w*0.97))]:
        attempts += 1
        roi = bgr[y0:y1, xx0:xx1]
        if roi.size == 0:
            continue
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
        flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
        _, bw  = cv2.threshold(g, 0 if THRESH_OTSU else 127, 255, flags_bin)

        txt = ocr_text(bw, psm=6)
        lines = [clean_name_line(l.upper()) for l in txt.splitlines()]
        lines = [l for l in lines if l and is_upper_name(l)]
        owner = ""
        if lines:
            owner = lines[0]
            # fusión con 2ª línea si es palabra corta (tipo “LUIS”)
            for l2 in lines[1:2]:
                owner = merge_second_token_if_short(owner, l2)
            if owner:
                return owner, (xx0, y0, xx1, y1), attempts
    return "", (x0, y0, x1, y1), attempts

def assign_linderos_with_rows(bgr: np.ndarray) -> Tuple[Dict[str,str], dict, np.ndarray]:
    """
    1) Segmenta filas, calcula lado por vector verde→rosa.
    2) Por cada fila, extrae owner por OCR de columna.
    """
    vis = bgr.copy()
    crop, (ox, oy) = crop_map(bgr)
    h, w = crop.shape[:2]
    band_h = h // 4

    # Dibuja guías y etiquetas cardinales (en /preview?labels=1)
    rows = split_rows_and_centers(bgr)

    linderos = {"norte":"","sur":"","este":"","oeste":""}
    owners_rows: List[dict] = []

    for idx, r in enumerate(rows):
        # límites absolutos de banda
        y0_abs = oy + idx*band_h
        y1_abs = oy + ((idx+1)*band_h if idx < 3 else oy + h)

        owner, roi, attempts = extract_owner_from_column(bgr, (y0_abs, y1_abs))
        if owner:
            if r["side"] in linderos and not linderos[r["side"]]:
                linderos[r["side"]] = owner

        owners_rows.append({
            "row_y": (y0_abs + y1_abs)//2,
            "main_center": r["main_center"],
            "neigh_center": r["neigh_center"],
            "side": r["side"],
            "owner": owner,
            "roi": roi,
            "attempts": attempts
        })

        # visual
        xx0, yy0, xx1, yy1 = roi
        cv2.rectangle(vis, (xx0,yy0), (xx1,yy1), (255,255,255), 2)
        if r["main_center"]:
            cv2.circle(vis, r["main_center"], 9, (0,255,0), -1)
        if r["neigh_center"]:
            cv2.circle(vis, r["neigh_center"], 7, (0,0,255), -1)
        if r["side"]:
            label = r["side"][:1].upper()
            put = (xx0+6, yy0+22)
            cv2.putText(vis, label, put, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    dbg = {
        "rows": owners_rows
    }
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "version": app.version, "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY,
            "cv2_flags":{"OTSU": bool(THRESH_OTSU)}}

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...), labels: bool = Query(False)):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, dbg, vis = assign_linderos_with_rows(bgr)

        # Si labels=1, sobreimprime N/S/E/O grandes en la columna izquierda
        if labels:
            crop, (ox, oy) = crop_map(bgr)
            h, w = crop.shape[:2]
            band_h = h // 4
            label_map = {"norte":"N","sur":"S","este":"E","oeste":"O"}
            rows = split_rows_and_centers(bgr)
            for idx, r in enumerate(rows):
                y_mid = oy + idx*band_h + band_h//2
                x_label = ox + 30
                lab = label_map.get(r["side"], "?")
                cv2.putText(vis, lab, (x_label, y_mid), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255,255,255), 3, cv2.LINE_AA)
    except Exception as e:
        # mini-imagen con error (evita 5xx)
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
def preview_post(data: ExtractIn = Body(...), labels: bool = Query(False)):
    return preview_get(pdf_url=data.pdf_url, labels=labels)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto (páginas ≥2): mapa parcela→titular (respaldo general)
    parcel2owner = extract_owners_map_text(pdf_bytes)

    if TEXT_ONLY:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = "Modo TEXT_ONLY activo: mapa desactivado."
        dbg = {"TEXT_ONLY": True, "owners_by_parcel_sample": dict(list(parcel2owner.items())[:6])} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected, note=note, debug=dbg)

    # 2) Visión (pág. 2): filas → lado → OCR columna (una línea + posible token corto)
    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = assign_linderos_with_rows(bgr)

        # owners_detected por si se desea lista
        owners_detected = [v for v in linderos.values() if v]
        owners_detected = list(dict.fromkeys(owners_detected))

        note = None if any(linderos.values()) else "No se pudo determinar lado/vecino con suficiente confianza."
        dbg = vdbg
        # Adjunta una pequeña muestra de texto de la p.2 para diagnósticos
        if debug:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    p2 = normalize_text(pdf.pages[1].extract_text() or "")
                    sample = "\n".join(p2.split("\n")[:28])
            except Exception:
                sample = ""
            dbg["p2_text_sample"] = sample.split("\n") if sample else []
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=(dbg if debug else None))

    except Exception as e:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = f"Excepción visión/OCR: {e}"
        dbg = {"exception": str(e)} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected, note=note, debug=dbg)




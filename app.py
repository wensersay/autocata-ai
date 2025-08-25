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
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.4.6")

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
    # Quita marcas de provincia/municipio y paréntesis
    s = s.strip()
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\([^)]*\)", "", s)

    # Elimina DNIs completos
    s = DNI_RE.sub("", s)

    # Separa en tokens para filtrar abreviaturas
    toks = [t for t in re.split(r"\s+", s) if t]

    # Quita números y abreviaturas demasiado cortas salvo conectores comunes
    keep_short = {"DE","LA","DEL","LOS","LAS","DA","DO","EL","Y"}
    cleaned = []
    for t in toks:
        if any(ch.isdigit() for ch in t):
            continue
        t2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.']", "", t.upper())
        if not t2:
            continue
        if len(t2) <= 2 and t2 not in keep_short:
            continue
        cleaned.append(t2)

    s = " ".join(cleaned)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Corta a 26 chars (regla empírica)
    if len(s) > 26:
        s = s[:26].rstrip()
    return s

def merge_second_token_if_short(first: str, second: str) -> str:
    t = (second or "").strip()
    if 1 <= len(t) <= 10 and re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ]+", t):
        if not first.endswith(" " + t):
            return (first + " " + t).strip()
    return first

# Respaldo (solo texto) por si lo quieres usar en TEXT_ONLY
def extract_owners_map_text(pdf_bytes: bytes) -> Dict[str, str]:
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
                if "PARCELA" in up:
                    tokens = [t for t in up.replace(",", " ").split() if t.isdigit()]
                    if tokens:
                        curr_parcel = tokens[-1]
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
                        s_clean = clean_name_line(U)
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
# OpenCV helpers (con fallbacks)
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
    ang = math.degrees(math.atan2(-(sy), sx))  # Norte ↑
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

def split_rows_and_centers(bgr: np.ndarray) -> List[dict]:
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
        g_in = [(x,y,a) for (x,y,a) in g_pts if y0 <= y < y1]
        main_center = None
        if g_in:
            main = max(g_in, key=lambda t: t[2])
            main_center = (main[0] + ox, main[1] + oy)
        neigh_center = None
        side = ""
        if main_center:
            cand = [(x+ox, y+oy) for (x,y,_a) in p_pts if y0 <= y < y1]
            if cand:
                best = min(cand, key=lambda p: (p[0]-main_center[0])**2 + (p[1]-main_center[1])**2)
                neigh_center = best
                side = side_of(main_center, neigh_center)
        rows.append({
            "row_y": (y0 + y1)//2 + oy,
            "y_abs": (y0 + oy, y1 + oy),
            "main_center": main_center,
            "neigh_center": neigh_center,
            "side": side
        })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Extracción de titular por TEXTO en la misma banda/columna
# ──────────────────────────────────────────────────────────────────────────────
def prepare_page2_words(pdf_bytes: bytes):
    pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    page = pdf.pages[1]  # página 2 (index 1)
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False, x_tolerance=2, y_tolerance=2)
    return pdf, page, words

def cluster_lines(words: List[dict]) -> List[str]:
    if not words:
        return []
    # Agrupa por 'top' (desde arriba, en puntos)
    rows = defaultdict(list)
    for w in words:
        key = int(round(w["top"]/2.0))*2  # bin suave
        rows[key].append(w)
    out = []
    for key in sorted(rows.keys()):
        ws = sorted(rows[key], key=lambda z: z["x0"])
        out.append(" ".join([w["text"] for w in ws]))
    return out

def extract_owner_from_text_band(words_all: List[dict],
                                 page_w: float, page_h: float,
                                 img_w: int, img_h: int,
                                 band_px: Tuple[int,int],
                                 col_px: Tuple[int,int]) -> Tuple[str, dict]:
    """Filtra palabras dentro de la banda (y) y columna (x) mapeando píxeles→puntos."""
    sx = img_w / float(page_w)
    sy = img_h / float(page_h)

    x0_pt = col_px[0] / sx
    x1_pt = col_px[1] / sx
    y0_pt = band_px[0] / sy  # top-from-top
    y1_pt = band_px[1] / sy  # bottom-from-top

    # Filtra palabras por caja
    words = [w for w in words_all
             if (w["x0"] >= x0_pt and w["x1"] <= x1_pt and
                 w["top"] >= y0_pt and w["bottom"] <= y1_pt)]

    lines = cluster_lines(words)
    debug = {"lines": lines[:8], "x0_pt": x0_pt, "x1_pt": x1_pt, "y0_pt": y0_pt, "y1_pt": y1_pt}

    # Busca línea "PARCELA" y luego la 1ª línea con DNI o candidata a nombre
    idx_par = -1
    for i, L in enumerate(lines):
        if "PARCELA" in L.upper():
            idx_par = i
            break

    scan_range = range(idx_par+1, min(idx_par+10, len(lines))) if idx_par >= 0 else range(0, min(12, len(lines)))
    owner1 = ""
    owner2 = ""
    # preferimos la que contenga DNI; si no, la primera MAYÚSCULA razonable
    for j in scan_range:
        rawU = lines[j].upper().strip()
        if not rawU:
            continue
        has_dni = bool(DNI_RE.search(rawU))
        cand = clean_name_line(rawU)
        if not cand:
            continue
        if has_dni:
            owner1 = cand
            # mira si la siguiente es un token corto (tipo LUIS)
            if j+1 < len(lines):
                cand2 = clean_name_line(lines[j+1].upper())
                if cand2:
                    owner1 = merge_second_token_if_short(owner1, cand2)
            break
        if not owner1 and is_upper_name(cand):
            owner1 = cand
            # intenta fusionar siguiente corta
            if j+1 < len(lines):
                cand2 = clean_name_line(lines[j+1].upper())
                if cand2:
                    owner1 = merge_second_token_if_short(owner1, cand2)
            # no rompemos por si aparece una con DNI un poco más abajo
            owner2 = owner1

    owner = owner1 or owner2 or ""
    return owner, debug

# ──────────────────────────────────────────────────────────────────────────────
# Asignación de linderos combinando visión (lados) + texto (titulares)
# ──────────────────────────────────────────────────────────────────────────────
def assign_linderos_with_rows_text(pdf_bytes: bytes, bgr: np.ndarray) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    rows = split_rows_and_centers(bgr)

    # Columna derecha en píxeles (igual que en previas)
    h, w = bgr.shape[:2]
    col_x0 = int(w * 0.55)
    col_x1 = int(w * 0.97)

    # Prepara texto de página 2 una sola vez
    pdf, page, words_all = prepare_page2_words(pdf_bytes)
    try:
        linderos = {"norte":"","sur":"","este":"","oeste":""}
        owners_rows = []

        for r in rows:
            y0_abs, y1_abs = r["y_abs"]  # banda en píxeles (top..bottom)
            owner, dbg_txt = extract_owner_from_text_band(
                words_all, page.width, page.height, w, h,
                (y0_abs, y1_abs), (col_x0, col_x1)
            )
            if owner and r["side"] in linderos and not linderos[r["side"]]:
                linderos[r["side"]] = owner

            # Visual para depurar
            cv2.rectangle(vis, (col_x0, y0_abs), (col_x1, y1_abs), (255,255,255), 2)
            if r["main_center"]:
                cv2.circle(vis, r["main_center"], 9, (0,255,0), -1)
            if r["neigh_center"]:
                cv2.circle(vis, r["neigh_center"], 7, (0,0,255), -1)
            if r["side"]:
                lab = r["side"][:1].upper()
                cv2.putText(vis, lab, (col_x0+6, y0_abs+24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

            owners_rows.append({
                "row_y": r["row_y"],
                "main_center": r["main_center"],
                "neigh_center": r["neigh_center"],
                "side": r["side"],
                "owner": owner,
                "band_px": [y0_abs, y1_abs],
                "col_px": [col_x0, col_x1],
                "text_debug": dbg_txt
            })

        dbg = {"rows": owners_rows}
        return linderos, dbg, vis
    finally:
        try:
            pdf.close()
        except Exception:
            pass

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
        # Usamos solo visión para dibujar filas y lados + cajas columna
        linderos, dbg, vis = assign_linderos_with_rows_text(pdf_bytes, bgr)

        if labels:
            crop, (ox, oy) = crop_map(bgr)
            h, w = crop.shape[:2]
            band_h = h // 4
            rows = split_rows_and_centers(bgr)
            label_map = {"norte":"N","sur":"S","este":"E","oeste":"O"}
            for idx, r in enumerate(rows):
                y_mid = oy + idx*band_h + band_h//2
                x_label = ox + 30
                lab = label_map.get(r["side"], "?")
                cv2.putText(vis, lab, (x_label, y_mid), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255,255,255), 3, cv2.LINE_AA)
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
def preview_post(data: ExtractIn = Body(...), labels: bool = Query(False)):
    return preview_get(pdf_url=data.pdf_url, labels=labels)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # Respaldo solo texto si se quisiera activar
    parcel2owner = extract_owners_map_text(pdf_bytes)

    if TEXT_ONLY:
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        note = "Modo TEXT_ONLY activo: mapa desactivado."
        dbg = {"TEXT_ONLY": True, "owners_by_parcel_sample": dict(list(parcel2owner.items())[:6])} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected, note=note, debug=dbg)

    # Visión para lados + TEXTO (no OCR) para titular por banda/columna
    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = assign_linderos_with_rows_text(pdf_bytes, bgr)

        owners_detected = [v for v in linderos.values() if v]
        owners_detected = list(dict.fromkeys(owners_detected))
        note = None if any(linderos.values()) else "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg
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



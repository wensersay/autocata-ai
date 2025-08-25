# app.py — AutoCatastro AI v0.4.1 (revisada)
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

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.4.1")

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

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION",
    "REFERENCIA CATASTRAL", "NIF", "DOMICILIO"
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
    # descartar líneas obvias que no son nombre
    for bad in STOP_IN_NAME:
        if bad in U:
            return False
    # demasiados dígitos → no es nombre
    if sum(ch.isdigit() for ch in line) >= 1:
        return False
    # longitud máxima típica observada
    if len(line) > 26:
        return False
    return bool(UPPER_NAME_RE.match(line))

def owners_by_blocks_page2(pdf_bytes: bytes) -> List[str]:
    """
    Divide la página 2 en 'bloques' usando ocurrencias de 'Referencia catastral'
    y, dentro de cada bloque, tras 'Titularidad principal' toma 1–2 líneas en
    MAYÚSCULAS (≤26 chars, sin dígitos). Devuelve una lista de titulares por orden.
    """
    owners: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) < 2:
            return owners
        p2 = pdf.pages[1]
        text = normalize_text(p2.extract_text(x_tolerance=2, y_tolerance=2) or "")
        lines = text.split("\n")

    # índices donde empiezan bloques (aparición de "Referencia catastral")
    idxs = [i for i, ln in enumerate(lines) if "REFERENCIA CATASTRAL" in ln.upper()]
    if not idxs:
        return owners
    idxs.append(len(lines))  # centinela

    for b in range(len(idxs) - 1):
        lo, hi = idxs[b], idxs[b+1]
        block = lines[lo:hi]

        # buscar punto de arranque tras cabecera "Titularidad principal" o
        # cabecera de tabla "Apellidos Nombre / Razón social"
        start = 0
        for j, ln in enumerate(block):
            U = ln.upper()
            if "TITULARIDAD PRINCIPAL" in U or ("APELLIDOS NOMBRE" in U and "RAZON" in U):
                start = j + 1
                break

        # Ventana de hasta 12 líneas desde 'start' (permite separación por NIF/Domicilio)
        win = block[start:start + 12]

        picks: List[str] = []
        for ln in win:
            if is_upper_name(ln):
                picks.append(ln.strip())
            if len(picks) >= 2:
                break

        if picks:
            # unir si hay dos líneas de nombre (e.g. “… JOSE” + “LUIS”)
            owner = " ".join(picks)
            owner = re.sub(r"\s{2,}", " ", owner).strip()
            owners.append(owner)
        else:
            owners.append("")  # no encontrado en este bloque

    return owners

# ──────────────────────────────────────────────────────────────────────────────
# OpenCV helpers (seguros en entornos slim)
# ──────────────────────────────────────────────────────────────────────────────
def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
MORPH_OPEN        = cv_flag("MORPH_OPEN", 2)
MORPH_CLOSE       = cv_flag("MORPH_CLOSE", 3)
COLOR_BGR2HSV     = cv_flag("COLOR_BGR2HSV", 40)

# ──────────────────────────────────────────────────────────────────────────────
# Visión por computador (página 2)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 380 if FAST_MODE else 520
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def crop_maps_band(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    """
    Recorta la banda horizontal donde aparecen mini-mapas (izquierda).
    Devuelve (crop, (ox,oy)) en coords de página.
    """
    h, w = bgr.shape[:2]
    top = int(h * 0.18); bottom = int(h * 0.95)   # fuera cabecera y pie
    left = int(w * 0.05); right = int(w * 0.40)   # banda de mini-mapas
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    return bgr[top:bottom, left:right], (left, top)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, COLOR_BGR2HSV)
    # Verde (parcela propia)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # Rosa (vecinos)
    p_ranges = [
        (np.array([160, 20, 80], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 80], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges:
        mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges:
        mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))

    k3 = np.ones((3,3), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, MORPH_OPEN, k3)
    mg = cv2.morphologyEx(mg, MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, MORPH_OPEN, k3)
    mp = cv2.morphologyEx(mp, MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: x[1])  # por Y ascendente (fila superior→inferior)
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

def segment_rows_and_sides(bgr: np.ndarray) -> List[dict]:
    """
    Detecta filas (mini-mapas) y, por fila, calcula:
    - centro verde (parcela objetivo) si existe
    - centro rosa más cercano (vecino)
    - lado (N/S/E/O) según vector verde→rosa
    Devuelve lista de dicts ordenados de arriba a abajo.
    """
    band, (ox, oy) = crop_maps_band(bgr)
    mg, mp = color_masks(band)

    # Áreas mínimas bajas para no perder verdes pequeños
    min_g = 120 if FAST_MODE else 90
    min_p = 120 if FAST_MODE else 90

    greens = contours_centroids(mg, min_g)
    pinks  = contours_centroids(mp, min_p)

    rows: List[dict] = []

    # Si no se detecta verde en alguna fila, intentaremos usar solo rosa (menos fiable)
    # Emparejamos por proximidad vertical
    all_y = sorted(set([gy for _,gy,_ in greens] + [py for _,py,_ in pinks]))
    if not all_y:
        return rows

    # Agrupar por Y con umbral (altura típica de mini-mapa)
    groups: List[List[int]] = []
    thr = 140  # píxeles
    for y in all_y:
        if not groups or abs(y - groups[-1][-1]) > thr:
            groups.append([y])
        else:
            groups[-1].append(y)

    for g in groups:
        yc = int(sum(g)/len(g))
        # mejor centro verde de la banda alrededor de yc
        gv = None; best_d = 10**9
        for (xg, yg, _a) in greens:
            d = abs(yg - yc)
            if d < best_d:
                best_d = d; gv = (xg, yg)
        # mejor vecino rosa junto a ese verde (o al yc si gv es None)
        nv = None; best2 = 10**9
        anchor_y = gv[1] if gv else yc
        for (xp, yp, _a) in pinks:
            d = abs(yp - anchor_y)
            if d < best2:
                best2 = d; nv = (xp, yp)
        side = ""
        if gv and nv:
            side = side_of((gv[0]+ox, gv[1]+oy), (nv[0]+ox, nv[1]+oy))

        rows.append({
            "row_index": len(rows),
            "main_center": (gv[0]+ox, gv[1]+oy) if gv else None,
            "neigh_center": (nv[0]+ox, nv[1]+oy) if nv else None,
            "side": side
        })

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Emparejar filas ↔ titulares y formar linderos
# ──────────────────────────────────────────────────────────────────────────────
def assign_linderos_from_rows(rows: List[dict], owners_seq: List[str]) -> Tuple[Dict[str,str], List[dict]]:
    """
    Cruza filas detectadas (ordenadas arriba→abajo) con lista de titulares por
    bloque de texto (también arriba→abajo). Asigna linderos por lado.
    """
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    debug_rows: List[dict] = []

    n = min(len(rows), len(owners_seq))
    for i in range(n):
        side = rows[i].get("side") or ""
        owner = owners_seq[i].strip()
        row_dbg = dict(rows[i])
        row_dbg["owner"] = owner
        debug_rows.append(row_dbg)

        if side and owner and not linderos.get(side):
            linderos[side] = owner

    return linderos, debug_rows

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "version": app.version, "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY}

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...), labels: int = Query(0)):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    bgr = page2_bgr(pdf_bytes)

    rows = segment_rows_and_sides(bgr)

    vis = bgr.copy()
    # Dibujo
    for r in rows:
        if r["main_center"]:
            cv2.circle(vis, r["main_center"], 10, (0,255,0), -1)
        if r["neigh_center"]:
            cv2.circle(vis, r["neigh_center"], 8, (0,0,255), -1)
        if labels and r["side"]:
            ch = {"norte":"N","sur":"S","este":"E","oeste":"O"}[r["side"]]
            pt = r["neigh_center"] or r["main_center"]
            if pt:
                cv2.putText(vis, ch, (pt[0]+6, pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...), labels: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 1) Texto página 2 → titulares por bloque
    owners_seq = owners_by_blocks_page2(pdf_bytes)

    if TEXT_ONLY:
        owners_detected = [o for o in owners_seq if o][:8]
        note = "Modo TEXT_ONLY activo."
        dbg = {"owners_blocks": owners_seq} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","este":"","oeste":""},
                          owners_detected=owners_detected, note=note, debug=dbg)

    # 2) Visión: filas y lados
    bgr = page2_bgr(pdf_bytes)
    rows = segment_rows_and_sides(bgr)

    # 3) Emparejar por orden y formar linderos
    linderos, dbg_rows = assign_linderos_from_rows(rows, owners_seq)
    owners_detected = [o for o in owners_seq if o][:8]

    note = None
    if not any(linderos.values()):
        note = "No se pudo determinar lado/vecino con suficiente confianza."

    dbg = None
    if debug:
        dbg = {
            "rows": dbg_rows,
            "owners_rows": owners_seq
        }

    return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)




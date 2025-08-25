from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse, JSONResponse
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
app = FastAPI(title="AutoCatastro AI", version="0.4.0")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")  # aquí no se usa, pero lo mantenemos

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
    "POLÍGONO", "POLIGONO", "PARCELA",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA",
    "TITULARIDAD", "VALOR CATASTRAL",
    "LOCALIZACIÓN", "LOCALIZACION",
    "REFERENCIA CATASTRAL", "APELLIDOS NOMBRE", "RAZON SOCIAL",
    "NIF", "DOMICILIO"
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
    if not line: return False
    U = line.upper()
    if any(bad in U for bad in STOP_IN_NAME): return False
    if any(ch.isdigit() for ch in U): return False
    return bool(UPPER_NAME_RE.match(U)) and len(line) <= 26  # regla de 26 chars

def reconstruct_owner_candidates(lines: List[str]) -> str:
    """
    Toma hasta 2 líneas tipo nombre en MAYÚSCULAS (≤26 c/u) y las une si cabe.
    Evita números y rótulos.
    """
    cands = [ln.strip() for ln in lines if is_upper_name(ln)]
    if not cands:
        return ""
    name = cands[0]
    if len(cands) >= 2 and len(name) + 1 + len(cands[1]) <= 26:
        name = f"{name} {cands[1]}"
    return name

def owners_by_rows_from_page2(pdf_bytes: bytes) -> List[str]:
    """
    Devuelve lista de titulares por orden de aparición en la página 2.
    Estrategia: localizar bloques tras 'Titularidad principal' / cabecera de tabla
    y recoger 1–2 líneas en MAYÚSCULAS (≤26), ignorando NIF/Domicilio/números.
    """
    owners: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) < 2:
            return owners
        page = pdf.pages[1]  # página 2 (0-index)
        text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
        lines = normalize_text(text).split("\n")

        i = 0
        while i < len(lines):
            U = lines[i].strip().upper()
            # detectar cabecera que precede al nombre
            if ("TITULARIDAD PRINCIPAL" in U) or ("APELLIDOS NOMBRE" in U and "RAZON" in U):
                j = i + 1
                window: List[str] = []
                steps = 0
                # escanear como mucho 10 líneas tras la cabecera
                while j < len(lines) and steps < 10:
                    raw = lines[j].strip()
                    UU  = raw.upper()
                    if not raw: 
                        j += 1; steps += 1; continue
                    # cortar si vemos rótulos claros de otra sección
                    if any(tag in UU for tag in ("REFERENCIA CATASTRAL", "RELACIÓN DE PARCELAS", "RELACION DE PARCELAS")):
                        break
                    # ignorar explicitamente columnas de tabla
                    if any(tag in UU for tag in ("APELLIDOS NOMBRE", "RAZON SOCIAL", "NIF", "DOMICILIO")):
                        j += 1; steps += 1; continue
                    # recoger candidatos de nombre (MAYÚSCULAS ≤26)
                    if is_upper_name(raw):
                        window.append(raw)
                        # si ya tenemos 2 líneas, paramos
                        if len(window) >= 2:
                            break
                    j += 1; steps += 1

                owner = reconstruct_owner_candidates(window)
                if owner:
                    owners.append(owner)
                i = j
                continue
            i += 1

    return owners

# ──────────────────────────────────────────────────────────────────────────────
# Visión por computador (página 2)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 360 if FAST_MODE else 520
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (mask_verde_principal, mask_rosa_vecinos).
    Rangos amplios para aguantar distintas impresiones/escaneos.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160, 20, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))

    k3 = np.ones((3,3), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int,Tuple[int,int,int,int]]]:
    """
    Devuelve lista de (cx,cy,area, bbox) de contornos suficientemente grandes.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        x,y,w,h = cv2.boundingRect(c)
        out.append((cx, cy, int(a), (x,y,w,h)))
    out.sort(key=lambda x: x[1])  # por Y ascendente
    return out

def cluster_rows_by_y(points: List[Tuple[int,int]], thr: int) -> List[List[int]]:
    """
    Agrupa índices de puntos por proximidad vertical (simple binning).
    """
    if not points: return []
    # points: [(cx,cy), ...] ya en orden por Y
    rows: List[List[int]] = [[0]]
    for idx in range(1, len(points)):
        _, prev_y = points[idx-1]
        _, this_y = points[idx]
        if abs(this_y - prev_y) > thr:
            rows.append([idx])
        else:
            rows[-1].append(idx)
    return rows

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

def annotate_preview(img: np.ndarray, rows_info: List[dict], draw_names: bool):
    for r in rows_info:
        main = r.get("main_center")
        neigh = r.get("neigh_center")
        side = r.get("side","")
        owner = r.get("owner","")
        if main:
            cv2.circle(img, tuple(main), 9, (0,255,0), -1)
        if neigh:
            cv2.circle(img, tuple(neigh), 9, (0,0,255), -1)
        if neigh and side:
            label = {"norte":"N","sur":"S","este":"E","oeste":"O"}.get(side, "?")
            cv2.putText(img, label, (neigh[0]+8, neigh[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            if draw_names and owner:
                cv2.putText(img, owner, (neigh[0]+20, neigh[1]+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def detect_rows_and_sides(bgr: np.ndarray) -> Tuple[List[dict], np.ndarray]:
    """
    Detecta filas a partir de manchas ROSA; por cada fila, busca la mayor mancha VERDE
    cercana, calcula lado con el vector verde→rosa y devuelve estructura por-fila.
    """
    vis = bgr.copy()
    H, W = bgr.shape[:2]

    mg, mp = color_masks(bgr)

    # Contornos
    g_list = contours_centroids(mg, min_area=(250 if FAST_MODE else 180))
    p_list = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))

    # Si no hay vecinos rosa, no hay filas
    if not p_list:
        return [], vis

    # Agrupar ROSA por filas (proximidad vertical)
    p_points = [(cx,cy) for (cx,cy,_a,_bb) in p_list]
    row_thr = max(18, int(H * 0.06))
    p_rows_idx = cluster_rows_by_y(p_points, thr=row_thr)

    rows_info: List[dict] = []
    for ridx, idxs in enumerate(p_rows_idx):
        # centroid medio rosa de la fila (o el mayor área de la fila)
        idxs_sorted = sorted(idxs, key=lambda i: -p_list[i][2])
        i_top = idxs_sorted[0]
        neigh_cx, neigh_cy, _a, bb = p_list[i_top]
        neigh_center = [int(neigh_cx), int(neigh_cy)]

        # buscar VERDE más cercano por distancia euclídea en esta banda vertical
        best_g = None
        best_d = 10**9
        for (gcx,gcy,ga,_gbb) in g_list:
            dy = abs(gcy - neigh_cy)
            if dy > row_thr:  # fuera de banda vertical de la fila
                continue
            d = (gcx - neigh_cx)**2 + (gcy - neigh_cy)**2
            if d < best_d:
                best_d = d
                best_g = (gcx, gcy, ga)

        main_center = None
        side = ""
        if best_g:
            mcx, mcy, _ = best_g
            main_center = [int(mcx), int(mcy)]
            side = side_of((mcx, mcy), (neigh_cx, neigh_cy))

        rows_info.append({
            "row_index": ridx,
            "main_center": main_center,
            "neigh_center": neigh_center,
            "side": side,
            "bbox_hint": bb
        })

    return rows_info, vis

# ──────────────────────────────────────────────────────────────────────────────
# Lógica completa de extracción (v0.4.0)
# ──────────────────────────────────────────────────────────────────────────────
def extract_v040(pdf_bytes: bytes, want_preview: bool=False, draw_labels: bool=False):
    """
    1) Texto (pág. 2) → lista de titulares por filas.
    2) Visión (pág. 2) → filas visuales con lado (N/S/E/O).
    3) Emparejar por orden (fila 0 con fila 0, etc.).
    """
    # (1) titulares por filas
    owners_rows = owners_by_rows_from_page2(pdf_bytes)  # e.g. ["VAZQUEZ POMBO DOSINDA", "MOSQUERA ...", ...]
    # (2) filas visuales con lado
    bgr = page2_bgr(pdf_bytes)
    rows_info, vis = detect_rows_and_sides(bgr)

    # emparejar por orden top→bottom
    for i, r in enumerate(rows_info):
        r["owner"] = owners_rows[i] if i < len(owners_rows) else ""

    # construir linderos
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    for r in rows_info:
        sd = r.get("side","")
        nm = r.get("owner","")
        if sd and nm and not linderos[sd]:
            linderos[sd] = nm

    # owners_detected (deduplicado, top-8)
    owners_detected = []
    for r in rows_info:
        nm = r.get("owner","")
        if nm and nm not in owners_detected:
            owners_detected.append(nm)
        if len(owners_detected) >= 8: break

    debug = {
        "rows": rows_info,
        "owners_rows": owners_rows[:8]
    }

    if want_preview:
        annotate_preview(vis, rows_info, draw_labels)
        ok, png = cv2.imencode(".png", vis)
        if not ok:
            raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
        return png.tobytes()

    return linderos, owners_detected, debug, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True, "version": app.version,
        "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(pdf_url: AnyHttpUrl = Query(...), labels: int = Query(0)):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        png_bytes = extract_v040(pdf_bytes, want_preview=True, draw_labels=(labels == 1))
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        # mini-imagen con el error para no romper healthchecks
        err = str(e)
        blank = np.zeros((240, 800, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:70]}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...), labels: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))
    try:
        linderos, owners_detected, dbg, _vis = extract_v040(pdf_bytes, want_preview=False)
        return ExtractOut(
            linderos=linderos,
            owners_detected=owners_detected,
            note=None if any(linderos.values()) else "No se pudo determinar lado/vecino con suficiente confianza.",
            debug=dbg if debug else None
        )
    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note=f"Excepción v0.4.0: {e}",
            debug={"exception": str(e)} if debug else None
        )





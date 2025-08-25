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
app = FastAPI(title="AutoCatastro AI", version="0.4.3")

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
# Utilidades PDF / Texto (titulares POR COLUMNA en pág. 2)
# ──────────────────────────────────────────────────────────────────────────────
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

UPPER_TOKEN = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\.\-\' ]+$")

def owners_by_column_page2(pdf_bytes: bytes) -> List[dict]:
    """
    Lee SOLO la columna 'Apellidos Nombre / Razón social' bajo su cabecera.
    Devuelve [{y, x0, x1, name}] por fila detectada (y ~ línea de cabecera).
    """
    owners: List[dict] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if len(pdf.pages) < 2:
                return owners
            p2 = pdf.pages[1]
            words = p2.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
    except Exception:
        return owners

    from collections import defaultdict
    rows = defaultdict(list)
    for w in words:
        rows[int(round(w["top"]))].append(w)

    headers = []  # (y_hdr, x0_name, x0_nif)
    for y, ws in rows.items():
        tokens = [w["text"].upper() for w in ws]
        if "NIF" in tokens and any(t in tokens for t in ("APELLIDOS","NOMBRE","RAZÓN","RAZON","SOCIAL")):
            try:
                x0_nif  = min(w["x0"] for w in ws if w["text"].upper() == "NIF")
                x0_name = min(w["x0"] for w in ws if w["text"].upper() in ("APELLIDOS","NOMBRE","RAZÓN","RAZON","SOCIAL","/"))
                headers.append((y, x0_name, x0_nif))
            except ValueError:
                continue
    headers.sort(key=lambda t: t[0])
    if not headers:
        return owners

    # Para cada cabecera, leer palabras en la banda vertical de esa "tarjeta"
    for i, (y_hdr, x0_name, x0_nif) in enumerate(headers):
        y_lo = y_hdr + 2
        # a falta de otra cabecera: ventana conservadora
        y_hi = headers[i+1][0] - 2 if i + 1 < len(headers) else y_hdr + (140 if FAST_MODE else 170)

        col_words = [w for w in words
                     if y_lo <= w["top"] <= y_hi and (x0_name - 2) <= w["x0"] <= (x0_nif - 2)]
        col_words.sort(key=lambda w: (w["top"], w["x0"]))

        toks = []
        for w in col_words:
            t = (w["text"] or "").strip()
            if not t or any(ch.isdigit() for ch in t):  # corta números/NIF dentro de la columna
                continue
            if UPPER_TOKEN.match(t):
                toks.append(t)

        txt = " ".join(toks)
        # corta en posibles patrones numéricos largos o NIF
        txt = re.split(r"\b\d{5,}|\b\d{8}[A-Z]\b", txt)[0].strip()
        # Límite 26 chars respetando palabra
        if len(txt) > 26:
            cut = txt[:26]
            txt = cut[:cut.rfind(" ")] if " " in cut else cut

        if txt:
            owners.append({"y": y_hdr, "x0": x0_name, "x1": x0_nif, "name": txt})
    return owners

def pick_owner_for_y(owners_rows: List[dict], y: int) -> str:
    if not owners_rows:
        return ""
    best = min(owners_rows, key=lambda o: abs(o["y"] - y))
    return (best.get("name") or "").strip()

# ──────────────────────────────────────────────────────────────────────────────
# Visión en pág. 2 (N/S/E/O por verde→rosa)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 360 if FAST_MODE else 520
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def crop_map(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Recorte general para la zona de listados con mini-mapas."""
    h, w = bgr.shape[:2]
    top = int(h * 0.12); bottom = int(h * 0.96)
    left = int(w * 0.06); right = int(w * 0.92)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    if bottom - top < 100 or right - left < 100:
        return bgr, (0, 0)
    return bgr[top:bottom, left:right], (left, top)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde agua (parcela propia)
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
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int = 220) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]) ; cy = int(M["m01"] / M["m00"]) ;
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: x[1])  # Y ascendente (fila arriba→abajo)
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

def detect_rows_and_sides(bgr: np.ndarray) -> List[dict]:
    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    greens = contours_centroids(mg, min_area=(380 if FAST_MODE else 260))
    pinks  = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))

    rows: List[dict] = []
    if not greens:
        return rows

    for gi, (gx, gy, _ga) in enumerate(greens):
        g_abs = (gx + ox, gy + oy)
        # rosa más cercano
        best = None
        bestd = 1e9
        for (px, py, _pa) in pinks:
            d = (px-gx)**2 + (py-gy)**2
            if d < bestd:
                bestd = d; best = (px + ox, py + oy)
        side = ""
        if best is not None:
            side = side_of(g_abs, best)
        rows.append({
            "row_index": gi,
            "main_center": g_abs,
            "neigh_center": best,
            "side": side
        })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Vista previa con etiquetas (N/S/E/O y opcionalmente nombres)
# ──────────────────────────────────────────────────────────────────────────────
def draw_preview(bgr: np.ndarray, rows: List[dict], owners_rows: Optional[List[dict]], labels: bool, names: bool) -> np.ndarray:
    vis = bgr.copy()
    for r in rows:
        mc = r.get("main_center")
        nc = r.get("neigh_center")
        sd = r.get("side") or ""
        if mc:
            cv2.circle(vis, mc, 10, (0,255,0), -1)   # verde principal
        if nc:
            cv2.circle(vis, nc, 8, (0,0,255), -1)    # vecino rosa
            cv2.line(vis, mc, nc, (255,255,255), 2)
            if labels:
                label = {"norte":"N","sur":"S","este":"E","oeste":"O"}.get(sd, "?")
                cv2.putText(vis, label, (nc[0]+10, nc[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                if names and owners_rows:
                    yref = int((mc[1] + nc[1]) / 2) if mc and nc else (nc[1] if nc else mc[1])
                    name = pick_owner_for_y(owners_rows, yref)
                    if name:
                        cv2.putText(vis, name, (nc[0]+26, nc[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "version": app.version, "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY}

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: bool = Query(True),   # ← por defecto PINTA N/S/E/O
    names: bool = Query(False),   # ← añade nombres si se pide
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        rows = detect_rows_and_sides(bgr)
        owners_rows = owners_by_column_page2(pdf_bytes) if names else None
        vis = draw_preview(bgr, rows, owners_rows, labels=labels, names=names)
        ok, png = cv2.imencode(".png", vis)
        if not ok:
            raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")
    except Exception as e:
        err = str(e)
        blank = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:60]}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...), labels: bool = Query(True), names: bool = Query(False)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # (1) Titulares por columna en pág. 2
    owners_rows = owners_by_column_page2(pdf_bytes)
    owners_detected = [o["name"] for o in owners_rows if o.get("name")]
    owners_detected = list(dict.fromkeys(owners_detected))[:8]

    if TEXT_ONLY:
        note = "Modo TEXT_ONLY activo: visión desactivada."
        dbg = {"owners_rows": owners_rows[:6]} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","este":"","oeste":""},
                          owners_detected=owners_detected, note=note, debug=dbg)

    # (2) Visión en pág. 2 → lados
    try:
        bgr = page2_bgr(pdf_bytes)
        rows = detect_rows_and_sides(bgr)

        linderos = {"norte":"","sur":"","este":"","oeste":""}
        for r in rows:
            mc = r.get("main_center"); nc = r.get("neigh_center")
            sd = r.get("side") or ""
            if not sd:
                continue
            yref = None
            if mc and nc: yref = int((mc[1] + nc[1]) / 2)
            elif nc:      yref = nc[1]
            elif mc:      yref = mc[1]
            if yref is None:
                continue
            owner = pick_owner_for_y(owners_rows, yref)
            if owner and not linderos[sd]:
                linderos[sd] = owner

        note = None if any(linderos.values()) else "No se pudo determinar lado/vecino con suficiente confianza."
        dbg  = {"rows": rows, "owners_rows": owners_rows[:6]} if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        note = f"Excepción visión: {e}"
        dbg = {"exception": str(e)} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","este":"","oeste":""},
                          owners_detected=owners_detected, note=note, debug=dbg)




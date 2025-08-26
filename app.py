from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math, time
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.0")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI ajustables (por si quieres afinar sin redeploy)
FAST_DPI = int(os.getenv("FAST_DPI", "360"))     # ← bajado de 400 a 360
SLOW_DPI = int(os.getenv("SLOW_DPI", "500"))

# Forzar unir segunda línea del nombre (solo si lo necesitas)
PERMISSIVE_SECOND_LINE = (os.getenv("PERMISSIVE_SECOND_LINE", "0").strip() == "1")

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
# Utilidades comunes
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL"
}

# geo/dir abreviaturas típicas que NO son parte del nombre
ADDR_STOP = {
    "CL","CALLE","AV","AVDA","AVENIDA","CM","CARRETERA","LG","LUGAR","ES","ESC",
    "BLOQUE","PL","PT","PISO","PORTAL","KM","URB","POL","POLIGONO","C/","PZA","PLZ",
    "HUSO","ETRS","ESCALA","CSV","DIR","DOC"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

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

def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Raster (pág. 2) y masks
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> Tuple[np.ndarray, dict]:
    t0 = time.time()
    dpi = FAST_DPI if FAST_MODE else SLOW_DPI
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    arr = np.array(pil)[:, :, ::-1]  # RGB→BGR
    t1 = time.time()
    return arr, {"dpi": dpi, "raster_ms": int((t1 - t0) * 1000)}

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int,int,int]] = []
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
    ang = math.degrees(math.atan2(-(sy), sx))  # Norte arriba
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

# ──────────────────────────────────────────────────────────────────────────────
# OCR utils
# ──────────────────────────────────────────────────────────────────────────────
def enhance_gray(g: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

def binarize(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
    flags_inv = THRESH_BINARY_INV | (THRESH_OTSU if THRESH_OTSU else 0)
    _, bw  = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_bin)
    _, bwi = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_inv)
    return bw, bwi

def ocr_text(img: np.ndarray, psm: int, whitelist: Optional[str] = None) -> str:
    cfg = f"--psm {psm} --oem 3"
    if whitelist is not None:
        safe = (whitelist or "").replace('"','')
        cfg += f' -c tessedit_char_whitelist="{safe}"'
    txt = pytesseract.image_to_string(img, config=cfg) or ""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def clean_owner_tokens(line: str) -> str:
    """Limpia una línea de OCR asumiendo que es 'APELLIDOS NOMBRE / RAZÓN SOCIAL'."""
    if not line: return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in ADDR_STOP or t in GEO_TOKENS: break
        if t in BAD_TOKENS: continue
        out.append(t)
        # nombre típico: 2–4 palabras útiles
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 4:
            break
    # compactar conectores al inicio
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    return " ".join(compact).strip()[:48]

# ──────────────────────────────────────────────────────────────────────────────
# Extracción por filas (línea 1 + posible línea 2)
# ──────────────────────────────────────────────────────────────────────────────
def owner_rois_for_row(bgr: np.ndarray, row_y: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Calcula ROI de línea 1 y línea 2 para el titular en la columna
    'Apellidos Nombre / Razón social', con recorte aproximado por NIF si se detecta.
    """
    h, w = bgr.shape[:2]
    # Banda vertical alrededor de la fila
    pad_y = int(h * 0.10)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    # Columna del bloque de texto (izquierda de NIF)
    x_text0 = int(w * 0.26)     # un poco más a la izquierda (360 dpi → previene “mordiscos”)
    x_text1 = int(w * 0.95)

    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        # fallback conservador
        line_h = int(h * 0.036)
        y1_top = max(0, row_y - int(h*0.01))
        roi1 = (x_text0, y1_top, int(x_text0 + 0.55*(x_text1-x_text0)), y1_top + line_h)
        roi2 = (roi1[0], roi1[3] + int(h*0.010), roi1[2], roi1[3] + int(h*0.046))
        return roi1, roi2

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    header_bottom = None
    x_nif_local = None

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs    = data.get("left", [])
        ys    = data.get("top", [])
        ws    = data.get("width", [])
        hs    = data.get("height", [])

        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif_local = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            break

    # si no encontramos cabecera, estimamos alturas
    line_h = int(h * 0.036)
    gap_h  = int(h * 0.008)
    if header_bottom is None:
        y1_top_abs = max(0, row_y - int(h*0.01))
    else:
        y1_top_abs = y0s + header_bottom + 6

    # Cortes horizontales
    if x_nif_local is not None:
        x0 = x_text0
        x1 = min(x_text1, x_text0 + x_nif_local - 6)
    else:
        x0 = x_text0
        x1 = int(x_text0 + 0.55 * (x_text1 - x_text0))

    # ROI línea 1 y 2 (coordenadas absolutas)
    roi1 = (x0, y1_top_abs, x1, min(h, y1_top_abs + line_h))
    roi2 = (x0, roi1[3] + gap_h, x1, min(h, roi1[3] + gap_h + line_h))
    return roi1, roi2

def read_line_ocr(bgr: np.ndarray, roi: Tuple[int,int,int,int]) -> str:
    x0, y0, x1, y1 = roi
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
    g = enhance_gray(g)
    bw, bwi = binarize(g)
    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    variants = [
        ocr_text(bw,  psm=6,  whitelist=WL),
        ocr_text(bwi, psm=6,  whitelist=WL),
        ocr_text(bw,  psm=7,  whitelist=WL),
        ocr_text(bwi, psm=7,  whitelist=WL),
        ocr_text(bw,  psm=13, whitelist=WL),
    ]
    for t in variants:
        if t:
            return t
    return ""

def merge_two_lines(t1_raw: str, t2_raw: str) -> Tuple[str, dict]:
    """
    Une línea 1 y posible línea 2 con reglas conservadoras.
    Acepta 2ª línea si: una sola palabra tipo 'LUIS' o ≤2 tokens alfabéticos sin stop-words ni dígitos.
    Si PERMISSIVE_SECOND_LINE=1, la fuerza si existe, recortando a 26 chars.
    """
    info = {"picked_from": "strict", "second_line_used": False, "second_line_reason": ""}

    # Normalizar rupturas de línea (a veces Tesseract mete salto de línea en t1)
    if "\n" in (t1_raw or ""):
        parts = [p.strip() for p in t1_raw.split("\n") if p.strip()]
        if len(parts) >= 2:
            base = clean_owner_tokens(parts[0])
            extra = clean_owner_tokens(parts[1])
            name = (base + " " + extra).strip()
            if name:
                info.update({"second_line_used": True, "second_line_reason": "from_l1_break"})
                return name[:48], info
        else:
            t1_raw = parts[0]

    base = clean_owner_tokens(t1_raw)
    extra_raw = (t2_raw or "").strip()

    if not extra_raw:
        return base, info

    # Heurística estricta (no forzada)
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", extra_raw.upper()) if t]
    toks = [t for t in toks if not any(ch.isdigit() for ch in t)]
    toks = [t for t in toks if t not in ADDR_STOP and t not in GEO_TOKENS]
    toks = [t for t in toks if UPPER_NAME_RE.match(t)]

    if len(toks) == 1 and 3 <= len(toks[0]) <= 12:
        name = (base + " " + toks[0]).strip()
        info.update({"second_line_used": True, "second_line_reason": "1tok_ok"})
        return name[:48], info

    if len(toks) == 2 and all(2 <= len(t) <= 12 for t in toks):
        cand = " ".join(toks)
        if len(cand) <= 26:
            name = (base + " " + cand).strip()
            info.update({"second_line_used": True, "second_line_reason": "2tok_ok"})
            return name[:48], info

    # Forzado (opcional por env)
    if PERMISSIVE_SECOND_LINE:
        forced = re.sub(r"[\[\]\:\d]", " ", extra_raw.upper())
        forced = re.sub(r"\s{2,}", " ", forced).strip()
        name = (base + " " + forced).strip()
        info.update({"picked_from": "permissive", "second_line_used": True, "second_line_reason": "forced"})
        return name[:48], info

    return base, info

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte del bloque de croquis (izq) para detectar centroides
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(340 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"rows": [], "timings_ms": {}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # de arriba a abajo
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano para decidir lado
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        # ROIs línea 1 / línea 2
        roi1, roi2 = owner_rois_for_row(bgr, row_y=mcy)

        # OCR línea 1 y 2
        t1_raw = read_line_ocr(bgr, roi1)
        t2_raw = read_line_ocr(bgr, roi2)

        # Unir con reglas (posible 2ª línea)
        owner, merge_info = merge_two_lines(t1_raw, t2_raw)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            # puntos y etiqueta N/S/E/O
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {"norte":"N","sur":"S","este":"E","oeste":"O"}.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if annotate_names and owner:
                cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

            # dibuja las cajas usadas para L1/L2
            x0,y0,x1,y1 = roi1; cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)
            x0,y0,x1,y1 = roi2; cv2.rectangle(vis, (x0,y0), (x1,y1), (0,200,200), 2)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": {
                "band": [int(w*0.26), max(0, mcy-int(h*0.10)), int(w*0.95), min(h, mcy+int(h*0.10))],
                "y_line1": [roi1[1], roi1[3]],
                "y_line2_hint": [roi2[1], roi2[3]],
                "x0": roi1[0], "x1": roi1[2],
                "t1_raw": t1_raw, "t2_raw": t2_raw,
                "picked_from": merge_info.get("picked_from"),
                "second_line_used": merge_info.get("second_line_used"),
                "second_line_reason": merge_info.get("second_line_reason"),
            }
        })

    t1 = time.time()
    dbg = {"rows": rows_dbg, "timings_ms": {"rows_pipeline": int((t1 - t0) * 1000)}}
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "RASTER_DPI": FAST_DPI if FAST_MODE else SLOW_DPI,
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/S/E/O"),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr, rdbg = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:60]}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...),
                 labels: int = Query(0),
                 names: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr, rdbg = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = {**vdbg, "raster": rdbg} if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )



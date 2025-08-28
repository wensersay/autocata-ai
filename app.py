from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.9")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI: en FAST_MODE manda FAST_DPI; si no, PDF_DPI. Defaults seguros.
FAST_DPI = int(os.getenv("FAST_DPI", "340").strip() or "340")
PDF_DPI  = int(os.getenv("PDF_DPI",  "420").strip() or "420")

# Basura típica vista en 2ª línea (configurable)
JUNK_2NDLINE = {t.strip().upper() for t in (os.getenv("JUNK_2NDLINE","Z,VA,EO,SS,KO,KR").split(","))}
# Hints opcionales (no obligatorios ya): no son necesarios con el parche, pero se mantienen
NAME_HINTS_EXTRA = {t.strip().upper() for t in (os.getenv("NAME_HINTS_EXTRA","").split(",")) if t.strip()}

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

ADDR_TOKENS = {
    "CL","CALLE","CALLEJON","AV","AVENIDA","LG","LUGAR","CTRA","CARRETERA","URB","URBANIZACION",
    "BLOQUE","ESC","ESCALERA","PL","PLANTA","PT","PUERTA","CP","CODIGO","AP","APARTADO","KM",
    "Nº","NO","NUM","NÚM"
}
ADDR_TOKENS |= {"L'","L´","L’"}  # casos catalán

STOP_CHARS = set("[]:0123456789")
CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

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
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = FAST_DPI if FAST_MODE else PDF_DPI
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

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

# ──────────────────────────────────────────────────────────────────────────────
# Geometría: 8 rumbos (N, NE, E, SE, S, SW, W, NW)
# ──────────────────────────────────────────────────────────────────────────────
def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0° = Este; 90° = Norte
    # Bins de 45° centrados en los puntos cardinales
    if -22.5 <= ang < 22.5:   return "este"
    if 22.5 <= ang < 67.5:    return "noreste"
    if 67.5 <= ang < 112.5:   return "norte"
    if 112.5 <= ang < 157.5:  return "noroeste"
    if -67.5 <= ang < -22.5:  return "sureste"
    if -112.5 <= ang < -67.5: return "sur"
    if -157.5 <= ang < -112.5:return "suroeste"
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

def is_alpha_like(s: str) -> bool:
    s = s.strip().upper()
    if not s: return False
    if any(ch in STOP_CHARS for ch in s): return False
    if any(ch.isdigit() for ch in s): return False
    # Permitimos letras, espacios, apóstrofes y guiones
    return bool(re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ\s'´’-]+", s))

def looks_like_address_start(s: str) -> bool:
    if not s: return False
    s = s.strip().upper()
    # Si la primera palabra es token de dirección → lo consideramos dirección
    first = re.split(r"\s+", s)[0]
    return first in ADDR_TOKENS

def clean_name_tokens(s: str) -> str:
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", s.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        out.append(t)
        # Evitar novelas: quedarnos en ~6 tokens "reales"
        if len([x for x in out if x not in CONNECTORS]) >= 6:
            break
    # Compactar conectores al principio
    compact = []
    for t in out:
        if (not compact) and t in CONNECTORS: continue
        compact.append(t)
    return " ".join(compact).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Búsqueda por filas + OCR columna “Apellidos…”
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # 1) Croquis a la izquierda (márgenes conservadores)
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 180))
    if not mains:
        empty8 = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
        return empty8, {"rows": [], "raster":{"dpi": (FAST_DPI if FAST_MODE else PDF_DPI)}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    # 2) OCR en columna derecha (banda de "Apellidos Nombre / Razón social" → NIF)
    #    (estos percentiles han funcionado bien en tus PDFs)
    x0_col = int(w * 0.26)
    x1_col = int(w * 0.52)

    # debug
    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:8]:
        # Vecino más cercano → lado (8 rumbos)
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of8((mcx, mcy), best)

        # Banda vertical para la fila (dos renglones aprox)
        band_h = int(h * 0.11)
        y0 = max(0, mcy - band_h//2)
        y1 = min(h, mcy + band_h//2)
        band = bgr[y0:y1, x0_col:x1_col]
        owner, ocr_dbg = ocr_owner_from_band(band, abs_x0=x0_col, abs_y0=y0)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SW","oeste":"W","noroeste":"NW"
                }.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.44), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.44), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi": (FAST_DPI if FAST_MODE else PDF_DPI)}}
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# OCR puntual de la banda y REGLA NUEVA para unir 2ª línea "tipo nombre"
# ──────────────────────────────────────────────────────────────────────────────
def ocr_owner_from_band(band_bgr: np.ndarray, abs_x0: int, abs_y0: int) -> Tuple[str, dict]:
    dbg = {}
    if band_bgr.size == 0:
        return "", {"band":"empty"}

    # dividimos banda en L1 y L2 (mitades) y además leemos todo junto para capturar saltos de línea de Tesseract
    H, W = band_bgr.shape[:2]
    y_mid = H//2

    # OCR de la banda completa (captura posibles "\n" en la misma línea)
    g_full = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    g_full = cv2.resize(g_full, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    g_full = enhance_gray(g_full)
    bw_full, bwi_full = binarize(g_full)
    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    full_variants = [
        ocr_text(bw_full,  psm=6, whitelist=WL),
        ocr_text(bwi_full, psm=6, whitelist=WL),
    ]
    full_pick = ""
    for t in full_variants:
        if t:
            full_pick = t
            break

    # Extraemos línea1 + posible resto dentro de L1
    # y también una línea2 “geométrica” por si el OCR partió correctamente
    L1 = band_bgr[0:y_mid, :]
    L2 = band_bgr[y_mid:H, :]

    def ocr_line(img_bgr) -> str:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        for im in (bw, bwi):
            s = ocr_text(im, psm=6, whitelist=WL)
            if s: return s
        return ""

    l1_raw = ocr_line(L1)
    l2_raw = ocr_line(L2)

    # Descomponer L1 si contiene salto: “NOMBRE\nCONTINUACIÓN”
    l1_main = l1_raw
    l1_extra = ""
    if "\n" in l1_raw:
        parts = [p.strip() for p in l1_raw.split("\n") if p.strip()]
        if len(parts) >= 1:
            l1_main = parts[0]
            l1_extra = " ".join(parts[1:])

    # Si full_pick trae más líneas, aprovecha el primer bloque como L1 y el resto como extra si coincide
    if full_pick and not l1_extra and "\n" in full_pick:
        chunks = [c.strip() for c in full_pick.split("\n") if c.strip()]
        if chunks:
            l1_main = chunks[0]
            if len(chunks) >= 2:
                l1_extra = " ".join(chunks[1:])

    # Limpiar línea 1 (nombre base)
    base = clean_name_tokens(l1_main)

    # ─── PARCHE QUIRÚRGICO: UNIR SIEMPRE 2ª LÍNEA "TIPO NOMBRE" ─────────────
    # Candidate 1: l1_extra (misma mitad con salto)
    # Candidate 2: l2_raw (mitad inferior)
    cands = []
    if l1_extra: cands.append(l1_extra)
    if l2_raw:   cands.append(l2_raw)

    picked_from = "strict"
    extra_added = ""

    for c in cands:
        cand = c.strip()
        # limpiar separadores raros
        cand = re.sub(r"[^\wÁÉÍÓÚÜÑ\s'´’-]+", " ", cand).strip()
        # cortar en primer char prohibido
        for ch in "[]:": 
            if ch in cand:
                cand = cand.split(ch,1)[0].strip()
        # reglas de aceptación muy permisivas para nombres:
        if not cand: 
            continue
        candU = cand.upper()
        if candU in JUNK_2NDLINE:
            continue
        if looks_like_address_start(candU):
            # no unimos calles/AV/LG, etc.
            continue
        if not is_alpha_like(candU):
            continue

        # Si parece claramente nombre o fragmento, lo unimos
        extra_added = clean_name_tokens(candU)
        picked_from = "from_l1_break" if c is l1_extra else "from_l2"
        break

    owner = base
    if extra_added:
        # unir con límite suave para no crear supernombres (no truncamos apellidos largos habituales)
        join = (owner + " " + extra_added).strip()
        owner = re.sub(r"\s{2,}", " ", join)

    dbg.update({
        "band":[abs_x0, abs_y0, abs_x0 + W, abs_y0 + H],
        "y_line1":[abs_y0 + 0, abs_y0 + y_mid],
        "y_line2_hint":[abs_y0 + y_mid, abs_y0 + H],
        "x0":abs_x0, "x1":abs_x0 + W,
        "t1_raw": l1_raw,
        "t1_extra_raw": l1_extra,
        "t2_raw": l2_raw,
        "picked_from": picked_from
    })

    return owner, dbg

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
        "FAST_DPI": FAST_DPI,
        "PDF_DPI": PDF_DPI
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar rumbos (N/NE/E/SE/S/SW/W/NW)"),
    names: int = Query(0, description="1=mostrar nombre estimado junto al croquis")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
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
        empty8 = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
        return ExtractOut(
            linderos=empty8,
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        empty8 = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
        return ExtractOut(
            linderos=empty8,
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )




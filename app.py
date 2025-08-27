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
app = FastAPI(title="AutoCatastro AI", version="0.5.8")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI configurable por entorno. Si no se define, 340 en FAST y 500 fuera.
PDF_DPI_ENV = os.getenv("PDF_DPI")
try:
    PDF_DPI = int(PDF_DPI_ENV) if PDF_DPI_ENV else (340 if FAST_MODE else 500)
except Exception:
    PDF_DPI = 340 if FAST_MODE else 500

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

# Geografía / dirección (para descartar segundas líneas que no sean nombres)
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","[MADRID]"
}
ADDR_TOKENS = {
    "CL","C/","CALLE","AV","AV.","AVDA","LG","PL","PZO","PZ","PZA","PORTAL",
    "ESC","ESC.","ESCALERA","BLOQUE","BLOQ","PT","PT.","PL:","ES:","KM","Nº",
    "NUM","PI","PISO","CARRETERA","TRAVESIA","TRAVESÍA","CAMINO"
}
# Hints de nombres (para aceptar segundas líneas de una palabra como LUIS, MARÍA…)
NAME_HINTS_BASE = {
    "JOSE","JOSÉ","LUIS","MARIA","MARÍA","JESUS","JESÚS","ANTONIO","FRANCISCO",
    "CARLOS","ANA","ELENA","RAFAEL","MANUEL","MIGUEL","PABLO","DAVID","SERGIO",
    "LAURA","PAULA","ALEJANDRO","IRENE","MARTA","ROSA","ALVARO","ÁLVARO"
}
EXTRA = os.getenv("NAME_HINTS_EXTRA","").strip()
if EXTRA:
    for t in re.split(r"[,\s;]+", EXTRA):
        if t: NAME_HINTS_BASE.add(t.strip().upper())

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
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = PDF_DPI
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

# Octantes: N, NE, E, SE, S, SO, O, NO
def side_of_octant(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    dx, dy = x - cx, cy - y  # y invertido (imagen)
    ang = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    # 0° = E, 90° = N, 180° = O, 270° = S
    bins = [
        ("este",   (337.5, 360.0), (0.0, 22.5)),
        ("noreste",(22.5, 67.5)),
        ("norte",  (67.5, 112.5)),
        ("noroeste",(112.5,157.5)),
        ("oeste",  (157.5,202.5)),
        ("suroeste",(202.5,247.5)),
        ("sur",    (247.5,292.5)),
        ("sureste",(292.5,337.5)),
    ]
    for name, *ranges in bins:
        for lo, hi in ranges:
            if lo <= ang <= hi: return name
    return ""  # fallback

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

def clean_owner_line(line: str) -> str:
    if not line: return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in GEO_TOKENS or "[" in t or "]" in t: break
        if t in BAD_TOKENS: continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 4:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:48]

def pick_owner_from_text(txt: str) -> str:
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):
            continue
        if sum(ch.isdigit() for ch in U) > 1:
            continue
        if not UPPER_NAME_RE.match(U):
            continue
        name = clean_owner_line(U)
        if len(name) >= 8:
            return name
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Segunda línea: filtro fino anti-ruido (acepta “LUIS”, etc. / rechaza “P”, “SS”…)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_second_line(s: str) -> str:
    s = (s or "").strip().upper()
    # recortes duros
    s = re.split(r"[\[\]\d:|]+", s)[0].strip()
    s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ' \-]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s: return ""

    # tokens
    toks = [t for t in s.split(" ") if t]
    # descartes evidentes: dirección / geografía
    if any(t in ADDR_TOKENS or t in GEO_TOKENS for t in toks):
        return ""

    # si es un solo token
    if len(toks) == 1:
        t = toks[0]
        if len(t) <= 2 and t not in {"DE","DA","DO"}:
            return ""  # “P”, “SS”, etc.
        # exigir al menos una vocal para evitar “KR”, “KO”, “SS”
        if not re.search(r"[AEIOUÁÉÍÓÚÜ]", t):
            return ""
        # si no está en hints y es muy corto, rechazar
        if len(t) <= 3 and t not in NAME_HINTS_BASE:
            return ""
        return t[:26]

    # dos o tres tokens → deben ser todos alfabéticos y con alguna vocal global
    if len(toks) <= 3:
        if not re.search(r"[AEIOUÁÉÍÓÚÜ]", s):
            return ""
        return " ".join(toks)[:26]

    # más de tres suele ser ruido / domicilio
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columnas y extraer línea 1 + posible línea 2
# ──────────────────────────────────────────────────────────────────────────────
def find_header_band_and_x(bgr: np.ndarray, row_y: int) -> Tuple[int,int,int,int,int,int]:
    """
    Devuelve (band_x0, band_y0, band_x1, band_y1, line1_y0, line1_y1)
    y además calcula x0..x1 del bloque 'Apellidos ...' hasta 'NIF' si aparece.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    x_text0 = int(w * 0.25)     # algo más a la izqda. para asegurar
    x_text1 = int(w * 0.58)     # hasta antes de NIF

    band = bgr[y0s:y1s, x_text0:int(w*0.95)]
    if band.size == 0:
        # fallback conservador
        bh0, bh1 = row_y - int(h*0.02), row_y + int(h*0.02)
        return x_text0, max(0,bh0), int(w*0.58), min(h,bh1), max(0,bh0), min(h,bh1)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    header_left_abs, x_nif_abs = None, None
    header_bottom = None

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])

        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_left_abs = x_text0 + lx
                header_bottom = (y0s + ty + hh) if header_bottom is None else max(header_bottom, y0s + ty + hh)
            if T == "NIF":
                x_nif_abs = x_text0 + lx
                header_bottom = (y0s + ty + hh) if header_bottom is None else max(header_bottom, y0s + ty + hh)

    if header_bottom is None:
        # no encontramos cabecera cerca → delimitamos por proporciones
        band_y0 = max(0, row_y - int(h*0.04))
        band_y1 = min(h, row_y + int(h*0.10))
        line1_y0 = band_y0
        line1_y1 = min(h, line1_y0 + int(h*0.04))
        return x_text0, band_y0, x_text1, band_y1, line1_y0, line1_y1

    # banda y líneas
    line1_y0 = header_bottom + 6
    line1_y1 = min(h, line1_y0 + int(h*0.035))
    band_y0  = line1_y0
    band_y1  = min(h, band_y0 + int(h*0.12))

    x1 = x_nif_abs - 6 if x_nif_abs else x_text1
    return x_text0, band_y0, x1, band_y1, line1_y0, line1_y1

def ocr_owner_two_lines(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Devuelve (owner, debug_dict). Hace OCR de línea 1 + intenta línea 2 filtrada.
    """
    h, w = bgr.shape[:2]
    x0, by0, x1, by1, l1y0, l1y1 = find_header_band_and_x(bgr, row_y)

    # ROI de línea 1
    roi1 = bgr[l1y0:l1y1, x0:x1]
    # ROI de línea 2 sugerida (mismo ancho, siguiente franja)
    l2y0 = min(h, l1y1 + 2)
    l2y1 = min(h, l2y0 + (l1y1 - l1y0))
    roi2 = bgr[l2y0:l2y1, x0:x1]

    def do_ocr(roi: np.ndarray) -> str:
        if roi.size == 0: return ""
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
        for im in (bw, bwi):
            for p in (6,7,13):
                t = ocr_text(im, psm=p, whitelist=WL)
                if t: return t
        return ""

    t1_raw = do_ocr(roi1).replace("\n", " ").strip()
    t2_raw = do_ocr(roi2).replace("\n", " ").strip()

    # A veces Tesseract parte “RODRIGUEZ ALVAREZ JOSE\nLUIS” en la L1;
    # recuperamos el trozo de salto de línea si está pegado a la L1
    t1_extra_raw = ""
    if "\n" in ocr_text(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), psm=6, whitelist=None):
        # ya vienen sin nuevas líneas tras el replace de arriba; conservamos nada
        pass

    # Limpieza y elección
    t1 = clean_owner_line(t1_raw)
    picked_from = "strict"
    if len(t1) < 6:
        # fallback permisivo si salió muy corto
        t1 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ' \-]", " ", t1_raw.upper()).strip()
        picked_from = "fallback"

    # Segunda línea con filtro fino
    second = normalize_second_line(t2_raw)
    used_second = False
    second_reason = ""
    if not second and t1_extra_raw:
        # Caso “from_l1_break”: si L1 contenía salto y OCR lo vio en extra
        second = normalize_second_line(t1_extra_raw)
        if second:
            used_second = True
            second_reason = "from_l1_break"
    elif second:
        used_second = True
        second_reason = "second_filtered_ok"

    owner = t1
    if second and second not in owner:
        owner = (owner + " " + second).strip()

    dbg = {
        "band":[x0, by0, x1, by1],
        "y_line1":[l1y0, l1y1],
        "y_line2_hint":[l2y0, l2y1],
        "x0":x0, "x1":x1,
        "t1_raw":t1_raw, "t1_extra_raw":t1_extra_raw,
        "t2_raw":t2_raw,
        "picked_from":picked_from,
        "second_line_used":used_second,
        "second_line_reason":second_reason
    }
    return owner, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:

    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Zona donde viven croquis (izquierda) para hallar centroides
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}, {"rows": []}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of_octant((mcx, mcy), best)

        owner, ocr_dbg = ocr_owner_two_lines(bgr, row_y=mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {"norte":"N","noreste":"NE","este":"E","sureste":"SE",
                       "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"}.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if annotate_names and owner:
                cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi": PDF_DPI}}
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
        "PDF_DPI": PDF_DPI,
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
    names: int = Query(0, description="1=mostrar nombre estimado")
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
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
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
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )



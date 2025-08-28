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
app = FastAPI(title="AutoCatastro AI", version="0.6.2")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# DPI (FAST_MODE usa FAST_DPI; si no, PDF_DPI)
def get_raster_dpi() -> int:
    if FAST_MODE:
        return int(os.getenv("FAST_DPI", 340))
    return int(os.getenv("PDF_DPI", 420))

# Lista de ruidos para 2ª línea (configurable)
JUNK_2NDLINE = [s.strip().upper() for s in os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",") if s.strip()]

# Pistas de nombres (opcionales) desde entorno
NAME_HINTS_EXTRA = [s.strip().upper() for s in os.getenv("NAME_HINTS_EXTRA", "").split(",") if s.strip()]

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

class BatchIn(BaseModel):
    pdf_urls: List[AnyHttpUrl]

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades comunes
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","AP","AN","A","R","AR","NR","ANR","APN","APNOM",
    "APELLIDOS/NOMBRE","APELLIDOS/NOMBRE/RAZON","APELLIDOS NOMBRE/RAZON"
}

# geotokens / ruido que no deben entrar en el nombre
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","DE","DEL","DA","DO","DAS","DOS"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Pistas base + extra (para ayudar a aceptar segundos nombres sueltos)
COMMON_NAME_HINTS = set(NAME_HINTS_EXTRA + [
    "LUIS","MARIA","MARÍA","JOSE","JOSÉ","ANTONIO","MANUEL","FRANCISCO","ANA","CARLOS",
    "JAVIER","JUAN","PEDRO","ALVARO","ÁLVARO","MARTA","PABLO","RAFAEL","ALFONSO","ALFONSA",
])

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
    dpi = max(260, min(520, get_raster_dpi()))
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verdes (parcelas objeto)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # Magentas/rojos (vecinas)
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

# 8 direcciones
def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=E, 90=N
    # octantes (±22.5°)
    if -22.5 <= ang <= 22.5:   return "este"
    if 22.5 < ang <= 67.5:     return "noreste"
    if 67.5 < ang <= 112.5:    return "norte"
    if 112.5 < ang <= 157.5:   return "noroeste"
    if (ang > 157.5) or (ang <= -157.5): return "oeste"
    if -157.5 < ang <= -112.5: return "suroeste"
    if -112.5 < ang <= -67.5:  return "sur"
    return "sureste"

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
    U = line.upper()
    # cortar si aparecen números o corchetes/brackets
    U = re.split(r"[\[\]:0-9]", U)[0]
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", U) if t]
    out = []
    for t in toks:
        if t in GEO_TOKENS: break
        if t in BAD_TOKENS: continue
        out.append(t)
        # parar si ya tenemos 4–5 tokens útiles
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    # compactar conectores múltiples
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:48]

def line_has_many_oneletters(s: str) -> bool:
    toks = [t for t in re.split(r"\s+", s.strip()) if t]
    if not toks: return False
    ones = sum(1 for t in toks if len(t) == 1)
    return ones >= 2 and ones >= len(toks)/2

def is_junk_2ndline(s: str) -> bool:
    U = s.strip().upper()
    if not U: return True
    if any(ch.isdigit() for ch in U): return True
    if any(b in U for b in ["[", "]", ":", ";", "/", "\\"]): return True
    if U in JUNK_2NDLINE: return True
    if line_has_many_oneletters(U): return True
    return False

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
        if len(name) >= 6:
            return name
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna y extraer línea 1 + 2
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_guides(bgr: np.ndarray, row_y: int, x0_hint: int, x1_hint: int) -> Tuple[int,int,int,int,int,int]:
    """
    Devuelve (x0, x1, y1_top, y1_bot, y2_top, y2_bot)
    x0..x1: límites laterales seguros (desde cabecera 'APELLIDOS...' a 'NIF')
    y1_top..y1_bot: franja de la primera línea
    y2_top..y2_bot: franja de la segunda línea
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x0_hint:x1_hint]
    if band.size == 0:
        # fallback conservador
        y1_top = max(0, row_y - int(h*0.02))
        y1_bot = min(h, y1_top + int(h*0.03))
        y2_top = y1_bot + 2
        y2_bot = min(h, y2_top + int(h*0.03))
        return x0_hint, int(x0_hint + 0.55*(x1_hint-x0_hint)), y1_top, y1_bot, y2_top, y2_bot

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    header_left = None
    x_nif = None
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
            if "APELLIDOS" in T or "APELLIDOS/NOMBRE" in T or "APELLIDOS NOMBRE" in T:
                header_left = lx
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            break

    if header_bottom is None:
        # fallback
        y1_top = max(0, row_y - int(h*0.02))
        y1_bot = min(h, y1_top + int(h*0.03))
        y2_top = y1_bot + 2
        y2_bot = min(h, y2_top + int(h*0.03))
        return x0_hint, int(x0_hint + 0.55*(x1_hint-x0_hint)), y1_top, y1_bot, y2_top, y2_bot

    y1_top = y0s + header_bottom + 6
    y1_bot = min(h, y1_top + int(h * 0.028))
    y2_top = y1_bot + 6
    y2_bot = min(h, y2_top + int(h * 0.028))

    x0 = x0_hint if header_left is None else x0_hint + header_left + 4
    if x_nif is not None:
        x1 = min(x1_hint, x0_hint + x_nif - 8)
    else:
        x1 = int(x0_hint + 0.55*(x1_hint-x0_hint))
    if x1 - x0 < (x1_hint - x0_hint) * 0.22:
        # asegurar ancho mínimo
        x0 = x0_hint
        x1 = int(x0_hint + 0.55*(x1_hint-x0_hint))

    return x0, x1, y1_top, y1_bot, y2_top, y2_bot

def ocr_band_first_second_lines(bgr: np.ndarray, row_y: int) -> Tuple[str, str, Dict]:
    """
    Devuelve (t1_clean, second_pick, dbg)
    - Lee la banda del nombre para L1 y L2
    - Si L1 contiene salto de línea, usa la parte extra como segunda línea (from_l1_break)
    - Si no, usa L2 si no es basura y <= 26 chars
    """
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.33)
    x_text1 = int(w * 0.96)

    x0, x1, y1t, y1b, y2t, y2b = find_header_and_guides(bgr, row_y, x_text0, x_text1)

    # recorte L1
    roi1 = bgr[y1t:y1b, x0:x1]
    roi2 = bgr[y2t:y2b, x0:x1]
    t1_clean, t2_clean = "", ""
    picked_from = "strict"
    dbg = {"band":[x0,y1t,x1,y2b], "y_line1":[y1t,y1b], "y_line2_hint":[y2t,y2b], "x0":x0, "x1":x1}

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    if roi1.size:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g1 = enhance_gray(g1)
        bw1, bwi1 = binarize(g1)
        v1 = [
            ocr_text(bw1, 6, WL),
            ocr_text(bwi1, 6, WL),
            ocr_text(bw1, 7, WL),
            ocr_text(bwi1, 7, WL),
        ]
        raw1 = ""
        for t in v1:
            if t:
                raw1 = t
                break
        # ¿L1 trae salto de línea (p.ej. "RODRIGUEZ ...\nLUIS")?
        if "\n" in raw1:
            parts = [p.strip() for p in raw1.split("\n") if p.strip()]
            if parts:
                t1_clean = clean_owner_line(parts[0])
                extra = " ".join(parts[1:]).strip()
                extra = re.sub(r"\s+", " ", extra)
                # limitar extra a 26 y limpiar mínima basura
                extra_u = extra.upper()
                if not is_junk_2ndline(extra_u):
                    t2_clean = extra_u[:26]
                    picked_from = "from_l1_break"
        else:
            t1_clean = clean_owner_line(raw1)

    if (not t2_clean) and roi2.size:
        # Intentar 2ª línea (sólo si no vino de L1)
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g2 = enhance_gray(g2)
        bw2, bwi2 = binarize(g2)
        v2 = [
            ocr_text(bw2, 6, WL),
            ocr_text(bwi2, 6, WL),
            ocr_text(bw2, 7, WL),
            ocr_text(bwi2, 7, WL),
        ]
        raw2 = ""
        for t in v2:
            if t:
                raw2 = t
                break
        raw2 = raw2.strip().upper()
        # limpieza simple y corte
        raw2 = re.split(r"[\[\]:/\\0-9]", raw2)[0]
        raw2 = re.sub(r"\s+", " ", raw2).strip()
        if raw2:
            if (raw2 in COMMON_NAME_HINTS) or not is_junk_2ndline(raw2):
                t2_clean = raw2[:26]

    dbg.update({"t1_raw": t1_clean, "t2_raw": t2_clean, "picked_from": picked_from})
    return t1_clean, t2_clean, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Zona de minicroquis (izquierda)
    top = int(h * 0.10); bottom = int(h * 0.90)
    left = int(w * 0.05); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    if not mains:
        return ({"norte":"","noreste":"","este":"","sureste":"",
                 "sur":"","suroeste":"","oeste":"","noroeste":""},
                {"rows":[], "raster":{"dpi":get_raster_dpi()}}, vis)

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"",
                "sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of8((mcx, mcy), best)

        # OCR columna a la derecha (L1 + L2)
        t1, t2, dbg_ocr = ocr_band_first_second_lines(bgr, row_y=mcy)
        owner = t1
        if t2:
            owner = (t1 + " " + t2).strip()

        if side and owner and side not in used_sides:
            # no sobrescribir si ya hay contenido en ese lado (primer valor gana)
            if not linderos.get(side):
                linderos[side] = owner
                used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (255,0,255), -1)
                lbl = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SW","oeste":"O","noroeste":"NW"
                }.get(side,"")
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
            "ocr": dbg_ocr
        })

    t1 = time.time()
    dbg = {"rows": rows_dbg, "timings_ms": {"rows_pipeline": int((t1 - t0)*1000)}, "raster":{"dpi":get_raster_dpi()}}
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "msg": "AutoCatastro AI up", "version": app.version}

@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "dpi_in_use": get_raster_dpi()
    }

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "dpi_in_use": get_raster_dpi()
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SW/O/NW"),
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
            linderos={"norte":"","noreste":"","este":"","sureste":"",
                      "sur":"","suroeste":"","oeste":"","noroeste":""},
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
            linderos={"norte":"","noreste":"","este":"","sureste":"",
                      "sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )

# Batch: procesa hasta 10 PDFs
@app.post("/batch/extract", dependencies=[Depends(check_token)])
def batch_extract(data: BatchIn = Body(...), debug: bool = Query(False)):
    urls = list(data.pdf_urls)[:10]
    results = []
    for url in urls:
        try:
            out = extract(ExtractIn(pdf_url=url), debug=debug)
            # extract() ya devuelve ExtractOut; convertir a dict
            results.append({
                "pdf_url": str(url),
                "linderos": out.linderos,
                "owners_detected": out.owners_detected,
                "note": out.note,
                "debug": out.debug if debug else None
            })
        except Exception as e:
            results.append({
                "pdf_url": str(url),
                "linderos": {"norte":"","noreste":"","este":"","sureste":"",
                             "sur":"","suroeste":"","oeste":"","noroeste":""},
                "owners_detected": [],
                "note": f"error: {e}",
                "debug": {"exception": str(e)} if debug else None
            })
    return {"ok": True, "count": len(results), "items": results}




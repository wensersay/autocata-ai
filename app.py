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
app = FastAPI(title="AutoCatastro AI", version="0.5.7")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI (preferencia: FAST_DPI si FAST_MODE, si no PDF_DPI; fallback 340)
def _to_int(val: Optional[str], default: int) -> int:
    try:
        return int(str(val).strip())
    except Exception:
        return default

FAST_DPI = _to_int(os.getenv("FAST_DPI", ""), 340)
PDF_DPI  = _to_int(os.getenv("PDF_DPI",  ""), 400)

def pick_dpi() -> int:
    return FAST_DPI if FAST_MODE else PDF_DPI

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

# geotokens / ruido que no deben entrar en el nombre
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Pistas de nombres (internas + añadibles por ENV)
DEFAULT_NAME_HINTS = {
    "ANA","LUIS","JOSE","JOSÉ","MARIA","MARÍA","JUAN","CARMEN","LAURA",
    "ANTONIO","ANTÓN","MARTA","PABLO","JAVIER","RAQUEL","ALVARO","ÁLVARO",
    "JESUS","JESÚS","FRANCISCO","MANUEL","DOLORES","JOSE LUIS","JOSÉ LUIS",
    "MARIA JOSE","MARÍA JOSÉ"
}
EXTRA_HINTS = {t.strip().upper() for t in os.getenv("NAME_HINTS_EXTRA","").split(",") if t.strip()}
NAME_HINTS = DEFAULT_NAME_HINTS | EXTRA_HINTS

# Basura típica en 2ª línea (por OCR)
JUNK_2NDLINE = {t.strip().upper() for t in os.getenv("JUNK_2NDLINE","Z,VA,EO,SS,KO,KR").split(",") if t.strip()}

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
    dpi = pick_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    bgr = np.array(pil)[:, :, ::-1]
    return bgr

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

# 8 rumbos
def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0º este, 90º norte
    # sectores de 45º (bordes a ±22.5, 67.5, 112.5, 157.5)
    if -22.5 <= ang <= 22.5:    return "este"
    if 22.5 < ang <= 67.5:      return "noreste"
    if 67.5 < ang <= 112.5:     return "norte"
    if 112.5 < ang <= 157.5:    return "noroeste"
    if ang > 157.5 or ang < -157.5: return "oeste"
    if -157.5 <= ang < -112.5:  return "suroeste"
    if -112.5 <= ang < -67.5:   return "sur"
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

def pick_owner_from_l1(raw: str) -> Tuple[str, str]:
    """
    Devuelve (owner, extra_from_break) usando SOLO la banda de la 1ª línea.
    Si Tesseract metió un salto en esa misma banda (dos renglones), extra_from_break
    contendrá lo que haya tras el primer salto si parece un nombre.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", ""
    parts = [p.strip() for p in raw.split("\n") if p.strip()]
    if not parts:
        return "", ""
    first = parts[0]
    cand = clean_owner_line(first.upper())
    extra = ""
    if len(parts) >= 2:
        second_inline = parts[1]
        s2 = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", second_inline.upper()).strip()
        if s2 and len(s2) <= 26 and not any(ch.isdigit() for ch in s2):
            extra = s2
    return cand, extra

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna “Apellidos…” y extraer 1ª línea + posible 2ª línea
# ──────────────────────────────────────────────────────────────────────────────
def find_columns_once(bgr: np.ndarray) -> Tuple[int,int]:
    """
    Busca 'APELLIDOS' y 'NIF' una vez por página (zona media-izquierda).
    Devuelve (x_header_left, x_nif) aproximados. Si no encuentra, usa heurística.
    """
    h, w = bgr.shape[:2]
    x0 = int(w * 0.28); x1 = int(w * 0.90)
    y0 = int(h * 0.18); y1 = int(h * 0.88)
    roi = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    def scan(img) -> Tuple[Optional[int], Optional[int]]:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])
        x_header = None
        x_nif = None
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                x_header = x0 + lx
            if T == "NIF":
                x_nif = x0 + lx
        return x_header, x_nif

    xh, xn = scan(bw)
    if xh is None or xn is None:
        xh2, xn2 = scan(bwi)
        if xh is None: xh = xh2
        if xn is None: xn = xn2

    if xh is None: xh = int(w * 0.35)
    if xn is None: xn = int(w * 0.63)
    return xh, xn

def extract_owner_from_row(bgr: np.ndarray, row_y: int, cols_hint: Tuple[int,int]) -> Tuple[str, dict]:
    """
    Devuelve (owner, debug_dict) para una fila centrada en row_y.
    cols_hint = (x_header_left, x_nif) estimados.
    """
    h, w = bgr.shape[:2]
    x_header_left, x_nif = cols_hint

    # Banda vertical alrededor de la fila
    band_h = int(h * 0.09)  # ± ~4.5% por arriba/abajo
    y0s = max(0, row_y - band_h//2)
    y1s = min(h, row_y + band_h//2)

    # 1ª línea: inmediatamente debajo del encabezado
    line_h = int(h * 0.03)
    y1_top  = max(0, y0s + int(h*0.03))
    y1_bot  = min(h, y1_top + line_h)

    # 2ª línea: justo debajo de la 1ª
    y2_top  = min(h, y1_bot + 2)
    y2_bot  = min(h, y2_top + line_h)

    x0 = max(0, x_header_left)
    x1 = min(w, x_nif - 4) if x_nif > x_header_left else min(w, x_header_left + int(0.55*(w - x_header_left)))

    # Recortes
    roi1 = bgr[y1_top:y1_bot, x0:x1]
    roi2 = bgr[y2_top:y2_bot, x0:x1]

    t1_raw = ""
    t1_extra = ""
    t2_raw = ""

    if roi1.size > 0:
        g = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '\n-"
        cand1 = ocr_text(bw,  psm=6, whitelist=WL)
        cand2 = ocr_text(bwi, psm=6, whitelist=WL)
        best  = cand1 if len(cand1) >= len(cand2) else cand2
        t1_raw = best
        cand_name, extra_inline = pick_owner_from_l1(best)
    else:
        cand_name, extra_inline = "", ""

    # 2ª línea de banda (ROI2)
    if roi2.size > 0:
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(g2, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g2 = enhance_gray(g2)
        bw2, bwi2 = binarize(g2)
        WL2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
        c21 = ocr_text(bw2,  psm=7, whitelist=WL2)
        c22 = ocr_text(bwi2, psm=7, whitelist=WL2)
        t2_raw = c21 if len(c21) >= len(c22) else c22
        t2_raw = t2_raw.strip()

    # Parche: filtro robusto de 2ª línea
    def accept_second(s: str) -> str:
        s = (s or "").upper().strip()
        if not s:
            return ""
        # cortar en números, corchetes, dos puntos
        s = re.split(r"[\[\]:0-9]", s)[0].strip()
        if not s:
            return ""
        # limpiar caracteres raros
        s = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", s).strip()
        if not s:
            return ""

        # tokens basura por ENV (Z,VA,EO,SS,KO,KR, ...)
        if s in JUNK_2NDLINE:
            return ""

        # ¿todo la misma letra? (S, SS, SSS, AAA, …)
        if re.fullmatch(r"([A-ZÁÉÍÓÚÜÑ])\1{1,}$", s):
            return ""

        # tokenización
        toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", s) if t]
        if not toks:
            return ""

        # si es un único token muy corto (≤3) y no es un nombre conocido, descartar
        if len(toks) == 1 and len(toks[0]) <= 3 and toks[0] not in NAME_HINTS:
            return ""

        # limitar largo total
        if len(s) > 26:
            s = s[:26].rstrip()

        return " ".join(toks)

    owner = cand_name
    picked_from = "strict"

    # Si la 1ª línea traía un salto con un nombre (inline)
    if owner and extra_inline:
        s2 = accept_second(extra_inline)
        if s2:
            owner = f"{owner} {s2}".strip()
            picked_from = "from_l1_break"

    # Si no se añadió nada arriba, intenta con la 2ª banda
    if owner and owner.strip() and t2_raw:
        s2 = accept_second(t2_raw)
        if s2 and s2 not in owner:
            owner = f"{owner} {s2}".strip()
            picked_from = "with_l2"

    dbg = {
        "band": [x0, y0s, x1, y1s],
        "y_line1": [y1_top, y1_bot],
        "y_line2_hint": [y2_top, y2_bot],
        "x0": x0, "x1": x1,
        "t1_raw": t1_raw,
        "t1_extra_raw": extra_inline,
        "t2_raw": t2_raw,
        "picked_from": picked_from
    }
    return owner, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # ROI izquierda donde están los croquis
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    if not mains:
        # 8 rumbos vacíos
        empty8 = {"norte":"","noreste":"","este":"","sureste":"",
                  "sur":"","suroeste":"","oeste":"","noroeste":""}
        return empty8, {"rows": [], "raster":{"dpi": pick_dpi()}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    # localizar columnas una sola vez
    cols_hint = find_columns_once(bgr)
    x_header_left, x_nif = cols_hint

    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"",
                "sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of8((mcx, mcy), best)

        owner, ocr_dbg = extract_owner_from_row(bgr, row_y=mcy, cols_hint=cols_hint)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl_map = {"norte":"N","noreste":"NE","este":"E","sureste":"SE",
                           "sur":"S","suroeste":"SW","oeste":"W","noroeste":"NW"}
                lbl = lbl_map.get(side,"")
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
            "ocr": ocr_dbg,
            "header_left_abs": x_header_left,
            "x_nif_abs": x_nif
        })

    t1 = time.time()
    dbg = {
        "rows": rows_dbg,
        "timings_ms": {"rows_pipeline": int((t1-t0)*1000)},
        "raster": {"dpi": pick_dpi()}
    }
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
        "dpi": pick_dpi(),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar rumbos N/NE/E/..."),
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
        blank = np.zeros((260, 900, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:120]}", (10,140),
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
        empty8 = {"norte":"","noreste":"","este":"","sureste":"",
                  "sur":"","suroeste":"","oeste":"","noroeste":""}
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
        empty8 = {"norte":"","noreste":"","este":"","sureste":"",
                  "sur":"","suroeste":"","oeste":"","noroeste":""}
        return ExtractOut(
            linderos=empty8,
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )




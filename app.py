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
import pdfplumber

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.5.3")

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
# Utilidades comunes (texto / OCR)
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL"
}
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}
ADDR_TOKENS = {
    "CL","CALLE","AV","AVDA","AVENIDA","LG","LUGAR","BLOQUE","ESC","ES",
    "PL","PISO","PT","PORTAL","URB","URBANIZACION","URBANIZACIÓN","CTRA",
    "CARRETERA","KM"
}
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

COMMON_SECOND_NAMES = {
    "LUIS","MARIA","MARÍA","JAVIER","ANTONIO","MANUEL","MIGUEL","ANGEL","ÁNGEL",
    "FRANCISCO","CARLOS","ALEJANDRO","JOSE","JOSÉ","PABLO","ALBERTO","ENRIQUE",
    "DANIEL","RAFAEL","FERNANDO","ALFONSO","CARMEN","ISABEL","JESUS","JESÚS"
}

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

def _tokens_upper(s: str) -> List[str]:
    return [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", (s or "").upper()) if t]

def clean_owner_line(line: str) -> str:
    if not line: return ""
    toks = _tokens_upper(line)
    out = []
    non_conn = 0
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in GEO_TOKENS or "[" in t or "]" in t: break
        if t in BAD_TOKENS or t in ADDR_TOKENS: continue
        # descarta tokens demasiado cortos que no sean conectores
        if len(t) <= 2 and t not in NAME_CONNECTORS: 
            continue
        out.append(t)
        if t not in NAME_CONNECTORS:
            non_conn += 1
            # Limitamos a 3 “tokens de nombre” para evitar colas raras
            if non_conn >= 3:
                break
    # compactar conectores iniciales
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
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):            continue
        name = clean_owner_line(U)
        if len(name) >= 6:
            return name
    return ""

# ────────── heurística para 2ª línea (solo si parece “continuación de nombre”) ──────────
def is_probable_name_continuation(line: str) -> Tuple[bool, dict]:
    dbg = {"reason": "ok"}
    if not line: 
        dbg["reason"] = "empty"; 
        return False, dbg
    # cortar por primer dígito, [, :
    m = re.search(r"[\d\[:]", line)
    if m:
        line = line[:m.start()]
    toks = _tokens_upper(line)
    if not toks:
        dbg["reason"] = "no_tokens"
        return False, dbg
    # no geografía ni abreviaturas de dirección
    if any(t in GEO_TOKENS or t in ADDR_TOKENS for t in toks):
        dbg["reason"] = "geo_or_addr_token"
        return False, dbg
    # máximo 26 chars
    line = line.strip()[:26]
    toks = _tokens_upper(line)

    # Regla de tamaño/forma: 1–2 tokens (3 si lleva DE/DEL/Y)
    if len(toks) == 1:
        ok = toks[0] in COMMON_SECOND_NAMES or len(toks[0]) >= 3
        dbg["reason"] = "1tok_ok" if ok else "1tok_too_short_or_unknown"
        return ok, dbg
    if len(toks) == 2:
        # Ambos alfabéticos y no “ruido” muy corto
        ok = all(len(t) >= 3 or t in NAME_CONNECTORS for t in toks)
        dbg["reason"] = "2tok_ok" if ok else "2tok_has_short_noise"
        return ok, dbg
    if len(toks) == 3 and toks[1] in NAME_CONNECTORS:
        ok = (len(toks[0]) >= 3 and len(toks[2]) >= 3)
        dbg["reason"] = "3tok_connector_ok" if ok else "3tok_connector_short"
        return ok, dbg

    dbg["reason"] = "too_many_tokens"
    return False, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Raster (pág. 2) y máscaras
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 400 if FAST_MODE else 500
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

def crop_map(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    h, w = bgr.shape[:2]
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    top = max(0, top); bottom = min(h, bottom)
    left = max(0, left); right = min(w, right)
    if bottom - top < 100 or right - left < 100:
        return bgr, (0,0)
    return bgr[top:bottom, left:right], (left, top)

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
    ang = math.degrees(math.atan2(-(sy), sx))
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

# ──────────────────────────────────────────────────────────────────────────────
# Texto PDF (página 2): dos líneas desde columna (preferente)
# ──────────────────────────────────────────────────────────────────────────────
def page2_words(pdf_bytes: bytes):
    pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    try:
        if len(pdf.pages) < 2:
            return None, None, None
        p = pdf.pages[1]
        words = p.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        return p, words, pdf
    except Exception:
        pdf.close()
        return None, None, None

def find_column_near_y(words, y_pt: float, y_tol: float = 220.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not words:
        return None, None, None
    cands = []
    for w in words:
        t = (w.get("text") or "").upper()
        top = float(w.get("top", 0.0))
        bottom = float(w.get("bottom", top + 10))
        if abs(((top+bottom)/2.0) - y_pt) <= y_tol:
            if "APELLIDOS" in t or "RAZON" in t or "RAZÓN" in t or "NOMBRE" in t or t == "NIF":
                cands.append(w)
    if not cands:
        heads = [w for w in words if (w.get("text") or "").upper() in ("APELLIDOS","NIF")]
        if not heads:
            return None, None, None
        rowy = float(min(heads, key=lambda w: abs(float(w.get("top",0.0))-y_pt)).get("top",0.0))
        cands = [w for w in words if abs(float(w.get("top",0.0))-rowy) < 30.0]

    x_left = None
    x_nif  = None
    header_bottom = None
    for w in cands:
        t = (w.get("text") or "").upper()
        x0 = float(w.get("x0", 0.0))
        bot = float(w.get("bottom", 0.0))
        if "APELLIDOS" in t:
            x_left = x0 if x_left is None else min(x_left, x0)
            header_bottom = max(header_bottom or 0.0, bot)
        if t == "NIF":
            x_nif = x0
            header_bottom = max(header_bottom or 0.0, bot)
    if x_left is None:
        xs = [float(w.get("x0",0.0)) for w in cands if (w.get("text") or "").upper() != "NIF"]
        if xs: x_left = min(xs)
    if x_left is None or x_nif is None or header_bottom is None:
        return None, None, None
    x_left  = max(0.0, x_left - 2.0)
    x_right = max(x_left + 10.0, x_nif - 4.0)
    return x_left, x_right, header_bottom

def words_in_rect(words, x0, y0, x1, y1) -> List[str]:
    line_words = []
    for w in words:
        wx0 = float(w.get("x0",0.0)); wx1 = float(w.get("x1",0.0))
        wt  = float(w.get("top",0.0)); wb  = float(w.get("bottom",0.0))
        if wx0 >= x0-0.5 and wx1 <= x1+0.5 and wt >= y0-0.5 and wb <= y1+0.5:
            txt = (w.get("text") or "").strip()
            if txt:
                line_words.append((wx0, txt))
    line_words.sort(key=lambda t: t[0])
    return [t for _x,t in line_words]

def join_words(words_list: List[str]) -> str:
    if not words_list: return ""
    s = " ".join(words_list)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def extract_two_lines_from_pdftext(pdf_bytes: bytes, row_y_px: int, bgr_shape: Tuple[int,int]) -> Tuple[str, str, dict]:
    p, words, pdf = page2_words(pdf_bytes)
    if p is None or words is None:
        return "", "", {"pdftext":"no_page2_or_no_words"}
    try:
        img_h, img_w = bgr_shape[:2]
        sx = p.width  / float(img_w)
        sy = p.height / float(img_h)

        y_pt = row_y_px * sy
        x_left, x_right, header_bottom = find_column_near_y(words, y_pt)
        if x_left is None:
            return "", "", {"pdftext":"no_header_near_y"}

        line_h = 18.0
        y1_0 = header_bottom + 4.0
        y1_1 = y1_0 + line_h
        w1 = words_in_rect(words, x_left, y1_0, x_right, y1_1)
        txt1 = join_words(w1)

        y2_0 = y1_1 + 2.0
        y2_1 = y2_0 + line_h
        w2 = words_in_rect(words, x_left, y2_0, x_right, y2_1)
        raw2 = join_words(w2)
        txt2 = raw2.strip()[:26]  # corte duro por 26

        name1 = pick_owner_from_text(txt1) or clean_owner_line(txt1)

        use2, why2 = is_probable_name_continuation(txt2)
        final2 = txt2 if use2 else ""
        return (name1 or "").strip(), final2, {
            "x_left": x_left, "x_right": x_right,
            "header_bottom": header_bottom,
            "line1_rect": [x_left, y1_0, x_right, y1_1],
            "line2_rect": [x_left, y2_0, x_right, y2_1],
            "txt1_raw": txt1,
            "txt2_raw": raw2,
            "txt2_final": final2,
            "second_line_used": use2,
            "second_line_reason": why2.get("reason","")
        }
    finally:
        pdf.close()

# ──────────────────────────────────────────────────────────────────────────────
# OCR anclado a cabecera “APELLIDOS” (fallback)
# ──────────────────────────────────────────────────────────────────────────────
def ocr_two_lines_anchored(bgr: np.ndarray, row_y: int) -> Tuple[str, str, dict]:
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.33)
    x_text1 = int(w * 0.96)

    pad_y = int(h * 0.08)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        return "", "", {"ocr":"no_band"}

    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(g)

    def find_header(data):
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])
        header_bot = None
        x_nif = None
        for t,lx,ty,ww,hh in zip(words,xs,ys,ws,hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bot = max(header_bot or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bot = max(header_bot or 0, ty + hh)
        return header_bot, x_nif

    header_bot, x_nif = None, None
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        hb, xn = find_header(data)
        if hb is not None:
            header_bot, x_nif = hb, xn
            break

    if header_bot is None:
        line_px = max(22, int(h * 0.018))
        y1_0 = y0s + int(0.35*(y1s - y0s))
        y1_1 = y1_0 + line_px
        y2_0 = y1_1 + int(line_px*0.30)
        y2_1 = y2_0 + line_px
    else:
        y1_0 = y0s + int(header_bot + 4)
        y1_1 = y1_0 + max(22, int(h*0.018))
        y2_0 = y1_1 + max(6, int(h*0.006))
        y2_1 = y2_0 + max(22, int(h*0.018))

    def _ocr_roi(x0,y0,x1,y1):
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0: return ""
        gg = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gg = cv2.resize(gg, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        gg = enhance_gray(gg)
        bw2, bwi2 = binarize(gg)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        for im in (bw2,bwi2):
            for psm in (6,7,13):
                t = ocr_text(im, psm=psm, whitelist=WL)
                if t: return t
        return ""

    t1 = _ocr_roi(x_text0, y1_0, x_text1, y1_1)
    t2 = _ocr_roi(x_text0, y2_0, x_text1, y2_1)

    name1 = pick_owner_from_text(t1) or clean_owner_line(t1)
    ok2, why2 = is_probable_name_continuation(t2)
    line2 = (t2.strip()[:26] if ok2 else "")

    return name1, line2, {
        "band":[x_text0,y0s,x_text1,y1s],
        "y_line1":[y1_0,y1_1],
        "y_line2":[y2_0,y2_1],
        "t1_raw":t1, "t2_raw":t2,
        "second_line_used": ok2,
        "second_line_reason": why2.get("reason",""),
        "header_found": header_bot is not None
    }

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (detectar N/S/E/O y extraer 1ª+2ª línea)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            pdf_bytes: Optional[bytes],
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"rows":[],"why":"no_main_green"}, vis

    mains_abs  = [(cx+ox, cy+oy, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+ox, cy+oy, a) for (cx,cy,a) in neighs]

    linderos = {"norte":"","sur":"","este":"","oeste":""}
    used_sides = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano → lado
        best, best_d = None, 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        # Preferente: PDF-text
        name1_pdf = name2_pdf = ""
        dbg_pdf = {}
        if pdf_bytes is not None:
            try:
                n1,n2,dbg_pdf = extract_two_lines_from_pdftext(pdf_bytes, row_y_px=mcy, bgr_shape=bgr.shape)
                name1_pdf, name2_pdf = n1, n2
            except Exception as e:
                dbg_pdf = {"pdftext_exception": str(e)}

        # Fallback: OCR
        name1, name2 = name1_pdf, name2_pdf
        dbg_ocr = {}
        if not name1 or not name2:
            o1,o2,dbg_ocr = ocr_two_lines_anchored(bgr, row_y=mcy)
            if not name1 and o1: name1 = o1
            if not name2 and o2: name2 = o2

        owner = (f"{name1} {name2}".strip() if name1 else name2).strip()

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
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

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "pdftext": dbg_pdf,
            "ocr": dbg_ocr
        })

    dbg = {"rows": rows_dbg}
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
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, pdf_bytes=pdf_bytes, annotate=bool(labels), annotate_names=bool(names)
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
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, pdf_bytes=pdf_bytes, annotate=False)
        owners_detected = []
        for r in vdbg.get("rows", []):
            if r.get("owner"):
                owners_detected.append(r["owner"])
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )


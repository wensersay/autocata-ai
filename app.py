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
app = FastAPI(title="AutoCatastro AI", version="0.6.1")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN     = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE      = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY      = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# Raster DPI (override por entorno si quieres)
FAST_DPI       = int(os.getenv("FAST_DPI", "340") or "340")
SLOW_DPI       = int(os.getenv("SLOW_DPI", "500") or "500")

# Direcciones: 8-way vs 4-way + snapping angular opcional
DIAG_MODE      = os.getenv("DIAG_MODE", "8").strip() or "8"  # "8" o "4"
try:
    _snap = os.getenv("DIAG_SNAP_DEG", "").strip()
    DIAG_SNAP_DEG: Optional[int] = int(_snap) if _snap else None
except:
    DIAG_SNAP_DEG = None

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

# ruído geográfico / dirección (para no contaminar la línea 1)
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Nombres frecuentes (permite concatenar 2ª línea corta si aparece)
NAME_HINTS = {
    "JOSE","JOSÉ","LUIS","MARIA","MARÍA","ANTONIO","MANUEL","FRANCISCO","CARLOS",
    "JAVIER","PEDRO","ANGEL","ÁNGEL","MARTA","ANA","ISABEL","PABLO","ALVARO","ÁLVARO",
    "DAVID","LAURA","SARA","RAFAEL","DANIEL","CRISTINA","ROSA","LUISA","ENRIQUE",
    "EMILIO","GONZALO","OSCAR","ÓSCAR","FERNANDO","ALFONSO","ANDRES","ANDRÉS"
}
_extra = os.getenv("NAME_HINTS_EXTRA","").strip()
if _extra:
    for tok in _extra.split(","):
        t = tok.strip().upper()
        if t:
            NAME_HINTS.add(t)

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
    dpi = FAST_DPI if FAST_MODE else SLOW_DPI
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
# Direcciones 8-way + snapping/colapso opcional
# ──────────────────────────────────────────────────────────────────────────────
CARD_ANGLES = { "este": 0.0, "norte": 90.0, "oeste": 180.0, "sur": -90.0 }

def angle_from_main(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> float:
    cx, cy = main_xy; x, y = pt_xy
    sx, sy = x - cx, y - cy
    return math.degrees(math.atan2(-(sy), sx))

def side_8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> Tuple[str, float]:
    ang = angle_from_main(main_xy, pt_xy)
    if -22.5 <= ang <= 22.5:   side = "este"
    elif 22.5 < ang <= 67.5:   side = "noreste"
    elif 67.5 < ang <= 112.5:  side = "norte"
    elif 112.5 < ang <= 157.5: side = "noroeste"
    elif ang > 157.5 or ang <= -157.5: side = "oeste"
    elif -157.5 < ang <= -112.5: side = "suroeste"
    elif -112.5 < ang <= -67.5:  side = "sur"
    else:                        side = "sureste"
    return side, ang

def nearest_cardinal(ang: float) -> str:
    best = None; best_d = 1e9
    for k, a in CARD_ANGLES.items():
        d = abs((ang - a + 180) % 360 - 180)
        if d < best_d:
            best_d = d; best = k
    return best or "este"

def apply_diag_prefs(side: str, ang: float) -> str:
    if DIAG_SNAP_DEG is not None:
        cand = nearest_cardinal(ang)
        d = abs((ang - CARD_ANGLES[cand] + 180) % 360 - 180)
        if d <= float(DIAG_SNAP_DEG):
            side = cand
    if DIAG_MODE == "4" and side not in ("norte","sur","este","oeste"):
        side = nearest_cardinal(ang)
    return side

# ──────────────────────────────────────────────────────────────────────────────
# OCR utils (línea 1 + posible continuación)
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
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
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
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):            continue
        name = clean_owner_line(U)
        if len(name) >= 8:
            return name
    return ""

def strip_leading_singletons(s: str) -> str:
    """Quita tokens iniciales de 1 carácter (p.ej. 'P ', 'Y ')."""
    if not s: return s
    toks = [t for t in re.split(r"\s+", s.strip()) if t]
    while toks and len(toks[0]) == 1:
        toks.pop(0)
    return " ".join(toks)

# Localizar banda bajo “Apellidos…/Razón social”
def find_owner_band(bgr: np.ndarray, row_y: int, x_left: int, x_right: int) -> Tuple[int,int,int,int,int,int]:
    """
    Devuelve (x0,x1,y0,y1, header_left_abs, x_nif_abs)
    x0 ahora se alinea con el borde izquierdo del encabezado detectado.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_left:x_right]
    if band.size == 0:
        y0 = max(0, row_y - int(h*0.01)); y1 = min(h, y0 + int(h*0.035))
        return x_left, int(x_left + 0.55*(x_right-x_left)), y0, y1, -1, -1

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", []); xs = data.get("left", [])
        ys = data.get("top", []); ws = data.get("width", []); hs = data.get("height", [])

        x_nif_local = None
        header_bottom = None
        header_left_local = None

        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T or "RAZON" in T or "RAZÓN" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
                header_left_local = min(header_left_local, lx) if header_left_local is not None else lx
            if T == "NIF":
                x_nif_local = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            y0 = y0s + header_bottom + 6
            y1 = min(h, y0 + int(h * 0.035))
            if x_nif_local is not None:
                x0 = x_left + (header_left_local - 4 if header_left_local is not None else 0)
                x0 = max(x_left, x0)
                x1 = min(x_right, x_left + x_nif_local - 8)
            else:
                x0 = x_left + (header_left_local - 4 if header_left_local is not None else 0)
                x0 = max(x_left, x0)
                x1 = int(x0 + 0.55*(x_right-x_left))

            if x1 - x0 > (x_right - x_left) * 0.22:
                return x0, x1, y0, y1, (x_left + (header_left_local or 0)), (x_left + (x_nif_local or 0))

    # Fallback
    y0 = max(0, row_y - int(h*0.01)); y1 = min(h, y0 + int(h*0.035))
    x0 = x_left; x1 = int(x_left + 0.55*(x_right-x_left))
    return x0, x1, y0, y1, -1, -1

def extract_owner_for_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Devuelve (owner, dbg). Intenta leer línea 1 (nombre) y posible continuación.
    """
    h, w = bgr.shape[:2]
    # zona de texto (columna de detalles)
    x_text0 = int(w * 0.29)  # ligeramente a la izquierda
    x_text1 = int(w * 0.92)

    # localizar banda "nombre" (línea 1)
    x0, x1, y0, y1, header_left_abs, x_nif_abs = find_owner_band(bgr, row_y, x_text0, x_text1)
    band_h = max(8, y1 - y0)

    # construir dos líneas verticalmente contiguas (l1 y l2)
    l1_y0, l1_y1 = y0, y0 + band_h
    l2_y0, l2_y1 = min(h, l1_y1 + 2), min(h, l1_y1 + band_h + 2)

    def read_line(x0, y0, x1, y1) -> str:
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0: return ""
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        candidates = [
            ocr_text(bw, 6, WL), ocr_text(bwi, 6, WL),
            ocr_text(bw, 7, WL), ocr_text(bwi, 7, WL),
            ocr_text(bw, 13, WL)
        ]
        cand = max((c or "" for c in candidates), key=lambda s: len(s), default="")
        return cand.strip()

    raw1 = read_line(x0, l1_y0, x1, l1_y1)
    raw2 = read_line(x0, l2_y0, x1, l2_y1)

    # quitar ruidos como "P " / "Y " al inicio
    raw1 = strip_leading_singletons(raw1)
    raw2 = strip_leading_singletons(raw2)

    # si Tesseract partió la 1ª en dos líneas, recupéralo
    extra_from_l1 = ""
    if "\n" in raw1:
        parts = [p.strip() for p in raw1.split("\n") if p.strip()]
        if len(parts) >= 2:
            raw1 = parts[0]
            extra_from_l1 = parts[1][:26]

    # limpiar y decidir
    name1 = pick_owner_from_text(raw1)

    def sanitize_l2(s: str) -> str:
        s = (s or "").upper()
        # corta al primer número / [ ] :
        s = re.split(r"[\[\]:0-9]", s)[0]
        s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ '\-]", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        # eliminar tokens iniciales de 1 carácter por si acaso
        s = strip_leading_singletons(s)
        return s[:26]

    name2 = ""
    if extra_from_l1:
        name2 = sanitize_l2(extra_from_l1)
    elif raw2:
        name2 = sanitize_l2(raw2)

    # Reglas de concatenación robustas:
    # - unir solo si name2 tiene ≥3 caracteres o está en NAME_HINTS
    # - y no contiene tokens geográficos claros
    if name1:
        if name2 and (len(name2) >= 3 or name2 in NAME_HINTS) and not any(tok in name2 for tok in GEO_TOKENS):
            full = f"{name1} {name2}".strip()
        else:
            full = name1
    else:
        full = name2  # fallback extremo (raro)

    dbg = {
        "band":[x0, y0, x1, y1],
        "y_line1":[l1_y0, l1_y1],
        "y_line2_hint":[l2_y0, l2_y1],
        "x0": x0, "x1": x1,
        "t1_raw": raw1, "t1_extra_raw": extra_from_l1,
        "t2_raw": raw2,
        "header_left_abs": header_left_abs,
        "x_nif_abs": x_nif_abs
    }
    return full, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (mapa → vecinos → lado 8-way → nombre por columna)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:

    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte donde están croquis (izquierda) vs tabla (derecha)
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return (
            {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            {"rows": [], "timings_ms":{"rows_pipeline": int((time.time()-t0)*1000)}},
            vis
        )

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # por Y ascendente (fila 1..4)
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_dirs = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:6]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)

        if best is None:
            rows_dbg.append({
                "row_y": mcy, "main_center":[mcx,mcy], "neigh_center": None,
                "side": "", "owner": "", "ocr": {}
            })
            continue

        raw_side, ang = side_8((mcx,mcy), best)
        side = apply_diag_prefs(raw_side, ang)

        owner, odbg = extract_owner_for_row(bgr, row_y=mcy)

        if side and owner and side not in used_dirs:
            linderos[side] = owner
            used_dirs.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            cv2.circle(vis, best, 8, (0,0,255), -1)
            lbl = {
                "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"
            }.get(side,"")
            if lbl:
                cv2.putText(vis, lbl, (best[0]-10, best[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.43), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.43), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center":[mcx,mcy],
            "neigh_center":[best[0],best[1]],
            "side": side,
            "owner": owner,
            "ocr": odbg
        })

    dbg = {
        "rows": rows_dbg,
        "timings_ms":{"rows_pipeline": int((time.time()-t0)*1000)},
        "raster":{"dpi": FAST_DPI if FAST_MODE else SLOW_DPI}
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
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)},
        "diag": {"mode": DIAG_MODE, "snap": DIAG_SNAP_DEG},
        "dpi": {"fast": FAST_DPI, "slow": SLOW_DPI}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar puntos cardinales/diagonales"),
    names: int = Query(0, description="1=mostrar nombre estimado en cada fila")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _l, _dbg, vis = detect_rows_and_extract(
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
        owners_detected = [r["owner"] for r in vdbg.get("rows", []) if r.get("owner")]
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





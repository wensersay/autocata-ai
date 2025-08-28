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
app = FastAPI(title="AutoCatastro AI", version="0.5.7")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# DPI: si FAST_MODE=1 usa FAST_DPI; si no, usa PDF_DPI
def get_target_dpi() -> int:
    if FAST_MODE:
        return int(os.getenv("FAST_DPI", "340"))
    return int(os.getenv("PDF_DPI", "400"))

# Hints extra desde entorno (coma-separado)
def get_name_hints_extra() -> List[str]:
    raw = os.getenv("NAME_HINTS_EXTRA", "").strip()
    if not raw:
        return []
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

# Junk de 2ª línea desde entorno (coma-separado)
def get_junk_2ndline() -> List[str]:
    raw = os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").strip()
    if not raw:
        return []
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

JUNK_2NDLINE = set(get_junk_2ndline())

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

GEO_TOKENS = {
    "LUGO","[LUGO]","BARCELONA","[BARCELONA]","MADRID","VALENCIA","SEVILLA",
    "A CORUÑA","CORUÑA","MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET",
    "SAVINAO","SAVIÑAO","O","OS","DE","DEL"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

def looks_geo_or_noise(s: str) -> bool:
    if not s:
        return True
    U = s.upper()
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", U) if t]
    if len("".join(toks)) < 6:
        return True
    if toks:
        bad = 0
        for t in toks:
            if t in GEO_TOKENS or any(ch.isdigit() for ch in t) or t in JUNK_2NDLINE:
                bad += 1
        if bad / len(toks) >= 0.5:
            return True
    return False

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
    dpi = get_target_dpi()
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

# 8 direcciones
def side_of_8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=E, 90=N
    # normalizar a (-180, 180]
    if ang <= -180: ang += 360
    if ang > 180: ang -= 360
    # sectores de 45° con bisectrices en E(0), NE(45)...
    if -22.5 <= ang <= 22.5:   return "este"
    if 22.5 < ang <= 67.5:     return "noreste"
    if 67.5 < ang <= 112.5:    return "norte"
    if 112.5 < ang <= 157.5:   return "noroeste"
    if ang > 157.5 or ang <= -157.5: return "oeste"
    if -157.5 < ang <= -112.5: return "suroeste"
    if -112.5 < ang <= -67.5:  return "sur"
    if -67.5 < ang < -22.5:    return "sureste"
    return "oeste"

SIDE_LABEL = {
    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
    "sur":"S","suroeste":"SW","oeste":"W","noroeste":"NW"
}

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

# ──────────────────────────────────────────────────────────────────────────────
# ROI: banda de "Apellidos / Nombre" con ancho mínimo + NIF si existe (Parche)
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_band(bgr: np.ndarray, row_y: int, x_text0: int, x_text1: int) -> Tuple[int,int,int,int]:
    """
    Devuelve (x0, x1, y0, y1) para la banda del NOMBRE del titular.
    Si el ancho resultante es demasiado pequeño, forzamos una franja segura.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    # banda provisional para localizar cabecera vía data
    band = bgr[y0s:y1s, x_text0:x_text1]
    min_band_w = max(int(w * 0.22), 320)  # mínimo razonable

    def safe_box(from_x0: int, to_x1: int, header_bottom_px: Optional[int]) -> Tuple[int,int,int,int]:
        if header_bottom_px is not None:
            y0 = y0s + header_bottom_px + 6
            y1 = min(h, y0 + int(h * 0.035))
        else:
            y0 = max(0, row_y - int(h*0.01))
            y1 = min(h, y0 + int(h * 0.035))
        x0 = max(0, from_x0)
        x1 = min(w, to_x1)
        if (x1 - x0) < min_band_w:
            x0 = max(0, int(w * 0.28))
            x1 = min(w, x0 + max(min_band_w, int(w * 0.34)))
        return x0, x1, y0, y1

    if band.size == 0:
        return safe_box(x_text0, x_text1, None)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])

        x_nif = None
        header_bottom = None

        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t:
                continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if "NIF" in T:
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            if x_nif is not None:
                abs_x_nif = x_text0 + x_nif
                return safe_box(x_text0, abs_x_nif - 6, header_bottom)
            else:
                return safe_box(x_text0, int(x_text0 + 0.55 * (x_text1 - x_text0)), header_bottom)

    return safe_box(x_text0, int(x_text0 + 0.55 * (x_text1 - x_text0)), None)

# ──────────────────────────────────────────────────────────────────────────────
# Extraer titular por fila (Parche: reintentos a la izquierda + L1/L1extra/L2)
# ──────────────────────────────────────────────────────────────────────────────
def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, Tuple[int,int,int,int], int, dict]:
    """
    (owner, (x0,y0,x1,y1), attempt_used, ocr_debug)
    Reintenta expandiendo ROI hacia la izquierda si el texto sale corto o geográfico.
    """
    h, w = bgr.shape[:2]
    base_left = int(w * 0.30)  # empezamos algo más a la izquierda
    x_text0 = max(0, base_left)
    x_text1 = int(w * 0.96)

    best_txt = ""
    best_box = (x_text0, max(0, row_y - int(h*0.01)), int(w*0.64), min(h, row_y + int(h*0.035)))
    best_attempt = 0
    best_dbg = {}

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    name_hints = set(get_name_hints_extra())

    for attempt in range(3):
        extra_left = attempt * max(140, int(w * 0.03))
        x0_base = max(0, x_text0 - extra_left)
        x0, x1, y0, y1 = find_header_and_owner_band(bgr, row_y, x0_base, x_text1)

        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)

        t_full = ocr_text(bw,  psm=6,  whitelist=WL) or ocr_text(bwi, psm=6, whitelist=WL)
        lns = [ln.strip() for ln in (t_full or "").split("\n") if ln.strip()]
        l1 = lns[0] if lns else ""
        l1_extra = lns[1] if len(lns) > 1 else ""

        mid_y = (y0 + y1) // 2
        roi2 = bgr[mid_y:y1, x0:x1]
        l2 = ""
        if roi2.size != 0:
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)
            l2 = ocr_text(bw2,  psm=7, whitelist=WL) or ocr_text(bwi2, psm=7, whitelist=WL)
            l2 = (l2 or "").strip()

        def clean_line(s: str) -> str:
            s = s.replace("  ", " ").strip()
            s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ '\-]", " ", s.upper())
            s = re.sub(r"\s{2,}", " ", s)
            return s.strip()

        l1_clean = clean_line(l1)
        l1_extra_clean = clean_line(l1_extra)
        l2_clean = clean_line(l2)

        picked_from = "strict"
        cand = l1_clean

        # concatenar segunda línea si no parece geo/ruido ni está en junk explícito
        if l1_extra_clean and not looks_geo_or_noise(l1_extra_clean):
            cand = (cand + " " + l1_extra_clean).strip()
            picked_from = "from_l1_break"

        if l2_clean and (l2_clean not in JUNK_2NDLINE) and not looks_geo_or_noise(l2_clean):
            cand = (cand + " " + l2_clean).strip()

        # hints de nombres (si alguno aparece, mantenlo)
        if name_hints:
            toks = set(re.split(r"\s+", cand))
            if not toks & name_hints and l1_extra_clean:
                # intenta añadir hint de L1extra si es un nombre conocido
                ex_toks = set(re.split(r"\s+", l1_extra_clean))
                if ex_toks & name_hints:
                    cand = (cand + " " + l1_extra_clean).strip()
                    picked_from = "from_l1_break"

        cand = re.sub(r"\s{2,}", " ", cand).strip()[:48]

        ocr_dbg = {
            "band":[x0,y0,x1,y1],
            "y_line1":[y0, (y0+y1)//2],
            "y_line2_hint":[(y0+y1)//2, y1],
            "x0":x0,"x1":x1,
            "t1_raw": l1_clean,
            "t1_extra_raw": l1_extra_clean,
            "t2_raw": l2_clean,
            "picked_from": picked_from
        }

        if not cand or looks_geo_or_noise(cand) or len(cand.replace(" ", "")) < 6:
            if len(cand) > len(best_txt):
                best_txt, best_box, best_attempt, best_dbg = cand, (x0,y0,x1,y1), attempt, ocr_dbg
            continue

        return cand, (x0,y0,x1,y1), attempt, ocr_dbg

    return best_txt, best_box, best_attempt, best_dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # zona izquierda donde están los croquis
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    if not mains:
        empty8 = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
        return empty8, {"rows": [], "raster":{"dpi": get_target_dpi()}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:8]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of_8((mcx, mcy), best)

        owner, (x0,y0,x1,y1), attempt_id, ocr_dbg = extract_owner_from_row(bgr, row_y=mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = SIDE_LABEL.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:26], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:26], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)

        row_info = {
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
        }
        rows_dbg.append(row_info)

    dbg = {"rows": rows_dbg, "raster":{"dpi": get_target_dpi()}}
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
        "dpi": get_target_dpi(),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SW/W/NW"),
    names: int = Query(0, description="1=mostrar nombre abreviado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((260, 820, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:70]}", (10,140),
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
        # lista única de propietarios válidos
        owners_detected = []
        for r in vdbg.get("rows", []):
            o = (r.get("owner") or "").strip()
            if o and o not in owners_detected:
                owners_detected.append(o)
        owners_detected = owners_detected[:8]

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



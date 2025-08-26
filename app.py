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
# Flags de entorno / seguridad / tuning
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI de raster para página 2 (se puede sobreescribir por ENV)
def _parse_int(envname: str, default: int) -> int:
    try:
        return int(os.getenv(envname, str(default)).strip())
    except Exception:
        return default

DPI_PAGE2 = _parse_int("DPI_PAGE2", 340)  # por defecto 340 para mejor rendimiento

# Clasificación 4/8 direcciones
EIGHT_WAY = (os.getenv("EIGHT_WAY", "1").strip() == "1")
try:
    ANGLE_TOL = float(os.getenv("ANGLE_TOL_DEG", "22.5"))
except ValueError:
    ANGLE_TOL = 22.5

# Pistas de nombres (lista base + extras por ENV)
BASE_NAME_HINTS = {
    # nombres frecuentes (mayúsculas)
    "JOSE","JOSÉ","LUIS","MARIA","MARÍA","ANTONIO","MANUEL","FRANCISCO","JUAN","JAVIER","CARLOS",
    "ANA","ISABEL","PABLO","MARTA","RAFAEL","ALBERTO","ROSA","DAVID","SERGIO","ALEJANDRO","ALVARO","ÁLVARO",
    "LOPEZ","LÓPEZ","GARCIA","GARCÍA","RODRIGUEZ","RODRÍGUEZ","FERNANDEZ","FERNÁNDEZ","ALVAREZ","ÁLVAREZ",
    "PEREZ","PÉREZ","SANCHEZ","SÁNCHEZ","GOMEZ","GÓMEZ","MARTINEZ","MARTÍNEZ","DIEZ","DÍEZ","DIAZ","DÍAZ",
    "VÁZQUEZ","VAZQUEZ","POMBO","DOSINDA","MOSQUERA","VARELA"
}
EXTRA = {t.strip().upper() for t in os.getenv("NAME_HINTS_EXTRA", "").split(",") if t.strip()}
NAME_HINTS = BASE_NAME_HINTS.union(EXTRA)

# Tokens no deseados en nombres
BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","CSV","ESCALA","HUSO","ETRS"
}

# Geo/ruido que no deben entrar en el nombre
GEO_TOKENS = {"LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A", "A CORUÑA",
              "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
              "GALICIA","[LUGO]","[BARCELONA]"}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}
UPPER_NAME_RE   = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

# OpenCV flags (compat.)
def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Auth header
# ──────────────────────────────────────────────────────────────────────────────
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
# Utilidades
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

def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = DPI_PAGE2
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

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

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int],
            eight_way: bool = EIGHT_WAY, tol: float = ANGLE_TOL) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    ang = math.degrees(math.atan2(-(y - cy), (x - cx)))  # 0°=Este, CCW+

    if not eight_way:
        if -45 <= ang <= 45:     return "este"
        if 45 < ang <= 135:      return "norte"
        if -135 <= ang < -45:    return "sur"
        return "oeste"

    centers = [
        (   0.0, "este"),
        (  45.0, "noreste"),
        (  90.0, "norte"),
        ( 135.0, "noroeste"),
        ( 180.0, "oeste"),
        (-135.0, "suroeste"),
        ( -90.0, "sur"),
        ( -45.0, "sureste"),
    ]
    def circ_dist(a, b):
        d = abs(a - b)
        return min(d, 360.0 - d)
    best = min(centers, key=lambda c: circ_dist(ang, c[0]))
    return best[1]

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

def pick_owner_strict(txt: str) -> str:
    """Elige la mejor línea tipo nombre en MAYÚSCULAS sin números."""
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 0:     continue
        if not UPPER_NAME_RE.match(U):            continue
        name = clean_owner_line(U)
        if len(name) >= 6:
            return name
    return ""

def all_alpha_space(s: str) -> bool:
    return bool(re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ ]{2,26}", s or ""))

def tokens_in_hints(s: str) -> bool:
    toks = [t for t in (s or "").split(" ") if t]
    if not toks: return False
    return all((t in NAME_HINTS) for t in toks)

# ──────────────────────────────────────────────────────────────────────────────
# Detección por filas + extracción de la columna “Apellidos Nombre / Razón Social”
# ──────────────────────────────────────────────────────────────────────────────
def find_header_band_and_nif_x(bgr: np.ndarray, row_y: int, x_text0: int, x_text1: int):
    """Busca 'APELLIDOS' y 'NIF' cerca de row_y para anclar y recortar la columna."""
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        return {"header_found": False, "x_nif_found": False, "header_bottom": None, "x_nif": None}

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    dbg = {"header_found": False, "x_nif_found": False, "header_bottom": None, "x_nif": None}
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", []); xs = data.get("left", []); ys = data.get("top", [])
        ws = data.get("width", []); hs = data.get("height", [])

        header_bottom = None
        x_nif = None
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            dbg["header_found"]  = True
            dbg["header_bottom"] = y0s + header_bottom
            if x_nif is not None:
                dbg["x_nif_found"] = True
                dbg["x_nif"]       = x_text0 + x_nif
            break

    return dbg

def extract_owner_two_lines(bgr: np.ndarray, row_y: int):
    """
    Devuelve (owner, ocr_dbg)
      - Línea 1: nombre principal (modo estricto).
      - Línea 2: si existe y pasa filtros suaves (long≤26, A-Z/espacios, tokens en NAME_HINTS → “muy seguro”),
                 se concatena al final. Si L1 ya trae salto con el nombre (caso '...JOSE\\nLUIS'), se usa sin mirar L2.
    """
    h, w = bgr.shape[:2]
    # columna de texto a la derecha
    x_text0 = int(w * 0.24)  # ligeramente más a la izquierda para no cortar primeras letras
    x_text1 = int(w * 0.62)  # aprox. antes de NIF

    hdr = find_header_band_and_nif_x(bgr, row_y, x_text0, x_text1)
    if hdr.get("x_nif_found"):
        x_text1 = min(x_text1, int(hdr["x_nif"] - 6))

    # Banda vertical para L1/L2 (conservador)
    band_top    = max(0, row_y - int(h * 0.10))
    band_bottom = min(h, row_y + int(h * 0.10))
    band = bgr[band_top:band_bottom, x_text0:x_text1]

    # Líneas aproximadas (L1 y L2)
    # Si detectamos header_bottom, ponemos L1 justo debajo; si no, centrado por row_y
    if hdr.get("header_found") and hdr.get("header_bottom") is not None:
        l1_top  = max(0, int(hdr["header_bottom"] + 4))
        l1_bot  = min(h, l1_top + int(h * 0.040))
    else:
        l1_top  = max(0, row_y - int(h * 0.015))
        l1_bot  = min(h, row_y + int(h * 0.015))

    l2_top  = min(h, l1_bot + 8)
    l2_bot  = min(h, l2_top + int(h * 0.040))

    # OCR línea 1
    x0, x1 = x_text0, x_text1
    roi1 = bgr[l1_top:l1_bot, x0:x1]
    owner = ""
    t1_raw = ""
    t1_extra_raw = ""
    second_used = False
    second_reason = ""
    t2_raw = ""

    if roi1.size != 0:
        g = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
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
        for txt in variants:
            if "\n" in txt:  # caso '...JOSE\nLUIS' en un único recorte
                parts = [p.strip() for p in txt.split("\n") if p.strip()]
                if parts:
                    t1_raw = parts[0].upper()
                    t1_extra_raw = " ".join(parts[1:]).upper()[:26]
                    name1 = pick_owner_strict(t1_raw)
                    if name1:
                        owner = name1
                        if all_alpha_space(t1_extra_raw) and tokens_in_hints(t1_extra_raw):
                            owner = f"{owner} {t1_extra_raw}"
                            second_used = True
                            second_reason = "from_l1_break"
                        break
            t1_raw = txt.upper()
            name1 = pick_owner_strict(t1_raw)
            if name1:
                owner = name1
                break

    # OCR línea 2 (solo si no vino con L1)
    if owner and not second_used:
        roi2 = bgr[l2_top:l2_bot, x0:x1]
        if roi2.size != 0:
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)
            WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
            variants2 = [
                ocr_text(bw2,  psm=7, whitelist=WL),
                ocr_text(bwi2, psm=7, whitelist=WL),
                ocr_text(bw2,  psm=6, whitelist=WL),
                ocr_text(bwi2, psm=6, whitelist=WL),
                ocr_text(bw2,  psm=13, whitelist=WL),
            ]
            for txt2 in variants2:
                t2 = txt2.upper().strip()
                t2 = re.split(r"[\d\[\]:]", t2)[0].strip()  # cortar en números/[/]:
                t2 = re.sub(r"\s{2,}", " ", t2)
                t2 = t2[:26]
                if t2:
                    t2_raw = t2
                    if all_alpha_space(t2) and tokens_in_hints(t2):
                        owner = f"{owner} {t2}"
                        second_used = True
                        second_reason = "2nd_ok_by_hints"
                        break

    ocr_dbg = {
        "band": [x0, band_top, x1, band_bottom],
        "y_line1": [l1_top, l1_bot],
        "y_line2_hint": [l2_top, l2_bot],
        "x0": x0, "x1": x1,
        "t1_raw": t1_raw, "t1_extra_raw": t1_extra_raw,
        "t2_raw": t2_raw,
        "picked_from": "strict" if owner else "none",
        "second_line_used": second_used,
        "second_line_reason": second_reason,
        "header_found": bool(hdr.get("header_found")),
        "x_nif_found": bool(hdr.get("x_nif_found")),
    }
    return owner, ocr_dbg

def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte aproximado del área de croquis (izquierda) para buscar verde/rosa
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        # 8 claves si EIGHT_WAY; 4 si no
        if EIGHT_WAY:
            lind = {k: "" for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
        else:
            lind = {"norte":"","sur":"","este":"","oeste":""}
        return lind, {"rows": []}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]
    mains_abs.sort(key=lambda t: t[1])  # por fila (y ascendente)

    if EIGHT_WAY:
        linderos = {k: "" for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
        lblmap = {"norte":"N","noreste":"NE","este":"E","sureste":"SE","sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"}
    else:
        linderos = {"norte":"","sur":"","este":"","oeste":""}
        lblmap = {"norte":"N","sur":"S","este":"E","oeste":"O"}

    used_sides = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecino más cercano
        best = None; best_d = 1e18
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        owner, ocr_dbg = extract_owner_two_lines(bgr, row_y=mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = lblmap.get(side, "")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-10),
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

    t1 = time.time()
    dbg = {"rows": rows_dbg, "timings_ms": {"rows_pipeline": int((t1-t0)*1000)} , "raster": {"dpi": DPI_PAGE2}}
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
        "dpi_page2": DPI_PAGE2,
        "eight_way": EIGHT_WAY,
        "angle_tol_deg": ANGLE_TOL,
        "name_hints_extra": sorted(EXTRA),
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(1, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
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
                 labels: int = Query(1),
                 names: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        lind = {k: "" for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")} if EIGHT_WAY \
               else {"norte":"","sur":"","este":"","oeste":""}
        return ExtractOut(
            linderos=lind,
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg.get("rows", []) if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        lind = {k: "" for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")} if EIGHT_WAY \
               else {"norte":"","sur":"","este":"","oeste":""}
        return ExtractOut(
            linderos=lind,
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )



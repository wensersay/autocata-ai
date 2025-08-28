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
app = FastAPI(title="AutoCatastro AI", version="0.6.0")

# ──────────────────────────────────────────────────────────────────────────────
# Flags / entorno
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI configurables
def _as_int(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, str(default)).strip())
    except Exception:
        return default

FAST_DPI = _as_int("FAST_DPI", 340)   # usado cuando FAST_MODE=1
SLOW_DPI = _as_int("SLOW_DPI", 500)   # usado cuando FAST_MODE=0

# Lista de “ruidos” típicos en segunda línea (override por ENV)
def _parse_junk_env() -> set:
    raw = os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR")
    toks = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return set(toks)
JUNK_2NDLINE = _parse_junk_env()

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

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
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

# 8 rumbos
def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = (math.degrees(math.atan2(-(sy), sx)) + 360.0) % 360.0
    # sectores de 45°
    if 337.5 <= ang or ang < 22.5:   return "este"
    if 22.5 <= ang < 67.5:           return "noreste"
    if 67.5 <= ang < 112.5:          return "norte"
    if 112.5 <= ang < 157.5:         return "noroeste"
    if 157.5 <= ang < 202.5:         return "oeste"
    if 202.5 <= ang < 247.5:         return "suroeste"
    if 247.5 <= ang < 292.5:         return "sur"
    if 292.5 <= ang < 337.5:         return "sureste"
    return ""

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
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):            continue
        name = clean_owner_line(U)
        if len(name) >= 8:
            return name
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Localizar cabecera y extraer línea 1 + línea 2 (parche anti-ruido y L1-break)
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_band(bgr: np.ndarray, row_y: int,
                               x_text0: int, x_text1: int) -> Tuple[int,int,int,int]:
    """
    Devuelve (x0, x1, y0, y1) para la banda del NOMBRE del titular.
    Se busca 'APELLIDOS' y se coloca y0 justo debajo.
    Alto ~ 6% de página para cubrir dos líneas.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        # Fallback
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + int(h*0.06))
        return x_text0, int(x_text0 + 0.55*(x_text1-x_text0)), y0, y1

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
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            y0 = y0s + header_bottom + 6
            y1 = min(h, y0 + int(h * 0.06))  # dos líneas aprox
            if x_nif is not None:
                x0 = x_text0
                x1 = min(x_text1, x_text0 + x_nif - 8)
            else:
                x0 = x_text0
                x1 = int(x_text0 + 0.55*(x_text1-x_text0))
            if x1 - x0 > (x_text1 - x_text0) * 0.22:
                return x0, x1, y0, y1

    # fallback
    y0 = max(0, row_y - int(h*0.01))
    y1 = min(h, y0 + int(h*0.06))
    x0 = x_text0
    x1 = int(x_text0 + 0.55*(x_text1-x_text0))
    return x0, x1, y0, y1

def _clean_second_line_raw(t2_raw: str) -> str:
    """Parche anti-ruido 2ª línea + lectura desde ENV JUNK_2NDLINE."""
    if not t2_raw:
        return ""
    t2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\s'\-]", " ", t2_raw.upper()).strip()
    t2 = re.sub(r"\s{2,}", " ", t2)[:26]
    if len(t2) < 3 or t2 in JUNK_2NDLINE:
        return ""
    return t2

def _pick_l1_break_extra(t1_raw: str) -> str:
    """
    Si Tesseract mete un salto de línea dentro de L1 (p.ej. '... JOSE\\nLUIS'),
    intenta devolver el primer token 'humano' de la segunda línea.
    """
    if not t1_raw:
        return ""
    lines = [l.strip().upper() for l in t1_raw.splitlines() if l.strip()]
    if len(lines) < 2:
        return ""
    # Tomamos la segunda línea y la limpiamos
    cand = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\s'\-]", " ", lines[1]).strip()
    cand = re.sub(r"\s{2,}", " ", cand)
    # Nos quedamos con el primer token útil
    toks = [t for t in cand.split() if t not in BAD_TOKENS and t not in GEO_TOKENS]
    if not toks:
        return ""
    t = toks[0][:26]
    if len(t) < 2 or t in JUNK_2NDLINE:
        return ""
    return t

def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Extrae la línea 1 (y opcionalmente la 2) del titular.
    Devuelve owner y un diccionario debug con cajas/elecciones.
    """
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.27)
    x_text1 = int(w * 0.96)

    x0, x1, y0, y1 = find_header_and_owner_band(bgr, row_y, x_text0, x_text1)
    band = bgr[y0:y1, x0:x1]
    dbg_ocr = {"band":[x0,y0,x1,y1]}

    if band.size == 0:
        return "", {"ocr":dbg_ocr}

    # dividimos la banda en dos mitades (L1 y L2)
    H = y1 - y0
    mid = y0 + H//2
    l1 = bgr[y0:mid, x0:x1]
    l2 = bgr[mid:y1, x0:x1]

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    def ocr_line(img):
        if img.size == 0: return ""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        for p in (bw,bwi):
            t = ocr_text(p, psm=6, whitelist=WL)
            if t: return t
        return ""

    t1_raw = ocr_line(l1)
    t2_raw = ocr_line(l2)

    dbg_ocr.update({
        "y_line1":[y0, mid], "y_line2_hint":[mid, y1],
        "x0":x0, "x1":x1, "t1_raw":t1_raw, "t2_raw":t2_raw
    })

    # Línea 1 (estricta)
    owner = pick_owner_from_text(t1_raw) or clean_owner_line(t1_raw)
    picked_from = "strict" if owner else "fallback"

    # Posible extra por salto dentro de L1 (e.g., '\nLUIS')
    extra_from_l1break = _pick_l1_break_extra(t1_raw)

    # Posible 2ª línea real (L2) con parche anti-ruido
    t2 = _clean_second_line_raw(t2_raw)

    # Concatenación:
    second_used = False
    second_reason = ""

    if owner and extra_from_l1break:
        owner = (owner + " " + extra_from_l1break).strip()
        second_used = True
        second_reason = "from_l1_break"
    elif owner and t2:
        # si l2 es un único token de 2..12 letras (e.g., LUIS, MARIA), únelo
        tok2 = [t for t in t2.split() if t not in BAD_TOKENS and t not in GEO_TOKENS]
        if len(tok2) == 1 and 2 <= len(tok2[0]) <= 12:
            owner = (owner + " " + tok2[0]).strip()
            second_used = True
            second_reason = "from_l2_single_token"

    dbg = {"ocr": dbg_ocr, "picked_from": picked_from, "second_line_used": second_used, "second_line_reason": second_reason}
    return owner[:62], dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (8 rumbos)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # zona izquierda con croquis pequeños
    top = int(h * 0.10); bottom = int(h * 0.93)
    left = int(w * 0.05); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 160))
    if not mains:
        l8 = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
        return l8, {"rows": [], "raster":{"dpi": FAST_DPI if FAST_MODE else SLOW_DPI}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)*2 + (ny-mcy)*2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of8((mcx, mcy), best)

        owner, ocr_dbg = extract_owner_from_row(bgr, row_y=mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"
                }.get(side,"")
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
            **ocr_dbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi": FAST_DPI if FAST_MODE else SLOW_DPI}}
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
        "FAST_DPI": FAST_DPI,
        "SLOW_DPI": SLOW_DPI,
        "JUNK_2NDLINE": sorted(list(JUNK_2NDLINE)),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
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
        blank = np.zeros((260, 720, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:80]}", (12,140),
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
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o.get("owner","") for o in vdbg.get("rows", []) if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )

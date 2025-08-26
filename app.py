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
app = FastAPI(title="AutoCatastro AI", version="0.5.6")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# Raster DPI configurables
FAST_DPI = int(os.getenv("FAST_DPI", "340"))
SLOW_DPI = int(os.getenv("SLOW_DPI", "500"))

# Refuerzo por lista de nombres comunes (opcional)
NAME_HINTS = (os.getenv("NAME_HINTS", "1").strip() == "1")

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
# Utilidades / diccionarios
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","CSV"
}

# LUGARES / palabras de dirección típicas
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","O","DE","DEL","DA","DO"  # algunas geo/preps
}
ADDR_TOKENS = {"CL","CALLE","AV","AV.","AVDA","AVENIDA","PL","PL.","PLAZA","BLOQUE","ESC","ESC.","ES","PT","PT.","POLIGONO","POLÍGONO","LG","LG."}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# **Mini lista de nombres comunes (refuerzo)**
COMMON_GIVEN_NAMES = {
    # Masculinos frecuentes
    "JOSE","LUIS","ANTONIO","MANUEL","FRANCISCO","JUAN","JAVIER","MIGUEL","CARLOS","DAVID",
    "DANIEL","ALEJANDRO","RAFAEL","PEDRO","ANGEL","ALBERTO","FERNANDO","PABLO","ANDRES","SERGIO",
    "JORGE","RICARDO","ENRIQUE","VICENTE","VICTOR","ADRIAN","ROBERTO","ALVARO","IGNACIO","RAMON",
    # Femeninos frecuentes
    "MARIA","CARMEN","ANA","LAURA","MARTA","PILAR","ROSA","PAULA","SARA","ELENA","DOLORES","ISABEL",
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

# ──────────────────────────────────────────────────────────────────────────────
# Raster (pág. 2) y masks
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = FAST_DPI if FAST_MODE else SLOW_DPI
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

def keep_letters_spaces(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\s\-']", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def clean_name_l1(line: str) -> str:
    if not line: return ""
    U = keep_letters_spaces(line)
    toks = [t for t in U.split() if t and t not in BAD_TOKENS and t not in GEO_TOKENS]
    # Podamos tokens que claramente son de dirección
    toks = [t for t in toks if t not in ADDR_TOKENS]
    # Limitar largo y compactar conectores desplegados al inicio
    out = []
    for t in toks:
        if not out and t in NAME_CONNECTORS:
            continue
        out.append(t)
    name = " ".join(out).strip()
    return name[:48]

def second_line_tokens(line2: str) -> List[str]:
    """Filtra tokens de la 2ª línea. Si NAME_HINTS=1 prioriza nombres comunes."""
    if not line2: return []
    U = keep_letters_spaces(line2)
    toks = [t for t in U.split() if t]
    good = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in BAD_TOKENS or t in ADDR_TOKENS: break
        if t in GEO_TOKENS: break
        if NAME_HINTS and t in COMMON_GIVEN_NAMES:
            good.append(t); continue
        # sin hint: aceptar tokens alfa >=3 (evita "KO","KR")
        if len(t) >= 3:
            good.append(t)
    # Máximo 2 tokens para evitar colar direcciones largas
    return good[:2]

# ──────────────────────────────────────────────────────────────────────────────
# Localizar bandas de texto (columna derecha) y leer 1ª y 2ª línea
# ──────────────────────────────────────────────────────────────────────────────
def read_two_lines_from_band(bgr: np.ndarray, band: Tuple[int,int,int,int]) -> dict:
    """Devuelve {x0,x1,y_line1[],y_line2_hint[], t1_raw, t2_raw, second_used, reason}"""
    x0,y0,x1,y1 = band
    roi = bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.22, fy=1.22, interpolation=cv2.INTER_CUBIC)
    g = enhance_gray(g)
    bw, bwi = binarize(g)

    # Heurística de posiciones: dividimos la banda en 2 franjas horizontales
    H = g.shape[0]
    l1_top  = max(0, int(H*0.48)-18); l1_bot = min(H, l1_top + 80)
    l2_top  = min(H, l1_bot + 8);     l2_bot = min(H, l2_top + 80)

    def txt_of(im, top, bot):
        crop = im[top:bot, :]
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ -'"
        return ocr_text(crop, psm=6, whitelist=wl)

    t1_raw = ""
    t2_raw = ""
    for im in (bw, bwi):
        if not t1_raw:
            t1_raw = txt_of(im, l1_top, l1_bot).strip()
        if not t2_raw:
            t2_raw = txt_of(im, l2_top, l2_bot).strip()

    t1_clean = clean_name_l1(t1_raw)
    t2_toks  = second_line_tokens(t2_raw)
    used2 = False
    reason = ""

    if t2_toks:
        used2 = True
        reason = "name_hint" if NAME_HINTS and any(t in COMMON_GIVEN_NAMES for t in t2_toks) else "tokens2_valid"
        t2_clean = " ".join(t2_toks)
        # Evitar duplicar si ya está incluido al final de L1
        final = (t1_clean + " " + t2_clean).strip()
    else:
        final = t1_clean

    # Truncar prudente
    final = final[:64].strip()
    return {
        "x0": x0, "x1": x1,
        "y_line1": [y0 + int(l1_top/1.22), y0 + int(l1_bot/1.22)],
        "y_line2_hint": [y0 + int(l2_top/1.22), y0 + int(l2_bot/1.22)],
        "t1_raw": t1_clean if t1_clean else t1_raw,
        "t2_raw": " ".join(t2_toks) if t2_toks else "",
        "second_line_used": used2,
        "second_line_reason": reason,
        "text_final": final
    }

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (mini-mapas + columna de texto)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Croquis a la izquierda
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"rows": [], "raster":{"dpi": FAST_DPI if FAST_MODE else SLOW_DPI}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    # Columna de texto (derecha)
    text_x0 = int(w * 0.24)   # un poco más a la izq. para no perder primeras letras
    text_x1 = int(w * 0.60)   # hasta antes de NIF/domicilio

    rows_dbg = []
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # 1) Vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        # 2) Banda de texto alineada a la fila (dos líneas)
        band_h = int(h * 0.17)   # altura cómoda que cubre l1/l2
        by0 = max(0, mcy - band_h//2)
        by1 = min(h, mcy + band_h//2)
        band = (text_x0, by0, text_x1, by1)
        ocrd = read_two_lines_from_band(bgr, band)

        owner = ocrd["text_final"]
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
                cv2.putText(vis, owner[:28], (int(w*0.62), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.62), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            # marco de banda usada
            cv2.rectangle(vis, (band[0], band[1]), (band[2], band[3]), (0,255,255), 2)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocrd
        })

    dbg = {
        "rows": rows_dbg,
        "raster": {"dpi": FAST_DPI if FAST_MODE else SLOW_DPI},
        "hints": {"NAME_HINTS": NAME_HINTS}
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
        "RASTER_DPI": FAST_DPI if FAST_MODE else SLOW_DPI,
        "NAME_HINTS": NAME_HINTS,
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
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )


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

# Refuerzos de nombre (segunda línea) por lista de nombres propios
NAME_HINTS = (os.getenv("NAME_HINTS", "1").strip() == "1")
COMMON_GIVEN_NAMES = {
    "JOSE","LUIS","MARIA","MARIO","JUAN","ANA","ANTONIO","CARLOS","JAVIER",
    "FRANCISCO","MANUEL","PABLO","MARTA","ALBERTO","ALVARO","ENRIQUE","RAFAEL",
    "ROBERTO","SERGIO","FERNANDO","IGNACIO","CRISTINA","LAURA","SARA","ANDRES",
    "JESUS","ANGEL","PEDRO","MIGUEL","NURIA","PAULA","RAQUEL","LORENA","DAVID",
    "DANIEL","ANDREA","ALEJANDRO","GONZALO","ALFONSO","VICENTE","VICTOR","RUBEN"
}
_extra = {t.strip().upper() for t in os.getenv("NAME_HINTS_EXTRA","").split(",") if t.strip()}
if _extra:
    COMMON_GIVEN_NAMES |= _extra

# DPI de raster configurable por ENV (toma prioridad si está presente)
def _default_dpi() -> int:
    return 340 if FAST_MODE else 500
try:
    RASTER_DPI = int(os.getenv("RASTER_DPI", str(_default_dpi())))
    if RASTER_DPI < 250 or RASTER_DPI > 600:
        RASTER_DPI = _default_dpi()
except ValueError:
    RASTER_DPI = _default_dpi()

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
    "GALICIA","[LUGO]","[BARCELONA]","OVIEDO","BILBAO","ZARAGOZA"
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
    dpi = RASTER_DPI
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde (parcela propia)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # Rosa (vecinos)
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
    ang = math.degrees(math.atan2(-(sy), sx))  # Norte arriba
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

# — Limpiezas / parsers —
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
    # compactar conectores múltiples
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

def normalize_second_line(raw: str) -> str:
    if not raw: return ""
    # cortar en primer número o corchete/dos puntos
    raw = re.split(r"[\[\]:0-9]", raw, maxsplit=1)[0]
    raw = raw.strip()
    # quitar caracteres raros
    raw = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\s\-\'\.]", " ", raw.upper())
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    return raw[:26]

# ──────────────────────────────────────────────────────────────────────────────
# Localizar banda “Apellidos…” y extraer línea 1 + posible línea 2
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_bands(bgr: np.ndarray, row_y: int,
                          x_text0: int, x_text1: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Devuelve (band_l1, band_l2) como (x0,y0,x1,y1).
    Usa ventana vertical alrededor de row_y y busca la línea 1 justo bajo la cabecera.
    La línea 2 se infiere inmediatamente debajo de la 1.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.10)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        # fallback conservador: franjas estrechas
        y0 = max(0, row_y - int(h*0.02))
        y1 = min(h, y0 + int(h*0.04))
        y2 = min(h, y1 + int(h*0.04))
        return (x_text0, y0, int(x_text0 + 0.55*(x_text1-x_text0)), y1), (x_text0, y1+2, int(x_text0 + 0.55*(x_text1-x_text0)), y2)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])

        header_bottom = None
        x_nif = None
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:   header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":         x_nif = lx; header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            # coordenadas absolutas
            line_h = int(h * 0.035)
            y1_top = y0s + header_bottom + 6
            y1_bot = min(h, y1_top + line_h)
            y2_top = min(h, y1_bot + 6)
            y2_bot = min(h, y2_top + line_h)

            if x_nif is not None:
                x0 = x_text0
                x1 = min(x_text1, x_text0 + x_nif - 8)
            else:
                x0 = x_text0
                x1 = int(x_text0 + 0.55*(x_text1-x_text0))

            if x1 - x0 > (x_text1 - x_text0) * 0.22:
                return (x0, y1_top, x1, y1_bot), (x0, y2_top, x1, y2_bot)

    # fallback
    line_h = int(h * 0.035)
    y1_top = max(0, row_y - int(h*0.01))
    y1_bot = min(h, y1_top + line_h)
    y2_top = min(h, y1_bot + 6)
    y2_bot = min(h, y2_top + line_h)
    x0 = x_text0
    x1 = int(x_text0 + 0.55*(x_text1-x_text0))
    return (x0, y1_top, x1, y1_bot), (x0, y2_top, x1, y2_bot)

def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Devuelve (owner, dbg) para una fila:
      - OCR de línea 1 (estricto, con limpieza)
      - OCR de línea 2 opcional:
          * si NAME_HINTS y contiene nombres comunes -> añadir
          * si L1 trae salto de línea con un 2º token -> añadir (caso "JOSE\nLUIS")
          * cortar por dígitos / corchetes y limitar a 26 chars
    """
    h, w = bgr.shape[:2]
    # margen amplio a la izquierda para no perder primeras letras
    x_text0 = int(w * 0.30)
    x_text1 = int(w * 0.95)

    (x0a, y1t, x1a, y1b), (x0b, y2t, x1b, y2b) = find_header_and_bands(bgr, row_y, x_text0, x_text1)

    # OCR L1
    roi1 = bgr[y1t:y1b, x0a:x1a]
    txt1_raw, t1_extra_from_break = "", ""
    if roi1.size > 0:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g1 = enhance_gray(g1)
        bw1, bwi1 = binarize(g1)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        v1 = [ocr_text(bw1, 6, WL), ocr_text(bwi1, 6, WL), ocr_text(bw1, 7, WL), ocr_text(bwi1, 7, WL)]
        # elegir mejor
        for t in v1:
            if not t: continue
            # si hay salto en la misma caja (p.ej. "JOSE\nLUIS"), separa
            parts = [p.strip() for p in t.split("\n") if p.strip()]
            if len(parts) >= 2:
                txt1_raw = parts[0]
                # segunda parte viene de L1 (alta confianza)
                t1_extra_from_break = parts[1]
                break
            txt1_raw = t.strip()
            if txt1_raw: break

    owner = pick_owner_from_text(txt1_raw)

    # OCR L2
    txt2_raw = ""
    roi2 = bgr[y2t:y2b, x0b:x1b]
    if roi2.size > 0:
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g2 = enhance_gray(g2)
        bw2, bwi2 = binarize(g2)
        WL2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        v2 = [ocr_text(bw2, 6, WL2), ocr_text(bwi2, 6, WL2), ocr_text(bw2, 7, WL2), ocr_text(bwi2, 7, WL2)]
        for t in v2:
            if t:
                txt2_raw = t.strip()
                break

    # decisión sobre segunda línea
    used_second = False
    second_reason = ""

    # 1) preferente: si L1 traía un salto (caso "JOSE\nLUIS")
    if t1_extra_from_break:
        s2 = normalize_second_line(t1_extra_from_break)
        if s2 and s2 not in {"KO","KR"}:
            owner = (owner + " " + s2).strip()
            used_second = True
        second_reason = "from_l1_break"

    # 2) si no, probar L2 según hints / heurística
    if not used_second and txt2_raw:
        tokens2 = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", txt2_raw.upper()) if t]
        has_name_hint = NAME_HINTS and any(t in COMMON_GIVEN_NAMES for t in tokens2)
        # Aceptar si: hay hint de nombre, o hay ≥2 tokens razonables.
        if has_name_hint or len(tokens2) >= 2:
            s2 = normalize_second_line(txt2_raw)
            # filtros suaves: descartar basuras cortas conocidas
            if s2 and s2 not in {"KO","KR","ECE","EO"}:
                owner = (owner + " " + s2).strip()
                used_second = True
                second_reason = "hint_or_2tokens"

    dbg = {
        "band": [x0a, y1t, x1a, y2b],
        "y_line1": [y1t, y1b],
        "y_line2_hint": [y2t, y2b],
        "x0": x0a, "x1": x1a,
        "t1_raw": txt1_raw,
        "t1_extra_raw": t1_extra_from_break,
        "t2_raw": txt2_raw,
        "second_line_used": used_second,
        "second_line_reason": second_reason or "",
    }
    return owner, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (detecta N/S/E/O con centroides y extrae titulares por fila)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte del área de croquis (izquierda) donde están los puntos
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"rows": [], "raster":{"dpi":RASTER_DPI}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        owner, ocrdbg = extract_owner_from_row(bgr, row_y=mcy)

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
            "ocr": ocrdbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi":RASTER_DPI}}
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
        "raster_dpi": RASTER_DPI,
        "name_hints": NAME_HINTS,
        "name_hints_extra": sorted(list(_extra)) if _extra else []
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



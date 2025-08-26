from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import numpy as np
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.5.0")

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
# Utilidades comunes
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","DIRECCIÓN","DIRECCION","COORDENADAS"
}
ADDR_TOKENS = {"CL","CALLE","AV","AVDA","AVE","LG","LUGAR","PT","PL","PZ",
               "CARRILET","BLOQUE","ES","ESC","NUM","Nº","NO","HUSO","ETRS"}

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
    dpi = 400 if FAST_MODE else 500
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

def clean_owner_line(line: str) -> str:
    if not line: return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in BAD_TOKENS or t in ADDR_TOKENS: continue
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
# Leer texto real de la pág. 2 y preparar líneas
# ──────────────────────────────────────────────────────────────────────────────
def page2_lines(pdf_bytes: bytes) -> List[str]:
    out: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) >= 2:
            t = (pdf.pages[1].extract_text(x_tolerance=2, y_tolerance=2) or "")
            t = t.replace("\r", "\n")
            t = re.sub(r"[ \t]+", " ", t)
            t = re.sub(r"\n{2,}", "\n", t)
            out = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return out

def first_alpha_prefix(line: str, max_chars: int = 26) -> str:
    """
    Devuelve el prefijo alfabético continuo (A–Z/Ñ/espacios/guiones/apóstrofe)
    antes de que aparezca un dígito, ':', '[' o un token típico de dirección.
    Se recorta a max_chars.
    """
    if not line: return ""
    # Cortar duro por dígito o corchete/':'
    m = re.split(r"[:\[\]0-9]", line, maxsplit=1)
    candidate = m[0] if m else line
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", candidate.upper()) if t]
    out = []
    for t in toks:
        if t in ADDR_TOKENS or t in BAD_TOKENS:
            break
        if not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ'-]+", t):
            break
        out.append(t)
        # no cojas más de 2–3 tokens por seguridad
        if len(out) >= 3:
            break
    name = " ".join(out).strip()
    return name[:max_chars]

def augment_from_p2(base_name: str, p2_lines_list: List[str]) -> str:
    """
    Busca base_name en p2_lines; si lo encuentra, usa la línea siguiente
    para extraer un prefijo alfabético (p.ej. 'LUIS') y concatenarlo.
    """
    if not base_name: return base_name
    BN = base_name.upper()
    for i, ln in enumerate(p2_lines_list):
        if BN in ln.upper():
            if i + 1 < len(p2_lines_list):
                nxt = p2_lines_list[i+1]
                ext = first_alpha_prefix(nxt, max_chars=26)
                if ext:
                    return (base_name + " " + ext).strip()
            break
    return base_name

# ──────────────────────────────────────────────────────────────────────────────
# Localizar cabecera y bandas (línea 1 + línea 2 OCR, pero 2ª se sobreescribe por PDF si hay match)
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_bands(bgr: np.ndarray, row_y: int,
                                x_text0: int, x_text1: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    line_h = int(h * 0.035)
    if band.size == 0:
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + line_h)
        roi1 = (x_text0, y0, int(x_text0 + 0.55*(x_text1-x_text0)), y1)
        roi2 = (roi1[0], min(h, roi1[3]+2), roi1[2], min(h, roi1[3]+2+line_h))
        return roi1, roi2

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

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
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)
        if header_bottom is not None:
            break

    if header_bottom is None:
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + line_h)
        roi1 = (x_text0, y0, int(x_text0 + 0.55*(x_text1-x_text0)), y1)
        roi2 = (roi1[0], min(h, roi1[3]+2), roi1[2], min(h, roi1[3]+2+line_h))
        return roi1, roi2

    y1_line1 = y0s + header_bottom + 6
    roi1_y0  = y1_line1
    roi1_y1  = min(h, roi1_y0 + line_h)

    if x_nif is not None:
        x0 = x_text0
        x1 = min(x_text1, x_text0 + x_nif - 8)
    else:
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))

    roi1 = (x0, roi1_y0, x1, roi1_y1)
    gap = 2
    roi2_y0 = min(h, roi1_y1 + gap)
    roi2_y1 = min(h, roi2_y0 + line_h)
    roi2 = (x0, roi2_y0, x1, roi2_y1)
    return roi1, roi2

def read_owner_two_lines_OCR(bgr: np.ndarray, roi1: Tuple[int,int,int,int], roi2: Tuple[int,int,int,int]) -> Tuple[str, str]:
    # 1ª línea con filtro (nombre)
    def read_roi_variants(roi: Tuple[int,int,int,int]) -> List[str]:
        x0,y0,x1,y1 = roi
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        return [
            ocr_text(bw,  psm=6,  whitelist=WL),
            ocr_text(bwi, psm=6,  whitelist=WL),
            ocr_text(bw,  psm=7,  whitelist=WL),
            ocr_text(bwi, psm=7,  whitelist=WL),
            ocr_text(bw,  psm=13, whitelist=WL),
        ]

    v1 = read_roi_variants(roi1)
    t1 = ""
    for cand in v1:
        t1 = pick_owner_from_text(cand)
        if t1:
            break

    # 2ª línea por OCR (solo para debug; no se usa si hay refuerzo PDF)
    v2 = [s.strip() for s in read_roi_variants(roi2) if s and s.strip()]
    t2_raw = max(v2, key=len).strip() if v2 else ""
    t2_raw = re.sub(r"\s+", " ", t2_raw).strip()
    t2_raw = t2_raw[:26] if t2_raw else ""

    owner = (t1 + (" " + t2_raw if t2_raw else "")).strip() if t1 else t2_raw
    return owner, t2_raw

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (N/S/E/O + 1ª línea OCR + refuerzo 2ª línea desde PDF)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            p2_lines_list: List[str],
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Marco croquis a la izquierda
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"rows": []}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano → lado
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        # columnas de texto
        x_text0 = int(w * 0.30)
        x_text1 = int(w * 0.96)
        roi1, roi2 = find_header_and_owner_bands(bgr, mcy, x_text0, x_text1)

        # OCR (1ª línea + 2ª solo para debug)
        owner_ocr, t2_raw = read_owner_two_lines_OCR(bgr, roi1, roi2)

        # Refuerzo con texto PDF (buscar la línea siguiente a la coincidencia del nombre de 1ª línea)
        owner_final = owner_ocr
        first_line = owner_ocr.split()  # podría incluir ya algo en 2ª por OCR, no importa
        if first_line:
            base = " ".join(first_line[:min(len(first_line), 6)])  # base_name razonable
            base_upper = base.upper()
            # si detectamos claramente una 1ª línea (>= 2 tokens), usarla para buscar en PDF
            if len(base_upper.split()) >= 2:
                owner_final = augment_from_p2(base_upper, p2_lines_list)

        if side and owner_final and side not in used_sides:
            linderos[side] = owner_final
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {"norte":"N","sur":"S","este":"E","oeste":"O"}.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner_final:
            cv2.putText(vis, owner_final[:28], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner_final[:28], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "txt1": " ".join(owner_ocr.split()[:6]) if owner_ocr else "",
            "txt2_raw": t2_raw,
            "aug_from_p2": owner_final if owner_final != owner_ocr else "",
            "roi1": list(roi1),
            "roi2": list(roi2),
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
        p2_lines_list = page2_lines(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, p2_lines_list, annotate=bool(labels), annotate_names=bool(names)
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
        p2_lines_list = page2_lines(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, p2_lines_list, annotate=False)
        owners_detected = [l for l in linderos.values() if l]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        if debug:
            dbg["p2_text_sample"] = p2_lines_list[:28]
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )



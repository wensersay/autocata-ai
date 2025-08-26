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

# Segunda línea (cont. del nombre)
SECOND_LINE_FORCE      = (os.getenv("SECOND_LINE_FORCE", "0").strip() == "1")
SECOND_LINE_MAXTOKENS  = int(os.getenv("SECOND_LINE_MAXTOKENS", "2").strip() or "2")
SECOND_LINE_MAXCHARS   = int(os.getenv("SECOND_LINE_MAXCHARS",  "26").strip() or "26")
SHORT_NAME_WHITELIST   = [t.strip().upper() for t in os.getenv(
    "SHORT_NAME_WHITELIST",
    "LUIS,JOSE,JUAN,MARIA,ANA,ANTONIO,MANUEL,PEDRO,CARLOS,FRANCISCO,RAFAEL,ROSA,LAURA"
).split(",") if t.strip()]

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
    "CATASTRAL","TITULARIDAD","PRINCIPAL","RELACION","RELACIÓN","DE","PARCELAS"
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
    dpi = 400 if FAST_MODE else 520
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
        if len(name) >= 6:
            return name
    return ""

def clean_second_line(raw: str) -> str:
    """
    Permisiva: corta en el primer dígito/ '[' / ':' , limita tokens y caracteres.
    """
    if not raw: return ""
    s = raw.upper().replace("\n", " ")
    s = re.split(r"[\[\]:0-9]", s)[0]  # cortar en meta/num
    s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ' \-]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s: return ""
    toks = [t for t in s.split() if t]
    # Si hay un nombre claro de whitelist, úsalo tal cual (p.ej. LUIS)
    for t in toks:
        if t in SHORT_NAME_WHITELIST:
            return t[:SECOND_LINE_MAXCHARS]
    # En modo forzado aceptamos hasta N tokens “limpios”
    toks = toks[:max(1, SECOND_LINE_MAXTOKENS)]
    s2 = " ".join(toks)[:SECOND_LINE_MAXCHARS].strip()
    return s2

# ──────────────────────────────────────────────────────────────────────────────
# Localizar cabecera y banda de la 1ª línea del nombre
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_band(bgr: np.ndarray, row_y: int,
                               x_text0: int, x_text1: int) -> Tuple[int,int,int,int,dict]:
    """
    Devuelve (x0, x1, y0, y1, dbg) para la banda de la 1ª línea del titular.
    Busca 'APELLIDOS' y opcionalmente 'NIF' y sitúa y0 justo debajo.
    Alto ≈ 3.7% de la página. Si no encuentra, usa fallback con fila.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        # fallback
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + int(h*0.037))
        return x_text0, int(x_text0 + 0.55*(x_text1-x_text0)), y0, y1, {"header_found": False, "x_nif_found": False}

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    header_found = False
    x_nif_found  = False
    x_nif_local  = None
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
                header_found = True
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif_found = True
                x_nif_local = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            break

    if header_bottom is None:
        # fallback si no hay cabecera reconocible
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + int(h*0.037))
        return x_text0, int(x_text0 + 0.55*(x_text1-x_text0)), y0, y1, {"header_found": False, "x_nif_found": False}

    # Coordenadas absolutas
    y0 = y0s + header_bottom + 6
    y1 = min(h, y0 + int(h * 0.037))
    if x_nif_found and x_nif_local is not None:
        x0 = x_text0
        x1 = min(x_text1, x_text0 + x_nif_local - 8)
    else:
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))

    if x1 - x0 < max(60, int(w*0.1)):
        # seguridad: ancho mínimo razonable
        x1 = min(x_text1, x0 + max(60, int(w*0.18)))

    dbg = {"header_found": header_found, "x_nif_found": x_nif_found}
    return x0, x1, y0, y1, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Extraer dueño por fila (con 2ª línea robusta)
# ──────────────────────────────────────────────────────────────────────────────
def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Devuelve (owner_full, dbg)
      - Primera línea: OCR con whitelist + filtrado de ruido
      - Segunda línea: barrido vertical + multi-PSM y limpieza permisiva
    """
    h, w = bgr.shape[:2]
    # Columna de texto: arrancamos un poco antes para evitar “mordidas”
    x_text0_base = int(w * 0.31)
    x_text1 = int(w * 0.96)

    # Encontrar cabecera
    x0, x1, y1_0, y1_1, hdr_dbg = find_header_and_owner_band(bgr, row_y, x_text0_base, x_text1)
    band_dbg = {}

    # Primera línea (L1)
    roi1 = bgr[y1_0:y1_1, x0:x1]
    name1 = ""
    t1_raw = ""
    if roi1.size != 0:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g1 = enhance_gray(g1)
        bw1, bwi1 = binarize(g1)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        variants = [
            ocr_text(bw1,  psm=6,  whitelist=WL),
            ocr_text(bwi1, psm=6,  whitelist=WL),
            ocr_text(bw1,  psm=7,  whitelist=WL),
            ocr_text(bwi1, psm=7,  whitelist=WL),
            ocr_text(bw1,  psm=13, whitelist=WL),
        ]
        # Elegir la mejor por “score” (letras A–Z)
        best = max(variants, key=lambda t: sum(ch.isalpha() for ch in t))
        t1_raw = best or ""
        name1 = pick_owner_from_text(best)

    # Segunda línea (L2): bajo L1
    line_h = (y1_1 - y1_0)
    y2_0 = min(h, y1_1 + 6)
    y2_1 = min(h, y2_0 + max(14, line_h))
    t2_raw = ""
    used_second = False
    second_reason = ""

    if SECOND_LINE_FORCE and x1 - x0 > 10:
        # Barrido vertical + multi-PSM (robusto ante desalineación)
        offsets = [0, +int(0.35*line_h), -int(0.25*line_h), +int(0.6*line_h)]
        best_txt = ""
        best_score = -1

        for dy in offsets:
            yy0 = max(0, y2_0 + dy)
            yy1 = min(bgr.shape[0], y2_1 + dy)
            roi2 = bgr[yy0:yy1, x0:x1]
            if roi2.size == 0:
                continue
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)

            for img in (bwi2, bw2):
                for psm in (7, 6, 13):
                    t = ocr_text(img, psm=psm, whitelist=None)
                    score = sum(ch.isalpha() for ch in t)
                    if score > best_score:
                        best_txt = t
                        best_score = score
            # Salida temprana si capturamos un nombre corto (whitelist)
            if any(tok in best_txt.upper().split() for tok in SHORT_NAME_WHITELIST):
                break

        t2_raw = (best_txt or "").strip()
        cand2 = clean_second_line(t2_raw)
        if cand2:
            used_second = True
            second_reason = "forced"
            # Si la 1ª línea es corta y 2ª es clara (p.ej. 'LUIS'), unimos
            if name1:
                name1 = (name1 + " " + cand2).strip()
            else:
                name1 = cand2

    # Componer salida
    owner_full = (name1 or "").strip()
    # Poda general por seguridad
    owner_full = owner_full[:max(26, SECOND_LINE_MAXCHARS + 24)].strip()

    dbg = {
        "band": [x0, y1_0, x1, y2_1],
        "y_line1": [y1_0, y1_1],
        "y_line2": [y2_0, y2_1],
        "x0": x0, "x1": x1,
        "t1_raw": t1_raw,
        "t2_raw": t2_raw,
        "second_line_used": used_second,
        "second_line_reason": second_reason,
        "header_found": hdr_dbg.get("header_found", False),
        "x_nif_found": hdr_dbg.get("x_nif_found", False)
    }
    return owner_full, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Zona donde suelen estar los mini-mapas (márgenes conservadores)
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
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        owner, ocr_dbg = extract_owner_from_row(bgr, row_y=mcy)

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
                cv2.putText(vis, owner[:30], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:30], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
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
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)},
        "second_line": {
            "force": SECOND_LINE_FORCE,
            "max_tokens": SECOND_LINE_MAXTOKENS,
            "max_chars": SECOND_LINE_MAXCHARS,
            "short_whitelist": SHORT_NAME_WHITELIST[:8]
        }
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



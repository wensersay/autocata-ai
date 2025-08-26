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
app = FastAPI(title="AutoCatastro AI", version="0.4.9")

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
    "CATASTRAL","TITULARIDAD","PRINCIPAL","CSV"
}

# Ruido geográfico y etiquetas frecuentes que NO son parte del nombre
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","OS","O","O."
}

# Prefijos típicos de direcciones/etiquetas que cortan la segunda línea
ADDR_PREFIX = {"CL","C/","CALLE","AV","AV.","AVDA","AVENIDA","LG","LUGAR","CM","CM.","CMNO","CAMINO","PZ","PL","PLAZA","ES:C","ES:","BLOQUE","ESC","ESC.","BL","BL."}

# Conectores válidos dentro de un nombre
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Tokens de ruido OCR muy comunes en 2ª línea (descartar)
NOISE_TOKENS = {"SSS","SE","ST","KO","CS","CO","CSV","OS","O"}

# Nombres propios muy comunes (para decidir si la 2ª línea es útil)
COMMON_GIVEN = {
    "JOSE","MARIA","JESUS","ANGEL","MIGUEL","LUIS","JUAN","ANTONIO","FRANCISCO",
    "CARLOS","MANUEL","JOSEFA","TERESA","PILAR","DOLORES","ROSA","ANA"
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
    dpi = 400 if FAST_MODE else 500
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]  # RGB→BGR

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde agua (parcela propia)
    g_ranges = [
        (np.array([35, 20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86, 15, 50], np.uint8), np.array([100,255,255], np.uint8)),
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

def sanitize_tokens(line: str) -> List[str]:
    # Divide en tokens alfanuméricos/acentuados y filtra basura
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", (line or "").upper()) if t]
    clean = []
    for t in toks:
        if t in NOISE_TOKENS:     continue
        if any(ch.isdigit() for ch in t): break
        if t in BAD_TOKENS:       continue
        if t in GEO_TOKENS:       break
        if len(t) == 1 and t not in NAME_CONNECTORS: continue
        # descarta repeticiones tipo "SSS"
        if re.match(r"^(.)\1{2,}$", t): continue
        clean.append(t)
    return clean

def compact_name(tokens: List[str], max_tokens: int = 6, max_chars: int = 48) -> str:
    out = []
    for t in tokens:
        if (not out) and t in NAME_CONNECTORS:
            continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= max_tokens:
            break
    name = " ".join(out).strip()
    return name[:max_chars]

def pick_owner_from_text(txt: str) -> str:
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):            continue
        name = compact_name(sanitize_tokens(U))
        if len(name) >= 6:
            return name
    return ""

def merge_first_second_line(first: str, second: str) -> str:
    """Funde 2ª línea solo si parece continuación de NOMBRE (evita ruidos tipo SSS/SE/ST/KO/CS/CO)."""
    if not first: return ""
    if not second: return first

    t2 = sanitize_tokens(second)
    if not t2:
        return first
    # descarta claro inicio de dirección
    if t2[0] in ADDR_PREFIX or t2[0].endswith(":"):
        return first

    # Solo 1–2 tokens alfabéticos válidos
    take = []
    for t in t2:
        if t in NOISE_TOKENS:                 continue
        if len(t) <= 2 and t not in NAME_CONNECTORS:
            continue
        if any(ch.isdigit() for ch in t):     break
        # evita repeticiones AAA/SSS
        if re.match(r"^(.)\1{2,}$", t):       continue
        take.append(t)
        if len(take) >= 2:
            break

    if not take:
        return first

    # Señal de continuación válida: primer token es nombre común o palabra ≥4
    if (take[0] in COMMON_GIVEN) or (len(take[0]) >= 4):
        merged = (first + " " + " ".join(take)).strip()
        return merged[:54]
    return first

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna y 2 líneas de titular
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_band(bgr: np.ndarray, row_y: int,
                               x_text0: int, x_text1: int) -> Tuple[int,int,int,int]:
    """
    Devuelve (x0, x1, y0, y1) para la banda de la 1ª línea del titular.
    Busca 'APELLIDOS' (o 'NIF') y coloca y0 justo debajo.
    Alto aprox de una línea (~3.5% de alto de página).
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + int(h*0.035))
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
            y1 = min(h, y0 + int(h * 0.035))  # 1 línea
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
    y1 = min(h, y0 + int(h*0.035))
    x0 = x_text0
    x1 = int(x_text0 + 0.55*(x_text1-x_text0))
    return x0, x1, y0, y1

def ocr_line(bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0: return ""
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
    g = enhance_gray(g)
    bw, bwi = binarize(g)
    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
    variants = [
        ocr_text(bw,  psm=6,  whitelist=WL),
        ocr_text(bwi, psm=6,  whitelist=WL),
        ocr_text(bw,  psm=7,  whitelist=WL),
        ocr_text(bwi, psm=7,  whitelist=WL),
        ocr_text(bw,  psm=13, whitelist=WL),
    ]
    best = ""
    for v in variants:
        name = pick_owner_from_text(v)
        if name:
            return name
        if len(v) > len(best):
            best = v
    return compact_name(sanitize_tokens(best))

def extract_owner_two_lines(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Extrae 1ª y 2ª línea del titular y las fusiona si la 2ª parece continuación.
    Devuelve (owner_final, dbg_dict)
    """
    h, w = bgr.shape[:2]
    x_text0_base = int(w * 0.33)   # algo más a la izq. para no “morder” iniciales
    x_text1      = int(w * 0.96)

    txt1 = txt2 = ""
    roi1 = roi2 = (0,0,0,0)
    used_attempt = 0

    for attempt in range(2):
        extra_left = attempt * max(18, int(w * 0.03))
        x0_base = max(0, x_text0_base - extra_left)
        x0, x1, y0, y1 = find_header_and_owner_band(bgr, row_y, x0_base, x_text1)

        t1 = ocr_line(bgr, x0, y0, x1, y1)

        # si la 1ª queda como una sola letra → reintentar con más apertura a la izq
        if attempt == 0 and re.match(r"^[A-ZÁÉÍÓÚÜÑ]\b", t1 or ""):
            continue

        # 2ª línea del mismo alto, inmediatamente debajo
        line_h = max(10, y1 - y0)
        y0b = min(h-1, y1 + 2)
        y1b = min(h, y0b + line_h)
        t2  = ocr_line(bgr, x0, y0b, x1, y1b)

        txt1, txt2 = t1, t2
        roi1, roi2 = (x0, y0, x1, y1), (x0, y0b, x1, y1b)
        used_attempt = attempt
        break

    owner1 = txt1
    owner2 = txt2

    # Fusión con reglas estrictas para evitar ruido (SSS/SE/ST/KO/CS/CO, etc.)
    owner_final = merge_first_second_line(owner1, owner2)

    dbg = {
        "attempt": used_attempt,
        "roi1": list(roi1), "roi2": list(roi2),
        "txt1": owner1, "txt2": owner2,
        "owner_final": owner_final
    }
    return owner_final, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte de franja izquierda donde están los mini-mapas
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

        owner_final, dbg_owner = extract_owner_two_lines(bgr, row_y=mcy)

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
            # ROIs de las dos líneas
            x0,y0,x1,y1 = dbg_owner["roi1"]; cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)
            x0,y0,x1,y1 = dbg_owner["roi2"]; cv2.rectangle(vis, (x0,y0), (x1,y1), (255,200,0), 2)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "txt1": dbg_owner["txt1"],
            "txt2": dbg_owner["txt2"],
            "owner": dbg_owner["owner_final"],
            "roi_attempt": dbg_owner["attempt"],
            "roi1": dbg_owner["roi1"],
            "roi2": dbg_owner["roi2"],
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
    labels: int = Query(1, description="1=mostrar N/S/E/O"),
    names:  int = Query(0, description="1=mostrar nombre estimado")
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
                 names:  int = Query(0)):
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
        owners_detected = [o["owner"] for o in vdbg.get("rows", []) if o.get("owner")]
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


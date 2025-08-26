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

# Segunda línea de nombre (configurable por entorno)
SECOND_LINE_FORCE      = (os.getenv("SECOND_LINE_FORCE", "0").strip() == "1")
SECOND_LINE_MAXCHARS   = int(os.getenv("SECOND_LINE_MAXCHARS", "26").strip() or "26")
SECOND_LINE_MAXTOKENS  = int(os.getenv("SECOND_LINE_MAXTOKENS", "2").strip() or "2")

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

# Prefijos típicos de dirección en la segunda línea y ruidos frecuentes 2ch
ADDRESS_PREFIX = {"CL","C","C/","CALLE","AV","AVDA","LG","PT","PL","ES","ESC","BLOQUE","BLQ","URB","PO","POL","KM"}
NOISE_2CH     = {"KO","KR","CS","ST","EM"}

# Whitelist de nombres cortos frecuentes en L2 (p. ej., "LUIS")
SHORT_NAME_WHITELIST = {
    "LUIS","MARIA","JESUS","ANA","IVAN","JUAN","JOSE","NOEL","NORA","RAUL","IÑAKI",
    "PAULA","DAVID","MARTA","PABLO","MARIO","SARA","RUBEN","RUBÉN","ANTONIO","MANUEL",
    "MIGUEL","JAVIER","FRAN","NURIA","ALVARO","ÁLVARO","ROCIO","ROCÍO"
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

def pick_owner_from_text(txt: str) -> str:
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):            continue
        return U[:48]
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Detección de cabecera y bandas de la 1ª/2ª línea
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_bands(bgr: np.ndarray, row_y: int,
                          x_text0: int, x_text1: int) -> Tuple[bool, bool, int, int, Tuple[int,int], Tuple[int,int]]:
    """
    Busca 'APELLIDOS/NOMBRE/RAZON SOCIAL' y 'NIF' cerca de row_y.
    Devuelve (header_found, x_nif_found, x0, x1, (y1_0,y1_1), (y2_0,y2_1))
    para recortar 1ª y 2ª línea del titular.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.08)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        # fallback: franjas aproximadas
        lh = int(h * 0.035)
        y1_0 = max(0, row_y - int(lh*1.0)); y1_1 = min(h, y1_0 + lh)
        y2_0 = min(h, y1_1 + 8);            y2_1 = min(h, y2_0 + lh)
        return False, False, x_text0, int(x_text0 + 0.55*(x_text1-x_text0)), (y1_0,y1_1), (y2_0,y2_1)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    found_header = False
    found_xnif   = False
    x_rel_nif = None
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
            if ("APELLIDOS" in T) or ("NOMBRE" in T) or ("RAZON" in T) or ("RAZÓN" in T) or ("SOCIAL" in T):
                found_header = True
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                found_xnif = True
                x_rel_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            break

    # Calcular bandas y x0/x1
    if header_bottom is None:
        # fallback sin cabecera fiable
        lh = int(h * 0.035)
        y1_0 = max(0, row_y - int(lh*1.0)); y1_1 = min(h, y1_0 + lh)
        y2_0 = min(h, y1_1 + 8);            y2_1 = min(h, y2_0 + lh)
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))
        return False, False, x0, x1, (y1_0, y1_1), (y2_0, y2_1)

    y1_0 = y0s + header_bottom + 6
    y1_1 = min(h, y1_0 + int(h * 0.035))
    y2_0 = min(h, y1_1 + 6)
    y2_1 = min(h, y2_0 + int(h * 0.035))

    if found_xnif and x_rel_nif is not None:
        x0 = x_text0
        x1 = min(x_text1, x_text0 + x_rel_nif - 8)
    else:
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))

    return bool(found_header), bool(found_xnif), x0, x1, (y1_0, y1_1), (y2_0, y2_1)

# ──────────────────────────────────────────────────────────────────────────────
# Combinar 1ª + 2ª línea con filtros anti-ruido (MEJORADO)
# ──────────────────────────────────────────────────────────────────────────────
def combine_first_second(first: str, second_raw: str, forced: bool) -> Tuple[str, bool, str]:
    """
    Devuelve (owner_final, used_second, reason)
    Reglas anti-ruido para L2:
      - corta en el primer dígito o [, :, (, /.
      - descarta tokens de 1–2 letras y NOISE_2CH (KO/KR/CS/ST/EM).
      - bloquea prefijos de dirección (CL, AV, LG, ...).
      - si forced=True: aceptar sólo si (1) un token whitelist (LUIS, MARIA, ...) o
        (2) primer token ≥3 letras y no geográfico/ruido/dirección.
    """
    if not second_raw:
        return first, False, "empty"

    # Normalizar y recortar por separadores
    t2 = second_raw.upper().strip().replace("\n", " ")
    m = re.search(r"[\d\[\:\(\/]", t2)
    if m:
        t2 = t2[:m.start()].strip()
    if not t2:
        return first, False, "cut_to_empty"

    # Limpiar
    t2 = re.sub(r"[^\wÁÉÍÓÚÜÑ\'\- ]+", " ", t2)
    t2 = re.sub(r"\s{2,}", " ", t2).strip()
    toks_all = [t for t in t2.split(" ") if t]

    # Si empieza por prefijo de dirección, descartar
    if toks_all and toks_all[0] in ADDRESS_PREFIX:
        return first, False, "addr_prefix"

    # Filtrado de tokens válidos
    valid = []
    for t in toks_all:
        if t in NOISE_2CH:
            continue
        if len(t) <= 2:
            continue
        if t in GEO_TOKENS:
            break
        if any(ch.isdigit() for ch in t):
            break
        valid.append(t)

    if not valid:
        # Si forced y un solo token whitelist corto (p.ej., LUIS), rescatarlo
        if forced and len(toks_all) == 1 and toks_all[0] in SHORT_NAME_WHITELIST:
            cand2 = toks_all[0]
        else:
            return first, False, "no_valid_tokens"
    else:
        # Modo forced: aceptar el primer token si ya pasó los filtros
        if forced and not (len(valid) == 1 and valid[0] in SHORT_NAME_WHITELIST):
            head = valid[0]
            cand2 = head
        else:
            # Modo normal: hasta MAXTOKENS (sin contar conectores)
            sel = []
            for t in valid:
                sel.append(t)
                if len([x for x in sel if x not in NAME_CONNECTORS]) >= max(1, SECOND_LINE_MAXTOKENS):
                    break
            cand2 = " ".join(sel)

    cand2 = cand2[:max(8, SECOND_LINE_MAXCHARS)].strip()
    if not cand2:
        return first, False, "too_short_after_cut"

    # Validación suave en modo normal
    if not forced and not UPPER_NAME_RE.match(cand2):
        return first, False, "regex_fail"

    final = f"{first} {cand2}".strip()
    return final, True, ("forced" if forced else "accepted")

# ──────────────────────────────────────────────────────────────────────────────
# Extracción por filas (detecta N/S/E/O y titula)
# ──────────────────────────────────────────────────────────────────────────────
def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Extrae 1ª y 2ª línea (opcional) de la columna de titular.
    Devuelve (owner_final, dbg)
    """
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.31)   # margen izq. texto
    x_text1 = int(w * 0.96)   # margen der. texto

    header_found, xnif_found, x0, x1, (y1_0,y1_1), (y2_0,y2_1) = find_header_and_bands(
        bgr, row_y, x_text0, x_text1
    )

    # OCR primera línea
    roi1 = bgr[y1_0:y1_1, x0:x1]
    txt1_raw = ""
    if roi1.size != 0:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g1 = enhance_gray(g1)
        bw1, bwi1 = binarize(g1)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        for img in (bw1, bwi1):
            txt1_raw = ocr_text(img, psm=7, whitelist=WL)
            if txt1_raw: break
    owner_first = pick_owner_from_text(txt1_raw)

    # OCR segunda línea (más permisivo)
    txt2_raw = ""
    if roi1.size != 0:
        roi2 = bgr[y2_0:y2_1, x0:x1]
        if roi2.size != 0:
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)
            for img in (bw2, bwi2):
                # aquí NO imponemos regex de nombre; lo hará combine_first_second
                txt2_raw = ocr_text(img, psm=7, whitelist=None)
                if txt2_raw: break

    # Combinar
    forced = bool(SECOND_LINE_FORCE)
    final_owner = owner_first
    used_second = False
    reason2 = "skipped"
    if owner_first:
        final_owner, used_second, reason2 = combine_first_second(owner_first, txt2_raw, forced)

    dbg = {
        "band": [x0, y1_0 - (y1_0 - (y2_0 - (y1_1 - y1_0))) , x1, y2_1],  # aproximado para contexto
        "y_line1": [y1_0, y1_1],
        "y_line2": [y2_0, y2_1],
        "x0": x0, "x1": x1,
        "t1_raw": owner_first or (txt1_raw or ""),
        "t2_raw": txt2_raw or "",
        "second_line_used": used_second,
        "second_line_reason": reason2,
        "header_found": header_found,
        "x_nif_found": xnif_found
    }
    return final_owner or "", dbg

def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte para localizar croquis (zona izquierda)
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
        # Vecino más cercano (para saber N/S/E/O)
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        # Extraer nombre (1ª + 2ª línea)
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
        "SECOND_LINE_FORCE": SECOND_LINE_FORCE,
        "SECOND_LINE_MAXCHARS": SECOND_LINE_MAXCHARS,
        "SECOND_LINE_MAXTOKENS": SECOND_LINE_MAXTOKENS,
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


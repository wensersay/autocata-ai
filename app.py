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
app = FastAPI(title="AutoCatastro AI", version="0.6.2")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# DPI: FAST_DPI (si existe) tiene prioridad sobre PDF_DPI. Defaults sensatos.
def resolve_dpi() -> int:
    fast_dpi = os.getenv("FAST_DPI")
    pdf_dpi  = os.getenv("PDF_DPI")
    if fast_dpi:
        try:
            return max(240, min(600, int(fast_dpi)))
        except:
            pass
    if pdf_dpi:
        try:
            return max(240, min(600, int(pdf_dpi)))
        except:
            pass
    return 340  # por defecto

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
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","APELLIDO","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","TITULAR","CSV","DIRECCIÓN","DIRECCION"
}
HEADER_TOKENS = {"APELLIDOS","NOMBRE","RAZON","RAZÓN","SOCIAL","NIF","DOMICILIO","TITULARIDAD","PRINCIPAL","TITULAR"}

# geotokens / ruido que no deben entrar en el nombre
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# 2ª línea con ruido configurable (coma-separado). Por defecto: Z,VA,EO,SS,KO,KR
JUNK_2NDLINE = [t.strip().upper() for t in os.getenv("JUNK_2NDLINE","Z,VA,EO,SS,KO,KR").split(",") if t.strip()]

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
    dpi = resolve_dpi()
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

def angle_to_compass(cx: int, cy: int, x: int, y: int) -> str:
    # 8 rumbos: N, NE, E, SE, S, SW, W, NW
    ang = math.degrees(math.atan2((cy - y), (x - cx)))  # 0° = E
    # Convertir a 0..360
    ang = (ang + 360.0) % 360.0
    labels = ["este","sureste","sur","suroeste","oeste","noroeste","norte","noreste"]
    # cada sector = 45°, offset 22.5°
    idx = int(((ang + 22.5) % 360.0) // 45.0)
    return labels[idx]

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

def is_header_line(txt: str) -> bool:
    U = (txt or "").upper()
    return any(tok in U for tok in HEADER_TOKENS)

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
    return " ".join(compact).strip()[:48]

# ──────────────────────────────────────────────────────────────────────────────
# Localizar y extraer NOMBRE (1ª línea + posible continuación)
# ──────────────────────────────────────────────────────────────────────────────
def find_owner_bands(bgr: np.ndarray, row_y: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Devuelve (roi_l1, roi_l2) donde cada ROI es (x0,y0,x1,y1).
    Estrategia robusta: ventana vertical alrededor de row_y y columna fija (texto bajo "Apellidos...").
    """
    h, w = bgr.shape[:2]
    # Columna de texto (izq. de NIF)
    x0 = int(w * 0.33)
    x1 = int(w * 0.60)

    # Ventana vertical en torno a la fila
    band_h = int(h * 0.09)            # alto de banda
    l1_h   = int(h * 0.04)            # alto por línea
    y_mid  = row_y
    y0_band = max(0, y_mid - band_h//2)
    y1_band = min(h, y_mid + band_h//2)

    # Intento de localizar "NIF" para cortar mejor por la derecha
    band = bgr[y0_band:y1_band, x0:int(w*0.90)]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)
    x_nif = None
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        for t, lx in zip(data.get("text", []), data.get("left", [])):
            if (t or "").strip().upper() == "NIF":
                x_nif = lx
                break
        if x_nif is not None:
            break
    if x_nif is not None:
        x1 = min(x1, x0 + x_nif - 10)

    # ROIs de línea 1 y 2 (justo debajo)
    y0_l1 = max(y0_band, y_mid - l1_h//2)
    y1_l1 = min(h, y0_l1 + l1_h)
    y0_l2 = min(h, y1_l1 + 4)
    y1_l2 = min(h, y0_l2 + l1_h)

    roi1 = (x0, y0_l1, x1, y1_l1)
    roi2 = (x0, y0_l2, x1, y1_l2)
    return roi1, roi2

def extract_owner_from_rois(bgr: np.ndarray, roi1: Tuple[int,int,int,int], roi2: Tuple[int,int,int,int]) -> Tuple[str, dict]:
    """
    Extrae nombre combinando:
      - Línea 1 limpia (ignorando cabeceras si aparecen)
      - Si L1 trae salto de línea (t1_raw con '\n'), toma la segunda parte
      - Si L2 parece una continuación válida y no está en JUNK_2NDLINE, se concatena
    """
    debug = {}
    (x0, y0, x1, y1) = roi1
    (x0b, y0b, x1b, y1b) = roi2

    def ocr_roi(x0, y0, x1, y1):
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        variants = [
            ocr_text(bw,  psm=6, whitelist=WL),
            ocr_text(bwi, psm=6, whitelist=WL),
            ocr_text(bw,  psm=7, whitelist=WL),
            ocr_text(bwi, psm=7, whitelist=WL),
            ocr_text(bw,  psm=13, whitelist=WL),
        ]
        best = ""
        for t in variants:
            if len(t) > len(best):
                best = t
        return best.strip()

    t1_raw = ocr_roi(*roi1)
    t2_raw = ocr_roi(*roi2)

    debug["t1_raw"] = t1_raw
    debug["t2_raw"] = t2_raw

    # Si L1 contiene cabecera, descártala y usa lo de debajo (salto o L2)
    t1_parts = [p.strip() for p in (t1_raw or "").split("\n") if p.strip()]
    t1_clean = ""
    t1_extra = ""

    if t1_parts:
        # Si la primera parte parece cabecera, ignórala
        if is_header_line(t1_parts[0]):
            if len(t1_parts) >= 2:
                t1_clean = t1_parts[1]
                t1_extra = " ".join(t1_parts[2:]) if len(t1_parts) > 2 else ""
            else:
                t1_clean = ""
        else:
            t1_clean = t1_parts[0]
            t1_extra = " ".join(t1_parts[1:]) if len(t1_parts) > 1 else ""
    # Si aún vacío (OCR raro), prueba L2 tal cual
    if not t1_clean and t2_raw:
        t1_clean = t2_raw

    # Limpiezas
    def is_junk_second(s: str) -> bool:
        if not s: return True
        U = s.strip().upper()
        if U in JUNK_2NDLINE: return True
        if len(U) <= 1: return True
        if any(ch.isdigit() for ch in U): return True
        if "[" in U or "]" in U: return True
        # Tokens de dirección típicos -> ruido
        if U.startswith(("CL ", "LG ", "AV ", "RUA ", "PZ ", "PL ")): return True
        return False

    base = clean_owner_line(t1_clean)
    # Si t1_extra trae 2ª línea “pegada” dentro de L1
    extra = ""
    if t1_extra:
        extra_cand = clean_owner_line(t1_extra)
        if extra_cand and not is_junk_second(extra_cand):
            extra = extra_cand

    # Si L2 parece nombre corto válido y no es basura, y aún no tenemos extra
    if (not extra) and t2_raw:
        cand2 = clean_owner_line(t2_raw)
        if cand2 and not is_junk_second(cand2):
            extra = cand2

    owner = " ".join([x for x in [base, extra] if x]).strip()[:64]

    debug["picked_from"] = "strict" if not t1_extra else "from_l1_break"
    debug["t1_extra_raw"] = t1_extra
    return owner, debug

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Zona de croquis (izquierda)
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=260)
    neighs = contours_centroids(mp, min_area=180)
    if not mains:
        return {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}, {"rows": []}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # por y
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.30)**2:
            side = angle_to_compass(mcx, mcy, best[0], best[1])

        roi1, roi2 = find_owner_bands(bgr, mcy)
        owner, odbg = extract_owner_from_rois(bgr, roi1, roi2)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl_map = {"norte":"N","noreste":"NE","este":"E","sureste":"SE",
                           "sur":"S","suroeste":"SW","oeste":"W","noroeste":"NW"}
                lbl = lbl_map.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.42), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        # dibujar ROIs
        (x0,y0,x1,y1) = roi1; (x2,y2,x3,y3) = roi2
        if annotate:
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)
            cv2.rectangle(vis, (x2,y2), (x3,y3), (0,200,200), 2)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": {
                "band": [min(x0,x2), min(y0,y2), max(x1,x3), max(y1,y3)],
                "y_line1": [y0,y1],
                "y_line2_hint": [y2,y3],
                "x0": x0,
                "x1": x1,
                **odbg
            }
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
        "dpi": resolve_dpi(),
        "JUNK_2NDLINE": JUNK_2NDLINE,
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SW/W/NW"),
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
        blank = np.zeros((260, 900, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:80]}", (10,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
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
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )






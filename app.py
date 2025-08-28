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
app = FastAPI(title="AutoCatastro AI", version="0.5.8")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI configurable
def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, "").strip())
        return v if 72 <= v <= 800 else default
    except Exception:
        return default

FAST_DPI = _env_int("FAST_DPI", 340)   # manda cuando FAST_MODE=1
PDF_DPI  = _env_int("PDF_DPI",  380)   # manda cuando FAST_MODE=0

def current_dpi() -> int:
    return FAST_DPI if FAST_MODE else PDF_DPI

# Lista de tokens basura en 2ª línea (configurable)
JUNK_2NDLINE = set([t.strip().upper() for t in os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",") if t.strip()])

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
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","APELLIDO","NOMBRE","NOMBRES",
    "RAZON","RAZÓN","SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION",
    "REFERENCIA","CATASTRAL","TITULARIDAD","PRINCIPAL","DIRECCIÓN","DIRECCION"
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
    dpi = current_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    bgr = np.array(pil)[:, :, ::-1]
    return bgr

def crop_map(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
    h, w = bgr.shape[:2]
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.05); right = int(w * 0.40)
    return bgr[top:bottom, left:right], (left, top)

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

# ──────────────────────────────────────────────────────────────────────────────
# Direcciones (8 vientos)
# ──────────────────────────────────────────────────────────────────────────────
def side_of8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=Este, 90=Norte
    # Sectores de 45° con umbral 22.5°
    if -22.5 <= ang <= 22.5:   return "este"
    if 22.5  < ang <= 67.5:    return "noreste"
    if 67.5  < ang <= 112.5:   return "norte"
    if 112.5 < ang <= 157.5:   return "noroeste"
    if ang > 157.5 or ang <= -157.5: return "oeste"
    if -157.5 < ang <= -112.5: return "suroeste"
    if -112.5 < ang <= -67.5:  return "sur"
    return "sureste"

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

# ── PARCHE: detectar encabezado “Apellidos/Nombre/Razón social” en L1 ────────
def looks_like_header_line(s: str) -> bool:
    """
    Detecta si un renglón se parece al encabezado 'Apellidos Nombre / Razón social'
    incluso con OCR sucio (AP, NOM, RAZ, SOC, etc.). Consideramos encabezado
    si aparecen ≥2 de estos indicios.
    """
    u = (s or "").upper()
    u = re.sub(r"[^A-ZÁÉÍÓÚÜÑ ]+", " ", u)
    u = re.sub(r"(.)\1{2,}", r"\1\1", u)  # AAAA -> AA
    hits = 0
    for pat in ("APELL", "APEL", "AP ", " A P ", "NOM", "NOMB", "RAZ", "RAZON", "RAZÓN", "SOC", "SOCIAL"):
        if pat in u:
            hits += 1
    return hits >= 2

def pick_owner_from_l1(raw: str) -> Tuple[str, str]:
    """
    Devuelve (owner, extra_from_break) usando SOLO la banda de la 1ª línea.
    - Si la primera línea parece encabezado → la ignoramos y buscamos el primer
      renglón siguiente que parezca nombre (sin extra).
    - Si la primera línea es nombre y trae salto con otro renglón → devolvemos ese
      'extra' solo si también parece nombre (sin dígitos, ≤26, etc.).
    """
    raw = (raw or "").strip()
    if not raw:
        return "", ""

    parts = [p.strip() for p in raw.split("\n") if p.strip()]
    if not parts:
        return "", ""

    # ¿La primera línea es encabezado?
    if looks_like_header_line(parts[0]):
        for p in parts[1:]:
            cand = clean_owner_line(p.upper())
            if len(cand) >= 8:
                return cand, ""  # NO adjuntamos extra si venimos de encabezado
        return "", ""

    # Primera línea NO parece encabezado → procesado normal
    first = parts[0]
    cand = clean_owner_line(first.upper())
    extra = ""
    if len(parts) >= 2:
        second_inline = parts[1]
        s2 = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", second_inline.upper()).strip()
        if s2 and len(s2) <= 26 and not any(ch.isdigit() for ch in s2):
            extra = s2
    return cand, extra

# ──────────────────────────────────────────────────────────────────────────────
# Localizar banda de texto y extraer dueño (L1 + posible L2)
# ──────────────────────────────────────────────────────────────────────────────
def row_text_band(bgr: np.ndarray, row_y: int) -> Tuple[int,int,int,int,int,int]:
    """
    Determina una banda a la derecha de los croquis para capturar:
    - L1: renglón del nombre (alrededor de row_y)
    - L2: renglón inmediatamente inferior (por si el nombre sigue)
    Devuelve (x0, x1, y1_top, y1_bot, y2_top, y2_bot)
    """
    h, w = bgr.shape[:2]
    # Columnas: entre ~44% y ~82% del ancho (se ajusta bien a los PDFs probados)
    x0 = int(w * 0.44)
    x1 = int(w * 0.82)
    # Alturas: ventana en torno a row_y
    y1_top = max(0, row_y - int(h * 0.04))
    y1_bot = min(h, row_y + int(h * 0.03))
    y2_top = min(h, y1_bot + int(h * 0.01))
    y2_bot = min(h, y2_top + int(h * 0.03))
    return x0, x1, y1_top, y1_bot, y2_top, y2_bot

def extract_owner_for_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Extrae el titular para una fila:
      1) OCR de L1 → pick_owner_from_l1() (PARCHE aplicado aquí)
      2) Si pick_owner_from_l1 no da 'extra', probamos L2 (hasta 26 chars, sin dígitos y no JUNK_2NDLINE)
    """
    x0, x1, y1t, y1b, y2t, y2b = row_text_band(bgr, row_y)
    debug = {
        "band":[x0, y1t, x1, y2b],
        "y_line1":[y1t, y1b],
        "y_line2_hint":[y2t, y2b],
        "x0":x0, "x1":x1
    }

    # L1
    roi1 = bgr[y1t:y1b, x0:x1]
    owner, l1_extra = "", ""
    if roi1.size != 0:
        g = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
        # Probamos inversos/psm distintos
        candidates = [
            ocr_text(bw,  psm=6, whitelist=WL),
            ocr_text(bwi, psm=6, whitelist=WL),
            ocr_text(bw,  psm=7, whitelist=WL),
            ocr_text(bwi, psm=7, whitelist=WL),
            ocr_text(bw,  psm=13, whitelist=WL),
        ]
        debug["t1_raw"] = ""
        for raw in candidates:
            if raw:
                debug["t1_raw"] = raw
            own, extra = pick_owner_from_l1(raw)
            if own:
                owner, l1_extra = own, extra
                break

    # Si hay extra dentro de L1 (siguiente renglón de L1) y no es basura → concatenar
    if l1_extra:
        s2 = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", l1_extra.upper()).strip()
        if s2 and len(s2) <= 26 and not any(ch.isdigit() for ch in s2) and s2 not in JUNK_2NDLINE:
            owner = (owner + " " + s2).strip()

    # Si no añadimos nada desde L1 y existe L2, probamos L2
    if roi1.size != 0 and y2b - y2t > 6:
        roi2 = bgr[y2t:y2b, x0:x1]
        if roi2.size != 0:
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.20, fy=1.20, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)
            WL2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '-"
            t2_candidates = [
                ocr_text(bw2,  psm=7,  whitelist=WL2),
                ocr_text(bwi2, psm=7,  whitelist=WL2),
                ocr_text(bw2,  psm=6,  whitelist=WL2),
                ocr_text(bwi2, psm=6,  whitelist=WL2),
                ocr_text(bw2,  psm=13, whitelist=WL2),
            ]
            t2_raw = ""
            for c in t2_candidates:
                if c:
                    t2_raw = c
                    break
            debug["t2_raw"] = t2_raw
            s2 = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+", "", (t2_raw or "").upper()).strip()
            if s2 and len(s2) <= 26 and not any(ch.isdigit() for ch in s2) and s2 not in JUNK_2NDLINE:
                # solo concatenamos si L1 dio un nombre base
                if owner:
                    owner = (owner + " " + s2).strip()

    # Limpieza final de owner
    owner = clean_owner_line(owner)
    return owner, debug

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (8 vientos)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract8(bgr: np.ndarray,
                             annotate: bool = False,
                             annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    crop, (ox, oy) = crop_map(bgr)
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(340 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(240 if FAST_MODE else 180))
    if not mains:
        empty8 = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
        return empty8, {"rows": [], "raster":{"dpi": current_dpi()}}, vis

    mains_abs  = [(cx+ox, cy+oy, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])  # por filas
    neighs_abs = [(cx+ox, cy+oy, a) for (cx,cy,a) in neighs]

    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    used_sides = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of8((mcx, mcy), best)

        owner, o_dbg = extract_owner_for_row(bgr, row_y=mcy)

        if side and owner and side not in used_sides and not linderos.get(side):
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl_map = {"norte":"N","noreste":"NE","este":"E","sureste":"SE",
                           "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"}
                lbl = lbl_map.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if annotate_names and owner:
                cv2.putText(vis, owner[:28], (int(w*0.44), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.44), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": o_dbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi": current_dpi()}}
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
        "dpi": current_dpi(),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar puntos cardinales"),
    names:  int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract8(
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
                 names:  int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        return ExtractOut(
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True, "raster":{"dpi": current_dpi()}} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract8(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
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
            debug={"exception": str(e), "raster":{"dpi": current_dpi()}} if debug else None
        )


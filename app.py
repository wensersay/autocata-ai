from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query, UploadFile, File
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math, time
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract
from zipfile import ZipFile

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.0-batch")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI: precedencia FAST_DPI si FAST_MODE=1, si no PDF_DPI, si no 340/420 por defecto
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

FAST_DPI = _int_env("FAST_DPI", 340)   # recomendado 300–360
PDF_DPI  = _int_env("PDF_DPI",  420)   # recomendado 400–460

# Lista de tokens ruidosos detectados a veces en la 2ª línea (configurable)
JUNK_2NDLINE = [t.strip().upper() for t in (os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",")) if t.strip()]

# Pistas de nombres (ayuda opcional para unir 2ª línea cuando L1 acaba en nombre compuesto)
NAME_HINTS = set("""
JOSE,JOSEP,JOAO,JUAN,JUANA,ANA,MARIA,MARÍA,LUIS,ANTONIO,ANTÓN,MANUEL,MANOEL,
CARLOS,ENRIQUE,FERNANDO,JAVIER,PABLO,ALBERTO,FRANCISCO,FRAN,RAFAEL,RAFA,ROBERTO,
LORENA,LAURA,PAULA,PILAR,SARA,SONIA,ALBA,ALICIA,ALMUDENA,BEATRIZ,CRISTINA,
DIEGO,DAVID,IGNACIO,NACHO,SERGIO,VICTOR,VÍCTOR,MIGUEL,GABRIEL,RAMON,RAMÓN,
JORGE,OSCAR,ÓSCAR,RODRIGO,ADRIAN,ADRIÁN,IVAN,IVÁN
""".replace("\n","").split(","))
NAME_HINTS |= {t.strip().upper() for t in os.getenv("NAME_HINTS_EXTRA","").split(",") if t.strip()}

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
    "GALICIA","[LUGO]","[BARCELONA]","DE","DEL","LA","LOS","LAS"
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
# Raster (pág. 2)
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = FAST_DPI if FAST_MODE else PDF_DPI
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    bgr = np.array(pil)[:, :, ::-1]
    return bgr

# ──────────────────────────────────────────────────────────────────────────────
# Detección de croquis (verde/rosa) y lados
# ──────────────────────────────────────────────────────────────────────────────
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
    for lo, hi in g_ranges: mg |= cv2.inRange(hsv, lo, hi)
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp |= cv2.inRange(hsv, lo, hi)
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
    ang = math.degrees(math.atan2(-(sy), sx))
    if -22.5 <= ang <= 22.5:   return "este"
    if 22.5 < ang <= 67.5:     return "noreste"
    if 67.5 < ang <= 112.5:    return "norte"
    if 112.5 < ang <= 157.5:   return "noroeste"
    if ang > 157.5 or ang < -157.5: return "oeste"
    if -157.5 <= ang < -112.5: return "suroeste"
    if -112.5 <= ang < -67.5:  return "sur"
    if -67.5 <= ang < -22.5:   return "sureste"
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
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
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
    # coger 1ª línea que parezca nombre; si contiene un salto interno con otro nombre, unirlo
    if lines:
        l1 = lines[0].upper()
        if UPPER_NAME_RE.match(l1):
            # si hay segundo renglón dentro del mismo bloque (p.ej., "JOSE\nLUIS")
            if len(lines) >= 2:
                l2 = lines[1].upper()
                if UPPER_NAME_RE.match(l2) and l2 not in BAD_TOKENS and l2 not in GEO_TOKENS:
                    if l2 in NAME_HINTS or len(l2) <= 12:
                        joined = f"{clean_owner_line(l1)} {clean_owner_line(l2)}".strip()
                        if 8 <= len(joined) <= 52:
                            return joined
            nm = clean_owner_line(l1)
            if len(nm) >= 8:
                return nm
    # fallback recorriendo todas
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):           continue
        nm = clean_owner_line(U)
        if len(nm) >= 8:
            return nm
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Localizar banda de texto y extraer titular (1ª línea + posible continuación)
# ──────────────────────────────────────────────────────────────────────────────
def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Devuelve (owner, dbg) usando una banda de texto a la derecha de los croquis.
    Heurística robusta que funciona con 300–420 DPI.
    """
    h, w = bgr.shape[:2]

    # Columna de texto a la derecha de los croquis (aprox entre 33% y 80% del ancho)
    x0 = int(w * 0.33)
    x1 = int(w * 0.80)

    # Ventana vertical: 2 renglones alrededor de row_y (alto ~ 210 px a 340 DPI → escala)
    half = max(90, int(h * 0.025))
    y0 = max(0, row_y - half)
    y1 = min(h, row_y + half)

    band = bgr[y0:y1, x0:x1]
    dbg = {"band":[x0,y0,x1,y1]}
    if band.size == 0:
        return "", dbg

    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    g = enhance_gray(g)
    bw, bwi = binarize(g)

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    # OCR variante directa (toda la banda) y también 2 subbandas L1/L2
    # Cortes horizontales a mitad de la banda para ayudar a separar renglones
    H = g.shape[0]
    cut = max(28, int(H * 0.48))
    l1 = bw[0:cut, :]
    l2 = bw[cut:, :]

    raw_l1 = ocr_text(l1, psm=6, whitelist=WL)
    raw_l2 = ocr_text(l2, psm=6, whitelist=WL)
    dbg.update({"t1_raw":raw_l1, "t2_raw":raw_l2})

    # 1) Primera línea limpia
    first = pick_owner_from_text(raw_l1)
    owner = first

    # 2) Si la segunda línea trae un continuado razonable, intentar unir (con filtro de ruido)
    sec = (raw_l2.split("\n")[0].strip().upper() if raw_l2 else "")
    if sec and sec not in JUNK_2NDLINE and sec not in BAD_TOKENS and not any(ch.isdigit() for ch in sec):
        if UPPER_NAME_RE.match(sec) and sec not in GEO_TOKENS:
            sec_clean = clean_owner_line(sec)
            if sec_clean and len(sec_clean) <= 26:
                if (sec_clean in NAME_HINTS) or (first and len(sec_clean) <= 12):
                    owner = (first + " " + sec_clean).strip()

    # 3) Si L1 trae el salto embebido (p.ej. "RODRIGUEZ ALVAREZ JOSE\nLUIS"), pick_owner_from_text ya lo trata.

    return owner, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte donde viven croquis (izquierda)
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mins_main  = 320 if FAST_MODE else 220
    mins_neigh = 220 if FAST_MODE else 160
    mains  = contours_centroids(mg, min_area=mins_main)
    neighs = contours_centroids(mp, min_area=mins_neigh)
    if not mains:
        return {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}, {"rows":[]}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of8((mcx, mcy), best)

        owner, odbg = extract_owner_from_row(bgr, row_y=mcy)

        # asignación si el lado está libre
        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        # anotaciones
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
            "ocr": odbg
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi": (FAST_DPI if FAST_MODE else PDF_DPI)}}
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints básicos
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "dpi_in_use": FAST_DPI if FAST_MODE else PDF_DPI
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/S/E/O (8 rumbos)"),
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
        blank = np.zeros((260, 900, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:80]}", (10,130),
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
            debug={"exception": str(e)} if debug else None
        )

# ──────────────────────────────────────────────────────────────────────────────
# NUEVO: uploads en lote (hasta 10 PDFs)
# ──────────────────────────────────────────────────────────────────────────────
MAX_FILES = 10
MAX_BYTES_EACH = 20 * 1024 * 1024  # 20 MB por archivo (ajustable)

def _run_pipeline_bytes(pdf_bytes: bytes, want_preview: bool, labels: bool, names: bool, want_debug: bool):
    bgr = page2_bgr(pdf_bytes)
    linderos, dbg, vis = detect_rows_and_extract(
        bgr, annotate=bool(labels or names), annotate_names=bool(names)
    )
    owners_detected = [o["owner"] for o in dbg["rows"] if o.get("owner")]
    owners_detected = list(dict.fromkeys(owners_detected))[:8]

    note = None
    if not any(linderos.values()):
        note = "No se pudo determinar lado/vecino con suficiente confianza."

    preview_png = None
    if want_preview:
        ok, png = cv2.imencode(".png", vis)
        if ok:
            preview_png = png.tobytes()

    return linderos, owners_detected, note, (dbg if want_debug else None), preview_png

@app.post("/extract_batch", dependencies=[Depends(check_token)])
async def extract_batch(
    files: List[UploadFile] = File(..., description="Hasta 10 PDFs"),
    debug: int = Query(0, description="1=devolver debug por archivo"),
):
    if not files:
        raise HTTPException(status_code=400, detail="Adjunta al menos un PDF.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Máximo {MAX_FILES} archivos por lote.")

    results = []
    for f in files:
        started = time.time()
        entry = {"filename": f.filename}
        try:
            data = await f.read()
            if len(data) == 0:
                raise HTTPException(status_code=400, detail="Archivo vacío.")
            if len(data) > MAX_BYTES_EACH:
                raise HTTPException(status_code=400, detail=f"Archivo > {MAX_BYTES_EACH//(1024*1024)} MB.")

            linderos, owners, note, dbg, _ = _run_pipeline_bytes(
                pdf_bytes=data,
                want_preview=False,
                labels=False,
                names=False,
                want_debug=bool(debug),
            )

            entry.update({
                "linderos": linderos,
                "owners_detected": owners,
                "note": note,
            })
            if debug:
                entry["debug"] = dbg
        except Exception as e:
            entry["error"] = str(e)
        finally:
            entry["elapsed_ms"] = int((time.time() - started) * 1000)

        results.append(entry)

    return {"count": len(results), "results": results}

@app.post("/preview_batch", dependencies=[Depends(check_token)])
async def preview_batch(
    files: List[UploadFile] = File(..., description="Hasta 10 PDFs"),
    labels: int = Query(1, description="1=mostrar N/S/E/O"),
    names: int  = Query(1, description="1=anotar nombre abreviado"),
):
    if not files:
        raise HTTPException(status_code=400, detail="Adjunta al menos un PDF.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Máximo {MAX_FILES} archivos por lote.")

    memzip = io.BytesIO()
    with ZipFile(memzip, mode="w") as zf:
        idx = 0
        for f in files:
            idx += 1
            try:
                data = await f.read()
                if len(data) == 0:
                    raise ValueError("Archivo vacío")
                if len(data) > MAX_BYTES_EACH:
                    raise ValueError(f"Archivo > {MAX_BYTES_EACH//(1024*1024)} MB")

                _, _, _, _, png = _run_pipeline_bytes(
                    pdf_bytes=data,
                    want_preview=True,
                    labels=bool(labels),
                    names=bool(names),
                    want_debug=False,
                )

                base = (f.filename or f"file_{idx}.pdf").rsplit(".", 1)[0]
                out_name = f"{idx:02d}_{base}.png"
                if png is None:
                    blank = np.zeros((240, 640, 3), np.uint8)
                    cv2.putText(blank, "No preview", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    ok, fallback = cv2.imencode(".png", blank)
                    png = fallback.tobytes() if ok else b""
                zf.writestr(out_name, png)

            except Exception as e:
                err_name = f"{idx:02d}_{(f.filename or 'file')}.error.txt"
                zf.writestr(err_name, f"Error: {e}")

    memzip.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="previews.zip"'}
    return StreamingResponse(memzip, media_type="application/zip", headers=headers)



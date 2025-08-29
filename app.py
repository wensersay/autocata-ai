from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math, time
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.3")

# ──────────────────────────────────────────────────────────────────────────────
# Flags y entorno
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

# DPI (preferencia: ladder si AUTO_DPI=1; si no, FAST_DPI / PDF_DPI según FAST_MODE)
AUTO_DPI     = (os.getenv("AUTO_DPI", "1").strip() == "1")
DPI_LADDER   = [int(x) for x in os.getenv("DPI_LADDER", "300,340,360").replace(" ", "").split(",") if x.isdigit()]
FAST_DPI     = int(os.getenv("FAST_DPI", "300") or "300")
PDF_DPI      = int(os.getenv("PDF_DPI",  "340") or "340")

# Ruido 2ª línea (coma-separado)
JUNK_2NDLINE = [x.strip().upper() for x in os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",") if x.strip()]

# Hints de nombres extra (opcional)
NAME_HINTS_EXTRA = [x.strip().upper() for x in os.getenv("NAME_HINTS_EXTRA", "").split(",") if x.strip()]

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
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","L'HOSPITALET","HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

HEADER_NOISE_TOKENS = [
    "A N R", "A N RZ", "NIF", "MAROUZAS", "O SAVIÑAO", "O SAVINAO",
    "APELLIDOS", "RAZON", "RAZÓN", "SOCIAL"
]

def is_header_noise(text: str) -> bool:
    u = (text or "").upper()
    return any(tok in u for tok in HEADER_NOISE_TOKENS)

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
# Raster con ladder
# ──────────────────────────────────────────────────────────────────────────────
def raster_page2(pdf_bytes: bytes) -> Tuple[np.ndarray, dict]:
    timings = {}
    used_dpi = None
    ladder_used = []
    start = time.time()

    if not AUTO_DPI:
        dpi = FAST_DPI if FAST_MODE else PDF_DPI
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
        if not pages:
            raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
        pil: Image.Image = pages[0].convert("RGB")
        bgr = np.array(pil)[:, :, ::-1]
        used_dpi = dpi
        timings["raster_ms"] = int((time.time()-start)*1000)
        return bgr, {"dpi": used_dpi, "ladder_used": [used_dpi]}

    # Ladder
    for dpi in DPI_LADDER:
        ladder_used.append(dpi)
        t0 = time.time()
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
        if not pages:
            continue
        pil: Image.Image = pages[0].convert("RGB")
        bgr = np.array(pil)[:, :, ::-1]
        used_dpi = dpi
        # no scoring complicado: nos quedamos con el primero; si necesitas heurística, aquí.
        timings["raster_ms"] = int((time.time()-t0)*1000)
        return bgr, {"dpi": used_dpi, "ladder_used": ladder_used}

    raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2 con el ladder configurado.")

# ──────────────────────────────────────────────────────────────────────────────
# Color masks (verde=parcela principal, magenta/rosa=vecina)
# ──────────────────────────────────────────────────────────────────────────────
def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [
        (np.array([35,  20,  50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15,  50], np.uint8), np.array([100,255, 255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160, 20, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges:
        mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges:
        mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3)
    mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3)
    mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int,int,int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        out.append((cx,cy,int(a)))
    out.sort(key=lambda x: -x[2])
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Side (8 direcciones)
# ──────────────────────────────────────────────────────────────────────────────
def side_of_8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x,  y  = pt_xy
    vx, vy = x - cx, y - cy
    ang = (math.degrees(math.atan2(-(vy), vx)) + 360.0) % 360.0  # 0=E, 90=N
    bins = [
        ("este",       337.5,  22.5),
        ("noreste",     22.5,  67.5),
        ("norte",       67.5, 112.5),
        ("noroeste",   112.5, 157.5),
        ("oeste",      157.5, 202.5),
        ("suroeste",   202.5, 247.5),
        ("sur",        247.5, 292.5),
        ("sureste",    292.5, 337.5),
    ]
    for name, a0, a1 in bins:
        if a0 < a1 and a0 <= ang < a1:
            return name
        if a0 > a1 and (ang >= a0 or ang < a1):  # wrap-around
            return name
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

def pick_first_line(txt: str) -> Tuple[str, str]:
    """Devuelve (l1, l1_extra) donde l1 es la 1ª línea y l1_extra el posible 'salto' siguiente."""
    if not txt: return "", ""
    parts = [p for p in txt.split("\n") if p.strip()]
    if not parts: return "", ""
    l1 = parts[0].strip()
    l1_extra = ""
    if len(parts) >= 2:
        # Si la segunda línea parece seguir el nombre (p. ej. 'LUIS'), la devolvemos
        l1_extra = parts[1].strip()
    return l1, l1_extra

def clean_second_line(raw: str) -> str:
    if not raw: return ""
    U = raw.upper().strip()
    # corta en el primer número, [, : o tokens raros
    U = re.split(r"[\[\]:0-9]", U)[0].strip()
    if not U: return ""
    if U in JUNK_2NDLINE: return ""
    if len(U) > 26: U = U[:26].strip()
    return U

# ──────────────────────────────────────────────────────────────────────────────
# OCR por fila (banda) con PARCHE: preferir t2_raw si t1_raw huele a encabezado
# ──────────────────────────────────────────────────────────────────────────────
def ocr_owner_for_row(bgr: np.ndarray, row_y: int, x0: int, x1: int) -> Tuple[str, dict]:
    """
    Extrae nombre del titular usando dos sub-bandas:
    - L1 alrededor de row_y
    - L2 inmediatamente debajo
    Parche: si t1_raw contiene encabezado (A N R / NIF / MAROUZAS...), usar t2_raw.
    """
    h, w = bgr.shape[:2]
    band_h = int(h * 0.11)  # banda vertical moderada
    y_top  = max(0, row_y - band_h//2)
    y_bot  = min(h, row_y + band_h//2)

    x0 = max(0, x0); x1 = min(w, x1)
    band = bgr[y_top:y_bot, x0:x1]
    dbg = {"band":[x0, y_top, x1, y_bot]}

    if band.size == 0:
        return "", {"band": dbg["band"], "t1_raw":"", "t2_raw":"", "picked_from":"empty"}

    g  = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    g  = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    g  = enhance_gray(g)
    bw, bwi = binarize(g)

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    t_full = ocr_text(bw,  psm=6, whitelist=WL)
    if not t_full:
        t_full = ocr_text(bwi, psm=6, whitelist=WL)

    l1, l1_extra = pick_first_line(t_full)
    t1_raw = (l1 or "").strip()
    t1_extra_raw = (l1_extra or "").strip()

    # Segunda sub-banda (debajo) como L2 “por si acaso”
    l2_top = int(0.55 * band.shape[0])
    l2 = band[l2_top: , :]
    t2_raw = ""
    if l2.size != 0:
        lg = cv2.cvtColor(l2, cv2.COLOR_BGR2GRAY)
        lg = cv2.resize(lg, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        lg = enhance_gray(lg)
        lbw, lbwi = binarize(lg)
        t2_raw = ocr_text(lbw,  psm=6, whitelist=WL) or ocr_text(lbwi, psm=6, whitelist=WL)
        t2_raw = (t2_raw or "").split("\n")[0].strip()

    # Limpieza básica
    c1 = clean_owner_line(t1_raw)
    c2 = clean_owner_line(t2_raw)
    c1_extra = clean_second_line(t1_extra_raw)

    picked_from = "strict"

    # ── PARCHE: si L1 huele a encabezado y L2 tiene algo válido, preferimos L2 ──
    if c1 and is_header_noise(t1_raw) and c2:
        owner = c2
        picked_from = "prefer_t2_due_header"
    else:
        # si L1 es válido y L1_extra es nombre probable corto, lo añadimos
        if c1 and c1_extra and UPPER_NAME_RE.match(c1_extra):
            owner = (c1 + " " + c1_extra).strip()
            picked_from = "from_l1_break"
        elif c1:
            owner = c1
        elif c2:
            owner = c2
            picked_from = "fallback_t2"
        else:
            owner = ""

    dbg.update({
        "t1_raw": t1_raw,
        "t1_extra_raw": t1_extra_raw,
        "t2_raw": t2_raw,
        "picked_from": picked_from
    })
    return owner, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Detección filas + extracción
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray, annotate: bool=False, annotate_names: bool=False
) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Área izquierda donde están croquis (heurística robusta)
    top = int(h * 0.08); bottom = int(h * 0.92)
    left = int(w * 0.05); right = int(w * 0.42)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    if not mains:
        return (
            {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            {"rows":[],"timings_ms":{}},
            vis
        )

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])   # por y asc (arriba→abajo)
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    # Columna de texto (derecha del croquis)
    x0_text = int(w * 0.27)
    x1_text = int(w * 0.63)

    rows_dbg = []
    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecina más cercana
        best = None; best_d = 1e12
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)

        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of_8((mcx,mcy), best)

        # OCR banda
        owner, ocr_dbg = ocr_owner_for_row(bgr, mcy, x0_text, x1_text)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {
                    "norte":"N", "noreste":"NE", "este":"E", "sureste":"SE",
                    "sur":"S", "suroeste":"SW", "oeste":"W", "noroeste":"NW",
                }.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255,255,255), 2, cv2.LINE_AA)
            if annotate_names and owner:
                cv2.putText(vis, owner[:28], (int(w*0.66), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.66), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

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
        "raster": {
            "AUTO_DPI": AUTO_DPI,
            "DPI_LADDER": DPI_LADDER,
            "FAST_DPI": FAST_DPI,
            "PDF_DPI": PDF_DPI
        },
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar puntos cardinales"),
    names: int  = Query(0, description="1=mostrar nombres abreviados")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr, rast_dbg = raster_page2(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((260, 800, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:64]}", (10,140),
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
        bgr, rast_dbg = raster_page2(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False, annotate_names=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = None
        if debug:
            dbg = vdbg
            dbg["raster"] = rast_dbg
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )




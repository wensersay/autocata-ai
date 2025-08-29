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
app = FastAPI(title="AutoCatastro AI", version="0.6.2")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")

def get_dpi() -> int:
    fast_dpi = int(os.getenv("FAST_DPI", "340").strip() or "340")
    pdf_dpi  = int(os.getenv("PDF_DPI",  "420").strip() or "420")
    return fast_dpi if FAST_MODE else pdf_dpi

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _split_env_list(val: str) -> List[str]:
    if not val:
        return []
    parts = [p.strip().upper() for p in val.split(",")]
    return [p for p in parts if p]

# nombres comunes — ampliables con NAME_HINTS_EXTRA
COMMON_NAME_HINTS = {
    "JOSE","JOSÉ","MARIA","MARÍA","LUIS","ANTONIO","ANTONÍO","MANUEL","ANA",
    "JUAN","JAVIER","CARLOS","CARMEN","PABLO","PEDRO","RAFAEL",
    "MARTA","LAURA","PAULA","ALBA","SARA","ALVARO","ÁLVARO",
    "ANDRES","ANDRÉS","ALFONSO","ALFONSA","IGNACIO","NURIA","ROSA","BEATRIZ",
    "HECTOR","HÉCTOR","SERGIO","ROBERTO","NOELIA","IRENE","PILAR","SONIA",
    "PATRICIA","DAVID","DANIEL","RODRIGO","ADRIAN","ADRIÁN",
    "FRANCISCO","FRANCISCA","XOSE","XOSÉ","UXIA","UXÍA","NOA","IVAN","IVÁN",
    "BREOGAN","BREOGÁN","XOANA","DOLORES","ISABEL","LORENZO","LORENA"
}
COMMON_NAME_HINTS |= set(_split_env_list(os.getenv("NAME_HINTS_EXTRA","")))

JUNK_2NDLINE = set(_split_env_list(os.getenv("JUNK_2NDLINE","Z,VA,EO,SS,KO,KR,LN,LK")))

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

class BatchIn(BaseModel):
    pdf_urls: List[AnyHttpUrl]

class BatchOutItem(BaseModel):
    url: AnyHttpUrl
    result: ExtractOut

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades comunes OCR / limpieza
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","AP","A","AN","AR",
    "APELLIDOS/NOMBRE","APELLIDOS/NOMBRE/RAZÓN","RAZÓN/SOCIAL","RAZON/SOCIAL",
    "DIRECCION","DIRECCIÓN","CSV","SELLO"
}

# geotokens / ruido que no deben entrar en el nombre
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","DIRECCION","DIRECCIÓN","CSV","SELLO"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

FORBIDDEN_SIGLAS = {"N","S","E","O","NE","NO","SE","SO","NW","SW","EO"} | JUNK_2NDLINE

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
    dpi = get_dpi()
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
    for lo, hi in g_ranges:
        mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges:
        mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
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

DIR8 = ['este','noreste','norte','noroeste','oeste','suroeste','sur','sureste']
def side_of_8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=este, 90=norte
    if ang < 0: ang += 360
    idx = int((ang + 22.5) // 45) % 8
    return DIR8[idx]

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

def has_header_noise(s: str) -> bool:
    U = (s or "").upper()
    if not U: return False
    # si contiene cualquiera de estos, lo tratamos como cabecera/ruido
    header_keys = ["APELL", "RAZON", "RAZÓN", "NIF", "AP ", " NOM", "SOC", "APELLIDOS/NOMBRE", "RAZÓN/SOCIAL", "RAZON/SOCIAL"]
    return any(k in U for k in header_keys)

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

def looks_like_name_token(s: str) -> bool:
    U = s.strip().upper()
    if not U:
        return False
    if not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ ]+", U):
        return False
    compact = U.replace(" ","")
    if len(compact) <= 2:
        return False
    if U in FORBIDDEN_SIGLAS or U in JUNK_2NDLINE:
        return False
    if not re.search(r"[AEIOUÁÉÍÓÚÜ]", U):
        return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Columna nombres y OCR L1+L2
# ──────────────────────────────────────────────────────────────────────────────
def detect_name_column_bounds(bgr: np.ndarray) -> Tuple[int,int]:
    h, w = bgr.shape[:2]
    y0s, y1s = int(h*0.10), int(h*0.20)
    band = bgr[y0s:y1s, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)
    x_ap, x_nif = None, None
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        for t, lx in zip(words, xs):
            T = (t or "").upper()
            if "APELLIDOS" in T and x_ap is None:
                x_ap = lx
            if T == "NIF" and x_nif is None:
                x_nif = lx
    if x_ap is not None and x_nif is not None and x_nif - x_ap > w*0.12:
        return max(0, int(x_ap*0.95)), min(w-1, int((x_ap + x_nif)*0.98))
    return int(w*0.33), int(w*0.63)

def ocr_band_first_second_lines(bgr: np.ndarray, row_y: int, x0: int, x1: int):
    h, w = bgr.shape[:2]
    y0 = max(0, row_y - int(h*0.04))
    y1 = min(h, row_y + int(h*0.04))
    band = bgr[y0:y1, x0:x1]
    if band.size == 0:
        return {"band":[x0,y0,x1,y1],"y_line1":[y0,(y0+y1)//2],"y_line2_hint":[(y0+y1)//2,y1],
                "x0":x0,"x1":x1,"t1_raw":"","t1_extra_raw":"","t2_raw":"","picked_from":""}

    l1 = band[0: max(1, (y1-y0)//2), : ]
    l2 = band[max(1, (y1-y0)//2):, : ]

    def _prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        return bw, bwi

    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    t1_raw, t1_extra_raw, t2_raw = "", "", ""
    # OCR L1
    for im in _prep(l1):
        txt = ocr_text(im, psm=6, whitelist=WL)
        if txt:
            lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
            if lines:
                t1_raw = lines[0]
                if len(lines) > 1:
                    t1_extra_raw = " ".join(lines[1:]).strip()
            if t1_raw:
                break
    if not t1_raw:
        for im in _prep(l1):
            txt = ocr_text(im, psm=7, whitelist=WL)
            lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
            if lines:
                t1_raw = lines[0]
                if len(lines) > 1:
                    t1_extra_raw = " ".join(lines[1:]).strip()
                break
    # OCR L2
    for im in _prep(l2):
        txt2 = ocr_text(im, psm=6, whitelist=WL)
        if txt2:
            t2_raw = txt2.split("\n")[0].strip()
            break

    def _clean2(raw: str) -> str:
        if not raw: return ""
        raw = raw.upper()
        raw = re.split(r"[\[\]:/\\0-9]", raw)[0]
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:26]

    t1_raw_u = _clean2(t1_raw)
    t1_extra_u = _clean2(t1_extra_raw)
    t2_u = _clean2(t2_raw)

    # Filtrar cabecera/ruido en candidatos
    if has_header_noise(t1_raw_u):
        t1_raw_u = ""
        t1_extra_u = ""
    if has_header_noise(t2_u):
        t2_u = ""

    picked_from = ""

    # Candidatos con pequeña puntuación
    def score_name(s: str) -> int:
        if not s: return -10
        sc = 0
        toks = [t for t in s.split() if t]
        sc += min(len(toks), 5) * 2              # más tokens (hasta 5) suma
        if s in COMMON_NAME_HINTS: sc += 3
        if looks_like_name_token(s): sc += 2
        if has_header_noise(s): sc -= 10
        if s in FORBIDDEN_SIGLAS: sc -= 8
        return sc

    # Preferencias:
    # 1) si L1 parece nombre y su extra también, usar L1 + extra (from_l1_break)
    # 2) si L1 es nombre y L2 también, comparar puntuaciones y elegir mejor añadido
    # 3) si L1 es basura (o vacío) pero L2 parece nombre, usar L2
    l1_ok = bool(t1_raw_u) and looks_like_name_token(t1_raw_u) and not has_header_noise(t1_raw_u)
    l2_ok = bool(t2_u) and (looks_like_name_token(t2_u) or t2_u in COMMON_NAME_HINTS) and not has_header_noise(t2_u)
    extra_ok = bool(t1_extra_u) and (looks_like_name_token(t1_extra_u) or t1_extra_u in COMMON_NAME_HINTS)

    # Resultado base: solo L1 si válido
    best_line = t1_raw_u if l1_ok else ""
    picked_from = "l1_only" if best_line else ""

    # Intentar sumar una 2ª línea
    candidate = best_line
    if l1_ok and extra_ok:
        candidate = (t1_raw_u + " " + t1_extra_u).strip()
        picked_from = "from_l1_break"
    elif l1_ok and l2_ok:
        c2 = (t1_raw_u + " " + t2_u).strip()
        candidate = c2 if score_name(c2) >= score_name(t1_raw_u) else t1_raw_u
        picked_from = "strict"
    elif (not l1_ok) and l2_ok:
        candidate = t2_u
        picked_from = "l2_only"
    # si nada válido, quedará vacío

    return {
        "band":[x0,y0,x1,y1],
        "y_line1":[y0, y0 + (y1-y0)//2],
        "y_line2_hint":[y0 + (y1-y0)//2, y1],
        "x0":x0, "x1":x1,
        "t1_raw":t1_raw_u,
        "t1_extra_raw":t1_extra_u,
        "t2_raw":t2_u,
        "picked_from":picked_from
    }

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    t0 = time.time()
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    top = int(h * 0.10); bottom = int(h * 0.90)
    left = int(w * 0.05); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(320 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))

    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    rows_dbg = []

    if not mains:
        return linderos, {"rows": [], "raster": {"dpi": get_dpi()}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    x0_col, x1_col = detect_name_column_bounds(bgr)

    used_sides = set()
    for (mcx, mcy, _a) in mains_abs[:6]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of_8((mcx, mcy), best)

        band = ocr_band_first_second_lines(bgr, row_y=mcy, x0=x0_col, x1=x1_col)

        # escoger owner del band con reglas finales
        parts = []
        if band["picked_from"] == "from_l1_break":
            parts = [band["t1_raw"], band["t1_extra_raw"]]
        elif band["picked_from"] == "strict":
            # ya ponderado en OCR; elegir el mejor compuesto
            s1 = (band["t1_raw"] + " " + band["t2_raw"]).strip()
            s2 = band["t1_raw"]
            parts = [s1] if len(s1) >= len(s2) else [s2]
        elif band["picked_from"] == "l2_only":
            parts = [band["t2_raw"]]
        else:
            parts = [band["t1_raw"]]

        owner = " ".join([p for p in parts if p]).strip()
        owner = clean_owner_line(owner)
        owner = re.sub(r"\s+", " ", owner).strip()
        owner = owner[:64]

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

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
                cv2.putText(vis, owner[:28], (int(w*0.66), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner[:28], (int(w*0.66), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": {
                "band": band["band"],
                "y_line1": band["y_line1"],
                "y_line2_hint": band["y_line2_hint"],
                "x0": band["x0"], "x1": band["x1"],
                "t1_raw": band["t1_raw"],
                "t1_extra_raw": band.get("t1_extra_raw",""),
                "t2_raw": band["t2_raw"],
            },
            "picked_from": band.get("picked_from","")
        })

    timings = {"rows_pipeline": int((time.time()-t0)*1000)}
    dbg = {"rows": rows_dbg, "raster": {"dpi": get_dpi()}, "timings_ms": timings}
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
        "dpi": get_dpi(),
        "env": {
            "FAST_DPI": os.getenv("FAST_DPI",""),
            "PDF_DPI": os.getenv("PDF_DPI",""),
            "NAME_HINTS_EXTRA": os.getenv("NAME_HINTS_EXTRA",""),
            "JUNK_2NDLINE": os.getenv("JUNK_2NDLINE","")
        }
    }

def _blank_png(err: str) -> StreamingResponse:
    blank = np.zeros((260, 900, 3), np.uint8)
    cv2.putText(blank, f"ERR: {err[:110]}", (10,140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    ok, png = cv2.imencode(".png", blank)
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar letras N/NE/E/SE/S/SO/O/NO"),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        return _blank_png(str(e))

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
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )

@app.post("/extract-batch", dependencies=[Depends(check_token)])
def extract_batch(data: BatchIn = Body(...), debug: bool = Query(False)) -> List[BatchOutItem]:
    out: List[BatchOutItem] = []
    urls = list(data.pdf_urls)[:10]
    for u in urls:
        try:
            single = extract(ExtractIn(pdf_url=u), debug=debug)
        except Exception as e:
            single = ExtractOut(
                linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
                owners_detected=[], note=f"Error batch: {e}", debug={"exception": str(e)} if debug else None
            )
        out.append(BatchOutItem(url=u, result=single))
    return out






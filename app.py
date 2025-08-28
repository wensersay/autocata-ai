from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple, Set
import requests, io, re, os, math
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.4")

# ──────────────────────────────────────────────────────────────────────────────
# Flags / entorno
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v else default
    except Exception:
        return default

# DPI: cuando FAST_MODE=1 → FAST_DPI; si no → PDF_DPI
FAST_DPI = env_int("FAST_DPI", 340)   # si lo pones en Railway, manda éste en FAST_MODE
PDF_DPI  = env_int("PDF_DPI",  500)

def chosen_dpi() -> int:
    return FAST_DPI if FAST_MODE else PDF_DPI

# Lista de “códigos basura” que pueden colarse en 2ª línea (ENV: JUNK_2NDLINE="Z,VA,EO,SS,KO,KR")
def load_junk_2ndline() -> Set[str]:
    raw = os.getenv("JUNK_2NDLINE", "")
    if not raw:
        raw = "Z,VA,EO,SS,KO,KR,LE,LN,LO,BS,BT"
    toks = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return set(toks)

JUNK_2NDLINE = load_junk_2ndline()

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
# Utilidades de texto
# ──────────────────────────────────────────────────────────────────────────────
BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","AP","APS","APSGS","APSG","ANR","A","N","R","Z","D"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

def is_alpha_like(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # Solo letras, espacios, apóstrofes y guiones
    return re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ\s'´’-]+", s) is not None

def looks_like_address_start(s: str) -> bool:
    # abreviaturas direccionales y palabras de dirección
    addr_starts = {
        "CL","CALLE","AV","AVDA","AVENIDA","LG","LUGAR","PL","PLAZA","CTRA","CARRETERA",
        "RUA","RÚA","PO","PORTAL","ESC","ESCALERA","PISO","PT","CP","MONFORTE","LEMOS",
        "HOSPITALET","L'HOSPITALET","BARCELONA","LUGO","SAVIÑAO","O SAVIÑAO"
    }
    parts = [p for p in re.split(r"[^\wÁÉÍÓÚÜÑ]+", s) if p]
    if not parts:
        return False
    return parts[0] in addr_starts

def clean_name_tokens(line: str) -> str:
    # Limpia tokens no deseados y corta cuando empiezan direcciones/números
    if not line:
        return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t):
            break
        if t in GEO_TOKENS:
            break
        if t in BAD_TOKENS:
            continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 6:
            break
    # Compactar conectores
    compact = []
    for t in out:
        if not compact and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    return " ".join(compact).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Raster y color
# ──────────────────────────────────────────────────────────────────────────────
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

def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = chosen_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde (parcela objetivo)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # Magenta/rosa (vecinos)
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
# Orientación (8 vientos)
# ──────────────────────────────────────────────────────────────────────────────
def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=Este, 90=Norte
    # Segmentos de 45° (centro en E, NE, N, NW, W, SW, S, SE)
    if -22.5 <= ang <= 22.5:    return "este"
    if 22.5 < ang <= 67.5:      return "noreste"
    if 67.5 < ang <= 112.5:     return "norte"
    if 112.5 < ang <= 157.5:    return "noroeste"
    if ang > 157.5 or ang < -157.5: return "oeste"
    if -157.5 <= ang < -112.5:  return "suroeste"
    if -112.5 <= ang < -67.5:   return "sur"
    if -67.5 <= ang < -22.5:    return "sureste"
    return ""

SIDE_LABEL = {
    "norte":"N", "noreste":"NE", "este":"E", "sureste":"SE",
    "sur":"S", "suroeste":"SO", "oeste":"O", "noroeste":"NO"
}

# ──────────────────────────────────────────────────────────────────────────────
# OCR utilidades
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

# ──────────────────────────────────────────────────────────────────────────────
# Localizar banda de texto (columna “Apellidos Nombre / Razón social”)
# ──────────────────────────────────────────────────────────────────────────────
def find_text_band(bgr: np.ndarray, row_y: int) -> Tuple[int,int,int,int]:
    """
    Devuelve un rectángulo (x0,y0,x1,y1) que cubre ~2 líneas bajo la cabecera.
    Si no localiza la cabecera, usa una heurística estable basada en proporciones.
    """
    h, w = bgr.shape[:2]
    # Ventana vertical alrededor de la fila para buscar cabecera
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    # Columna de texto aproximada a la derecha del croquis
    x_text0_heur = int(w * 0.26)
    x_text1_heur = int(w * 0.63)

    band = bgr[y0s:y1s, x_text0_heur:x_text1_heur]
    if band.size == 0:
        # fallback muy conservador
        y0 = max(0, row_y - int(h*0.02))
        y1 = min(h, y0 + int(h*0.07))
        return x_text0_heur, y0, x_text1_heur, y1

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    # Buscar "APELLIDOS" y "NIF". Si aparece, situar la banda justo debajo.
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
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            # Coordenadas absolutas
            y0 = y0s + header_bottom + 4
            y1 = min(h, y0 + int(h * 0.06))  # cubrir ~2 líneas
            if x_nif is not None:
                x0 = x_text0_heur
                x1 = min(x_text1_heur, x_text0_heur + x_nif - 8)
            else:
                x0 = x_text0_heur
                x1 = x_text1_heur
            if x1 - x0 > (x_text1_heur - x_text0_heur) * 0.35:
                return x0, y0, x1, y1

    # Fallback por proporción si no se detecta cabecera
    y0 = max(0, row_y - int(h*0.02))
    y1 = min(h, y0 + int(h * 0.06))
    return x_text0_heur, y0, x_text1_heur, y1

# ──────────────────────────────────────────────────────────────────────────────
# OCR de la banda → titular (con PARCHE que prioriza t1_extra_raw)
# ──────────────────────────────────────────────────────────────────────────────
def ocr_owner_from_band(bgr: np.ndarray, row_y: int) -> Tuple[str, dict]:
    """
    Lee dos líneas de la banda. Obtiene:
      - l1_raw (línea 1 completa), l1_extra (si l1 contiene salto → segunda línea en el mismo bloque)
      - l2_raw (línea 2 inferior)
    PARCHE: prioriza l1_extra (t1_extra_raw). Si no hay, usa l2_raw.
    Limpieza: corta a 26 chars, descarta JUNK_2NDLINE, direcciones y no-alfabético.
    """
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = find_text_band(bgr, row_y)

    band = bgr[y0:y1, x0:x1]
    out_dbg = {
        "band":[x0,y0,x1,y1]
    }
    if band.size == 0:
        return "", out_dbg

    # Partimos la banda en dos mitades (L1 superior, L2 inferior)
    H = band.shape[0]
    mid = y0 + H//2
    y1a0, y1a1 = y0, mid
    y2a0, y2a1 = mid, y1

    out_dbg["y_line1"] = [y1a0, y1a1]
    out_dbg["y_line2_hint"] = [y2a0, y2a1]
    out_dbg["x0"] = x0
    out_dbg["x1"] = x1

    # OCR en cada mitad
    def read_half(yA, yB) -> str:
        roi = bgr[yA:yB, x0:x1]
        if roi.size == 0:
            return ""
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '´-"
        variants = [
            ocr_text(bw,  psm=6, whitelist=WL),
            ocr_text(bwi, psm=6, whitelist=WL),
            ocr_text(bw,  psm=7, whitelist=WL),
            ocr_text(bwi, psm=7, whitelist=WL),
        ]
        best = ""
        for t in variants:
            if len(t) > len(best):
                best = t
        return best.strip().upper()

    l1_raw = read_half(y1a0, y1a1)
    l2_raw = read_half(y2a0, y2a1)
    out_dbg["t1_raw"] = l1_raw
    out_dbg["t2_raw"] = l2_raw

    # Línea 1: si contiene salto, separar base + extra
    base = l1_raw
    l1_extra = ""
    if "\n" in l1_raw:
        parts = [p for p in l1_raw.split("\n") if p.strip()]
        base = parts[0].strip().upper()
        # Unir el resto como “extra”
        if len(parts) > 1:
            l1_extra = " ".join(p.strip().upper() for p in parts[1:] if p.strip())
    out_dbg["t1_extra_raw"] = l1_extra

    # Limpieza del nombre base
    base_clean = clean_name_tokens(base)

    # ─── PARCHE: priorizar siempre t1_extra_raw antes que L2 ──────────────────
    picked_from = "strict"
    extra_added = ""

    def accept_and_clean(c: str) -> Optional[str]:
        c = (c or "").strip().upper()
        if not c:
            return None
        # cortar por separadores duros si aparecen
        for ch in "[]:":
            if ch in c:
                c = c.split(ch, 1)[0].strip()
        c = re.sub(r"[^\wÁÉÍÓÚÜÑ\s'´’-]+", " ", c).strip()
        if not c:
            return None
        if c in JUNK_2NDLINE:
            return None
        if looks_like_address_start(c):
            return None
        if not is_alpha_like(c):
            return None
        c = c[:26].strip()  # regla de negocio
        return clean_name_tokens(c)

    # 1) intentar con extra de L1
    cand = accept_and_clean(l1_extra)
    if cand:
        extra_added = cand
        picked_from = "from_l1_break"
    else:
        # 2) intentar con L2
        cand = accept_and_clean(l2_raw)
        if cand:
            extra_added = cand
            picked_from = "from_l2"

    if not base_clean:
        owner = (extra_added or "").strip()
    else:
        owner = base_clean if not extra_added else f"{base_clean} {extra_added}"

    owner = re.sub(r"\s{2,}", " ", owner).strip()
    out_dbg["picked_from"] = picked_from
    return owner, out_dbg

# ──────────────────────────────────────────────────────────────────────────────
# Detección de filas (puntos) y extracción
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Zona del croquis (izquierda)
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    # Centroides
    mains  = contours_centroids(mg, min_area=(300 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]
    mains_abs.sort(key=lambda t: t[1])

    # Linderos con 8 claves
    linderos = {
        "norte":"", "noreste":"", "este":"", "sureste":"",
        "sur":"", "suroeste":"", "oeste":"", "noroeste":""
    }

    rows_dbg = []
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:8]:
        # Vecino más cercano
        best, best_d = None, 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            side = side_of((mcx, mcy), best)

        owner, ocr_dbg = ocr_owner_from_band(bgr, row_y=mcy)

        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        # anotaciones
        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)  # verde = main
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)     # rosa = vecino
                lbl = SIDE_LABEL.get(side, "")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.45), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.45), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": ocr_dbg
        })

    dbg = { "rows": rows_dbg }
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
        "dpi": chosen_dpi(),
        "junk_2ndline": sorted(list(JUNK_2NDLINE))
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
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
        blank = np.zeros((240, 800, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:90]}", (10,120),
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
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        # deduplicar conservando orden
        owners_detected = list(dict.fromkeys(owners_detected))[:10]

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


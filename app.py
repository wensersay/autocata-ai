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

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ──────────────────────────────────────────────────────────────────────────────
# DPI por entorno (con override)
# ──────────────────────────────────────────────────────────────────────────────
FAST_DPI = int(os.getenv("FAST_DPI", "340"))   # usado si FAST_MODE=1
SLOW_DPI = int(os.getenv("SLOW_DPI", "500"))   # usado si FAST_MODE=0
PDF_DPI  = os.getenv("PDF_DPI", "").strip()    # override para ambos modos (si presente)

def effective_dpi() -> int:
    try:
        return int(PDF_DPI) if PDF_DPI else (FAST_DPI if FAST_MODE else SLOW_DPI)
    except Exception:
        return FAST_DPI if FAST_MODE else SLOW_DPI

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
# Utilidades / Constantes
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

# Palabras/zonas que no deben contaminar el nombre (usadas mínimamente)
BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL"
}
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Pistas de nombres (se puede ampliar por ENV)
DEFAULT_NAME_HINTS = {
    "ANA","ANDRES","ANTONIO","CARLOS","DAVID","FRANCISCO","JAVIER","JOSE",
    "JOSE LUIS","JUAN","LAURA","LUIS","MANUEL","MARIA","MARTA","PEDRO","PILAR",
    "RAFAEL","ROSA","SARA"
}
NAME_HINTS_EXTRA = {t.strip().upper() for t in os.getenv("NAME_HINTS_EXTRA", "").split(",") if t.strip()}
NAME_HINTS = DEFAULT_NAME_HINTS | NAME_HINTS_EXTRA

# Ruido común en segunda línea (configurable)
JUNK_2NDLINE = {t.strip().upper() for t in os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",") if t.strip()}

# Columnas de texto (proporciones del ancho) – parametrizables por ENV
try:
    TEXT_COL_X0 = float(os.getenv("TEXT_COL_X0", "0.27"))
    TEXT_COL_X1 = float(os.getenv("TEXT_COL_X1", "0.59"))
except Exception:
    TEXT_COL_X0, TEXT_COL_X1 = 0.27, 0.59

def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Descarga y raster
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
    dpi = effective_dpi()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

# ──────────────────────────────────────────────────────────────────────────────
# Visión: máscaras y contornos
# ──────────────────────────────────────────────────────────────────────────────
def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # verde (parcela objetivo)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # magenta/rojo (vecinas)
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
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv5, k5) if (cv5:=k5) else mg
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, k5, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int,int,int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: 
            continue
        M = cv2.moments(c)
        if M["m00"] == 0: 
            continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

# 8 direcciones
def side_of_eight(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # Este=0°, Norte=90°
    if -22.5 <= ang < 22.5:   return "este"
    if 22.5 <= ang < 67.5:    return "noreste"
    if 67.5 <= ang < 112.5:   return "norte"
    if 112.5 <= ang < 157.5:  return "noroeste"
    if ang >= 157.5 or ang < -157.5: return "oeste"
    if -157.5 <= ang < -112.5:return "suroeste"
    if -112.5 <= ang < -67.5: return "sur"
    return "sureste"

# ──────────────────────────────────────────────────────────────────────────────
# OCR helpers
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

def clean_name_tokens(line: str) -> str:
    if not line:
        return ""
    U = line.upper().strip()
    # eliminar tokens con dígitos o corchetes
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", U) if t and not any(ch.isdigit() for ch in t)]
    # cortar si aparece un token sospechoso
    out = []
    for t in toks:
        if t in BAD_TOKENS:
            continue
        out.append(t)
    # compactar conectores
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name[:48]

def merge_second_line(base_name: str, l1_extra: str, l2_raw: str) -> Tuple[str, dict]:
    """
    Regla de unión:
      1) Si L1 ya trae un salto con un token plausible (p.ej. 'LUIS'), priorizarlo.
      2) Si L2 limpia produce 1–2 tokens y NO están en JUNK_2NDLINE, y
         alguno está en NAME_HINTS o es plausible (>=3 letras), unir.
    """
    dbg = {"second_line_used": False, "second_line_reason": ""}

    # (1) extra de L1
    l1e = clean_name_tokens(l1_extra) if l1_extra else ""
    if l1e:
        if l1e not in JUNK_2NDLINE:
            merged = (base_name + " " + l1e).strip()
            return merged[:64], {"second_line_used": True, "second_line_reason": "from_l1_break"}

    # (2) L2 cruda
    l2 = clean_name_tokens(l2_raw) if l2_raw else ""
    if l2:
        # tokens
        toks = l2.split()
        if 1 <= len(toks) <= 2:
            # descartar si todo es ruído conocido
            if all(t in JUNK_2NDLINE for t in toks):
                return base_name, dbg
            # si alguno es pista de nombre o tiene >=3 letras, unir
            if any((t in NAME_HINTS or len(t) >= 3) for t in toks):
                merged = (base_name + " " + l2).strip()
                return merged[:64], {"second_line_used": True, "second_line_reason": "l2_ok"}
    return base_name, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Extracción por filas (croquis + columna texto)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # ROI del croquis (lado izquierdo)
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    if crop.size == 0:
        return {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}, {"rows":[]}, vis

    mg, mp = color_masks(crop)
    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}, {"rows":[]}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    # columnas de texto absolutas
    x0_col = int(w * TEXT_COL_X0)
    x1_col = int(w * TEXT_COL_X1)

    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    used_sides = set()
    rows_dbg = []

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano
        best = None; best_d = 1e12
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None:
            side = side_of_eight((mcx, mcy), best)

        # Banda de texto: 2 renglones alrededor de la fila
        band_h = int(h * 0.06)         # altura total de banda
        line_h = int(h * 0.04)         # alto cada línea
        y1_top = max(0, mcy - int(line_h*0.5))
        y1_bot = min(h, y1_top + line_h)
        y2_top = min(h, y1_bot + int(line_h*0.1))
        y2_bot = min(h, y2_top + line_h)

        # OCR línea 1
        roi1 = bgr[y1_top:y1_bot, x0_col:x1_col]
        t1_raw = ""
        t1_extra_raw = ""
        if roi1.size != 0:
            g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            g1 = cv2.resize(g1, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
            g1 = enhance_gray(g1)
            bw1, bwi1 = binarize(g1)
            WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
            cand1 = [
                ocr_text(bw1, 6, WL),
                ocr_text(bwi1, 6, WL),
                ocr_text(bw1, 7, WL),
                ocr_text(bwi1, 7, WL),
            ]
            # si trae salto de línea en L1, cortamos y guardamos extra
            best1 = ""
            for c in cand1:
                if c:
                    best1 = c
                    break
            if "\n" in best1:
                parts = [p.strip() for p in best1.split("\n") if p.strip()]
                t1_raw = parts[0] if parts else ""
                t1_extra_raw = " ".join(parts[1:]) if len(parts) > 1 else ""
            else:
                t1_raw = best1

        name1 = clean_name_tokens(t1_raw)

        # OCR línea 2 (hint)
        roi2 = bgr[y2_top:y2_bot, x0_col:x1_col]
        t2_raw = ""
        if roi2.size != 0:
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)
            WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
            cand2 = [
                ocr_text(bw2, 6, WL),
                ocr_text(bwi2, 6, WL),
                ocr_text(bw2, 7, WL),
                ocr_text(bwi2, 7, WL),
            ]
            for c in cand2:
                if c:
                    t2_raw = c
                    break

        # unir con 2ª línea (y/o extra de L1)
        owner, sl_dbg = merge_second_line(name1, t1_extra_raw, t2_raw)

        # asignar si no está usado el lado
        if side and owner and side not in used_sides:
            linderos[side] = owner
            used_sides.add(side)

        # Anotaciones
        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SO","oeste":"O","noroeste":"NO"
                }.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.62), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.62), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": {
                "band": [x0_col, max(0, y1_top-int(line_h*0.5)), x1_col, min(h, y2_bot+int(line_h*0.5))],
                "y_line1": [y1_top, y1_bot],
                "y_line2_hint": [y2_top, y2_bot],
                "x0": x0_col, "x1": x1_col,
                "t1_raw": t1_raw,
                "t1_extra_raw": t1_extra_raw,
                "t2_raw": t2_raw,
            },
            **sl_dbg
        })

    dbg = {"rows": rows_dbg, "raster": {"dpi": effective_dpi()}}
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
        "dpi_plan": {
            "FAST_DPI": FAST_DPI,
            "SLOW_DPI": SLOW_DPI,
            "PDF_DPI_override": PDF_DPI or None,
            "effective": effective_dpi()
        }
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
        blank = np.zeros((260, 900, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:84]}", (10,140),
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

    # Solo texto (desactivar visión/OCR)
    if TEXT_ONLY:
        return ExtractOut(
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True, "raster_dpi": effective_dpi()} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [r.get("owner","") for r in vdbg.get("rows",[]) if r.get("owner")]
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
            debug={"exception": str(e), "raster_dpi": effective_dpi()} if debug else None
        )


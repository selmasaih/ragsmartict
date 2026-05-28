"""Text extraction for PDFs and images.

PDFs are read with PyMuPDF (better word spacing than pypdf). Image files (and
image-only PDF pages) fall back to OCR via pytesseract when available.
"""
import os

import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def ocr_available() -> bool:
    if not _PIL_OK:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_image(img) -> str:
    try:
        return pytesseract.image_to_string(img, lang="fra+eng")
    except Exception:
        try:
            return pytesseract.image_to_string(img)
        except Exception:
            return ""


def extract_image(path: str) -> str:
    """OCR a standalone image file. Returns '' if OCR is unavailable."""
    if not ocr_available():
        return ""
    return _ocr_image(Image.open(path)).strip()


def extract_pdf(path: str):
    """Yield (page_number, text) for each page that has usable text.
    Falls back to OCR for image-only pages when tesseract is available."""
    pages = []
    ocr = ocr_available()
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            if len(text.strip()) < 50 and ocr:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = _ocr_image(img)
            pages.append((i + 1, text))
    return pages


def extract_any(path: str):
    """Return a list of (page_number, text) for any supported file type."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in IMAGE_EXTS:
        return [(1, extract_image(path))]
    return []

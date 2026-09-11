import os
import glob
import shutil
import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")

def _find_poppler_bin():
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        return os.path.dirname(pdftoppm)
    matches = glob.glob(
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\oschwartz10612.Poppler_*\poppler-*\Library\bin")
    )
    return matches[0] if matches else None

POPPLER_PATH = _find_poppler_bin()

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    full_text = ""
    for image in images:
        full_text += pytesseract.image_to_string(image, lang="vie") + "\n"
    return full_text
import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.07.0\Library\bin"  # sửa đúng theo đường dẫn thật của anh

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    full_text = ""
    for image in images:
        full_text += pytesseract.image_to_string(image, lang="vie") + "\n"
    return full_text
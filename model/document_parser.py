from PyPDF2 import PdfReader
import docx
from odf.opendocument import load
from odf.text import P

def extract_text(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif filepath.endswith(".odt"):
        odt = load(filepath)
        paragraphs = odt.getElementsByType(P)
        return "\n".join([p.firstChild.data if p.firstChild else "" for p in paragraphs])
    else:
        raise ValueError("Format non supporté")

import fitz

def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract selectable text from a PDF, preserving 1-based PDF page numbers."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({"page": page_number, "text": page.get_text("text").strip()})
    return pages

import fitz


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}")

    text_parts: list[str] = []
    for page in doc:
        text_parts.append(page.get_text())

    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError("No readable text found in PDF")
    return text

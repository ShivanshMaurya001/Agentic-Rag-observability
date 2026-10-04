import pdfplumber


def extract_text_from_pdf(file_path: str) -> list[dict]:
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages
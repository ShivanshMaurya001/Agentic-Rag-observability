
from pathlib import Path

from app.pdf_service import extract_text_from_pdf
from app.chunking_service import create_page_chunks
from app.vector_service import store_chunks

PDF_DIR = Path("test_data")


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in test_data/")
        return

    total_chunks = 0

    for pdf_path in pdf_files:
        pages = extract_text_from_pdf(str(pdf_path))
        chunks = create_page_chunks(
            str(pdf_path),
            pages,
            pdf_path.name,
        )

        if chunks:
            store_chunks(chunks)

        total_chunks += len(chunks)
        print(f"{pdf_path.name}: {len(chunks)} chunks stored")

    print(f"\nPDFs processed: {len(pdf_files)}")
    print(f"Total chunks processed: {total_chunks}")


if __name__ == "__main__":
    main()

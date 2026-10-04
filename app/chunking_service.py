import hashlib

def generate_document_hash(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()



def generate_chunk_id(document_hash: str, chunk_position: int) -> str:
    return f"{document_hash}_{chunk_position}"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks_from_file(file_path: str) -> list[dict]:
    document_hash = generate_document_hash(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = chunk_text(text)

    return [
        {
            "chunk_id": generate_chunk_id(document_hash, position),
            "text": chunk,
        }
        for position, chunk in enumerate(chunks)
    ]


def create_page_chunks(file_path: str, pages: list[dict], doc_name: str) -> list[dict]:
    document_hash = generate_document_hash(file_path)
    chunks = []
    chunk_position = 0

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:
            chunks.append(
                {
                    "chunk_id": generate_chunk_id(document_hash, chunk_position),
                    "text": chunk,
                    "doc_name": doc_name,
                    "page": page["page"],
                }
            )
            chunk_position += 1

    return chunks



from app.config import K_DEFAULT
from app.vector_service import search_chunks


def retrieve(query: str, k: int = K_DEFAULT) -> list[dict]:
    results = search_chunks(
        question=query,
        top_k=k,
    )

    chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    distances = results.get("distances")

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}

        chunk = {
            "chunk_id": chunk_id,
            "text": documents[index],
        }

        if "doc_name" in metadata:
            chunk["doc_name"] = metadata["doc_name"]

        if "page" in metadata:
            chunk["page"] = metadata["page"]

        if distances is not None:
            chunk["score"] = distances[0][index]

        chunks.append(chunk)

    return chunks
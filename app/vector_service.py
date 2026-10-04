import chromadb

from app.embedding_service import generate_embedding


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="document_chunks"
)


def store_chunk(chunk: dict):
    embedding = generate_embedding(
        chunk["text"]
    )

    collection.upsert(
        ids=[chunk["chunk_id"]],
        documents=[chunk["text"]],
        embeddings=[embedding],
        metadatas=[
            {
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
            }
        ],
    )


def store_chunks(chunks: list[dict]):
    for chunk in chunks:
        store_chunk(chunk)


def search_chunks(question: str, top_k: int):
    question_embedding = generate_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    return results
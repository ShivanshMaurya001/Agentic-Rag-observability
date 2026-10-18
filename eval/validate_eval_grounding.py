
"""Offline, read-only lexical validation for the evaluation dataset.

This script flags suspicious chunk labels. It does not change the dataset
or decide whether an answer is semantically correct.
"""

import argparse
import json
import re
from pathlib import Path

import chromadb


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "in", "into", "is", "it", "its",
    "may", "might", "of", "on", "or", "our", "should", "that",
    "the", "their", "them", "there", "these", "this", "those",
    "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}


def key_terms(text: str) -> set[str]:
    """Extract simple content terms; this is not semantic analysis."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        word for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only lexical checks for eval chunk labels."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to eval_queries.json",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        required=True,
        help="Path to the existing persistent ChromaDB directory",
    )
    parser.add_argument(
        "--collection",
        default="document_chunks",
        help="Existing ChromaDB collection name",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval K used to report corpus coverage (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.k < 1:
        raise SystemExit("--k must be at least 1.")

    # Read the evaluation dataset.
    with args.queries.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise SystemExit("Evaluation dataset must be a JSON list.")

    # Read existing ChromaDB data only. No query/embedding operation is used.
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_collection(name=args.collection)
    stored = collection.get(include=["documents", "metadatas"])

    chunk_ids = stored["ids"]
    documents = stored["documents"] or []
    metadatas = stored["metadatas"] or []

    chunks = []
    for index, chunk_id in enumerate(chunk_ids):
        text = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        chunks.append({
            "id": chunk_id,
            "text": text or "",
            "metadata": metadata or {},
            "terms": key_terms(text or ""),
        })

    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    total_chunks = len(chunks)

    print("=" * 80)
    print("OFFLINE EVALUATION GROUNDING REPORT")
    print("Read-only mode: no dataset or ChromaDB writes")
    print("Matching method: lowercase keyword overlap (heuristic only)")
    print("=" * 80)
    print(f"Dataset: {args.queries}")
    print(f"ChromaDB path: {args.chroma_path}")
    print(f"Collection: {args.collection}")
    print(f"Total corpus chunks: {total_chunks}")
    print(f"Configured K: {args.k}")

    if total_chunks:
        effective_k = min(args.k, total_chunks)
        print(
            f"Configured K / total chunks: "
            f"{args.k}/{total_chunks} = {args.k / total_chunks:.1%}"
        )
        print(
            f"Effective top-K corpus coverage: "
            f"{effective_k}/{total_chunks} = "
            f"{effective_k / total_chunks:.1%}"
        )
    else:
        print("Corpus coverage: unavailable (no chunks found).")

    print(
        "Note: this coverage is the maximum share of the corpus represented "
        "by K chunks, not measured retrieval performance."
    )

    answerable_count = 0
    total_flagged_labelled = 0
    total_flagged_unlabelled = 0

    for number, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            print(f"\nWARNING: Dataset entry {number} is not an object.")
            continue

        if item.get("is_adversarial", False):
            continue

        answerable_count += 1
        query = str(item.get("query", ""))
        expected = item.get("expected_answer")
        labels = item.get("relevant_chunks", [])

        if not isinstance(labels, list):
            labels = []
            print("WARNING: relevant_chunks is not a list.")

        labels = [str(chunk_id) for chunk_id in labels]
        expected_text = expected if isinstance(expected, str) else ""
        terms = key_terms(expected_text)

        print("\n" + "=" * 80)
        print(f"ANSWERABLE QUERY #{number}")
        print(f"Query: {query}")
        print(f"Expected answer: {expected!r}")
        print(f"Expected-answer key terms: {sorted(terms)}")
        print(f"Labelled chunk IDs: {labels}")

        unknown_ids = [cid for cid in labels if cid not in chunk_by_id]
        if unknown_ids:
            print(f"FLAG — labelled IDs missing from corpus: {unknown_ids}")

        # A lexical hit means at least one extracted key term occurs in a chunk.
        # This is deliberately a flag, not a semantic-support verdict.
        hits = []
        for chunk in chunks:
            matched = terms & chunk["terms"]
            if matched:
                hits.append((chunk, matched))

        hit_ids = {chunk["id"] for chunk, _ in hits}
        labelled_ids = set(labels)

        print("\nChunks containing expected-answer key terms:")
        if not hits:
            print("  No lexical hits found.")
        else:
            for chunk, matched in hits:
                marker = (
                    "LABELLED"
                    if chunk["id"] in labelled_ids
                    else "UNLABELLED — FLAG for review"
                )
                if chunk["id"] not in labelled_ids:
                    total_flagged_unlabelled += 1

                print(f"\n  [{marker}] Chunk ID: {chunk['id']}")
                print(f"  Matched terms: {sorted(matched)}")
                print(f"  Metadata: {chunk['metadata']}")
                print("  Full chunk text:")
                print(chunk["text"])

        print("\nLabel checks:")
        for chunk_id in labels:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue

            matched = terms & chunk["terms"]
            if not matched:
                total_flagged_labelled += 1
                print(
                    f"  FLAG — labelled chunk has no expected-answer "
                    f"key-term overlap: {chunk_id}"
                )
            else:
                print(
                    f"  Keyword overlap found for labelled chunk "
                    f"{chunk_id}: {sorted(matched)}"
                )

        if not terms:
            print(
                "FLAG — expected answer produced no key terms; "
                "manual review is needed."
            )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Answerable queries examined: {answerable_count}")
    print(
        "Labelled chunks flagged with zero keyword overlap: "
        f"{total_flagged_labelled}"
    )
    print(
        "Unlabelled chunk hits flagged for review: "
        f"{total_flagged_unlabelled}"
    )
    print("No dataset labels were changed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

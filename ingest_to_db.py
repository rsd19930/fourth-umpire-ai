"""
Embed cricket rule chunks using Voyage AI and load them into ChromaDB.
Respects Voyage free-tier rate limits: 3 RPM, 10,000 TPM.

Usage:
    python ingest_to_db.py              # Load both broader and finer
    python ingest_to_db.py --broader    # Load only broader chunks
    python ingest_to_db.py --finer      # Load only finer chunks
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
import voyageai
import chromadb

load_dotenv()

CHROMA_PATH = "chroma_storage"
BATCH_SIZE = 20  # ~1,600 tokens per batch — conservative to stay under 10K TPM
WAIT_SECONDS = 25  # Stay well under 3 RPM limit

TARGETS = {
    "broader": {
        "json_file": "real_cricket_rules_broader_chunking.json",
        "collection": "mcc_rules_broader",
    },
    "finer": {
        "json_file": "real_cricket_rules_finer_chunking.json",
        "collection": "mcc_rules_finer",
    },
}


def load_chunks(json_file):
    """Read chunks from JSON and return ids, texts, and metadatas."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    texts = []
    metadatas = []

    for chunk in data["chunks"]:
        ids.append(chunk["id"])
        texts.append(chunk["text"])
        metadatas.append({
            "law_number": chunk["law_number"] if chunk["law_number"] is not None else -1,
            "law_title": chunk["law_title"] or "",
            "section": chunk["section"] or "",
            "section_title": chunk["section_title"] or "",
        })

    return ids, texts, metadatas


def call_voyage_with_retry(voyage_client, texts, max_retries=5):
    """Call Voyage AI embed with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return voyage_client.embed(texts, model="voyage-4-lite")
        except Exception as e:
            if "RateLimitError" in type(e).__name__ or "rate" in str(e).lower():
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s...
                print(f"  Rate limited. Waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded for Voyage API rate limit.")


def embed_and_load(voyage_client, chroma_collection, ids, texts, metadatas):
    """Embed texts in batches and insert into ChromaDB."""
    total = len(texts)
    loaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]

        # Generate embeddings via Voyage AI (with retry on rate limit)
        result = call_voyage_with_retry(voyage_client, batch_texts)
        batch_embeddings = result.embeddings

        # Insert into ChromaDB
        chroma_collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

        loaded += len(batch_ids)
        print(f"  Batch {i // BATCH_SIZE + 1}: added {len(batch_ids)} chunks ({loaded}/{total})")

        # Wait between batches to respect rate limits (skip after last batch)
        if loaded < total:
            print(f"  Waiting {WAIT_SECONDS}s for rate limit...")
            time.sleep(WAIT_SECONDS)

    return loaded


def main():
    # Determine which targets to load
    if "--broader" in sys.argv:
        keys = ["broader"]
    elif "--finer" in sys.argv:
        keys = ["finer"]
    else:
        keys = ["broader", "finer"]

    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    for key in keys:
        target = TARGETS[key]
        print(f"\n{'='*50}")
        print(f"Loading: {key} chunks")
        print(f"  Source: {target['json_file']}")
        print(f"  Collection: {target['collection']}")
        print(f"{'='*50}")

        ids, texts, metadatas = load_chunks(target["json_file"])
        print(f"  Read {len(ids)} chunks from JSON")

        collection = chroma_client.get_collection(name=target["collection"])
        loaded = embed_and_load(voyage_client, collection, ids, texts, metadatas)

        final_count = collection.count()
        print(f"\n  Done! Collection '{target['collection']}' now has {final_count} documents.")

    print(f"\nAll done!")


if __name__ == "__main__":
    main()

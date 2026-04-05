"""
Embed cricket rule chunks using Voyage AI and load them into ChromaDB.
Respects Voyage free-tier rate limits: 3 RPM, 10,000 TPM.

Smart re-embedding: only embeds new or changed chunks. Skips unchanged ones.
Use --fresh to force re-embedding everything.

Usage:
    python ingest_to_db.py              # Load both (smart — skips unchanged)
    python ingest_to_db.py --broader    # Load only broader chunks
    python ingest_to_db.py --finer      # Load only finer chunks
    python ingest_to_db.py --fresh      # Force re-embed everything
"""

import os
import sys
import json
import time
import hashlib
from dotenv import load_dotenv
import voyageai
import chromadb

from config import CHROMA_PATH, COLLECTIONS, EMBEDDING_MODEL, BATCH_SIZE, WAIT_SECONDS

load_dotenv()


def text_hash(text):
    """Generate a short hash of text to detect changes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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
            "text_hash": text_hash(chunk["text"]),
        })

    return ids, texts, metadatas


def find_chunks_to_embed(collection, ids, texts, metadatas):
    """Compare against existing data and return only new/changed chunks."""
    existing_count = collection.count()
    if existing_count == 0:
        return ids, texts, metadatas

    # Fetch existing chunks in batches (ChromaDB limits get() size)
    existing_hashes = {}
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        try:
            result = collection.get(ids=batch_ids, include=["metadatas"])
            for j, eid in enumerate(result["ids"]):
                meta = result["metadatas"][j]
                existing_hashes[eid] = meta.get("text_hash", None)
        except Exception:
            # If IDs don't exist yet, they'll need embedding
            pass

    # Filter to only new or changed chunks
    new_ids = []
    new_texts = []
    new_metas = []
    backfill_ids = []
    backfill_metas = []

    for i, chunk_id in enumerate(ids):
        current_hash = metadatas[i]["text_hash"]
        if chunk_id not in existing_hashes:
            # New chunk — needs embedding
            new_ids.append(chunk_id)
            new_texts.append(texts[i])
            new_metas.append(metadatas[i])
        elif existing_hashes[chunk_id] is None:
            # Exists but was loaded before hash tracking — backfill hash only
            backfill_ids.append(chunk_id)
            backfill_metas.append(metadatas[i])
        elif existing_hashes[chunk_id] != current_hash:
            # Text changed — needs re-embedding
            new_ids.append(chunk_id)
            new_texts.append(texts[i])
            new_metas.append(metadatas[i])
        # else: unchanged — skip

    # Backfill hashes for chunks loaded before this feature (no re-embedding needed)
    if backfill_ids:
        for i in range(0, len(backfill_ids), batch_size):
            batch_ids = backfill_ids[i : i + batch_size]
            batch_metas = backfill_metas[i : i + batch_size]
            collection.update(ids=batch_ids, metadatas=batch_metas)
        print(f"  Backfilled text_hash for {len(backfill_ids)} existing chunks (no re-embedding).")

    return new_ids, new_texts, new_metas


def call_voyage_with_retry(voyage_client, texts, max_retries=5):
    """Call Voyage AI embed with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return voyage_client.embed(texts, model=EMBEDDING_MODEL)
        except Exception as e:
            if "RateLimitError" in type(e).__name__ or "rate" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded for Voyage API rate limit.")


def embed_and_load(voyage_client, collection, ids, texts, metadatas):
    """Embed texts in batches and upsert into ChromaDB."""
    total = len(texts)
    loaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]

        result = call_voyage_with_retry(voyage_client, batch_texts)
        batch_embeddings = result.embeddings

        # Use upsert so changed chunks get updated instead of erroring
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

        loaded += len(batch_ids)
        print(f"  Batch {i // BATCH_SIZE + 1}: embedded {len(batch_ids)} chunks ({loaded}/{total})")

        if loaded < total:
            print(f"  Waiting {WAIT_SECONDS}s for rate limit...")
            time.sleep(WAIT_SECONDS)

    return loaded


def main():
    fresh = "--fresh" in sys.argv

    if "--broader" in sys.argv:
        keys = ["broader"]
    elif "--finer" in sys.argv:
        keys = ["finer"]
    else:
        keys = ["broader", "finer"]

    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    for key in keys:
        target = COLLECTIONS[key]
        print(f"\n{'='*50}")
        print(f"Processing: {key} chunks")
        print(f"  Source: {target['json_file']}")
        print(f"  Collection: {target['name']}")
        print(f"{'='*50}")

        # Ensure collection exists
        try:
            collection = chroma_client.get_collection(name=target["name"])
        except Exception:
            collection = chroma_client.create_collection(name=target["name"])
            print(f"  Created collection '{target['name']}'.")

        ids, texts, metadatas = load_chunks(target["json_file"])
        print(f"  Read {len(ids)} chunks from JSON")

        if fresh:
            print("  --fresh flag: re-embedding all chunks")
            embed_ids, embed_texts, embed_metas = ids, texts, metadatas
        else:
            embed_ids, embed_texts, embed_metas = find_chunks_to_embed(
                collection, ids, texts, metadatas
            )

        if len(embed_ids) == 0:
            print(f"  All {len(ids)} chunks up to date. Nothing to embed!")
        else:
            print(f"  {len(embed_ids)} chunks to embed ({len(ids) - len(embed_ids)} unchanged, skipped)")
            embed_and_load(voyage_client, collection, embed_ids, embed_texts, embed_metas)

        final_count = collection.count()
        print(f"\n  Collection '{target['name']}' has {final_count} documents.")

    print(f"\nAll done!")


if __name__ == "__main__":
    main()

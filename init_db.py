"""
Initialize ChromaDB vector database for the Fourth Umpire AI.
Creates empty collections for comparing chunking strategies.
Safe to re-run — it resets all collections each time.
"""

import chromadb
from config import CHROMA_PATH, COLLECTIONS


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in client.list_collections()]
    collection_names = [c["name"] for c in COLLECTIONS.values()]

    for name in collection_names:
        if name in existing:
            client.delete_collection(name)
            print(f"Deleted existing '{name}' collection.")
        client.create_collection(name=name)
        print(f"Created collection '{name}'.")

    print(f"\nDatabase ready at: {CHROMA_PATH}/")
    print(f"Collections: {', '.join(collection_names)}")
    print("\nNext step: run ingest_to_db.py to load data.")


if __name__ == "__main__":
    main()

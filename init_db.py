"""
Initialize ChromaDB vector database for the Fourth Umpire AI.
Creates two empty collections for comparing chunking strategies.
Safe to re-run — it resets both collections each time.
"""

import chromadb

CHROMA_PATH = "chroma_storage"
COLLECTIONS = ["mcc_rules_broader", "mcc_rules_finer"]


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in client.list_collections()]

    for name in COLLECTIONS:
        if name in existing:
            client.delete_collection(name)
            print(f"Deleted existing '{name}' collection.")
        client.create_collection(name=name)
        print(f"Created collection '{name}'.")

    print(f"\nDatabase ready at: {CHROMA_PATH}/")
    print(f"Collections: {', '.join(COLLECTIONS)}")
    print("\nNext step: run ingest_to_db.py to load data.")


if __name__ == "__main__":
    main()

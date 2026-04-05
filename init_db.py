"""
Initialize ChromaDB vector database for the Fourth Umpire AI.
Creates collections for comparing chunking strategies.

Default: preserves existing collections (safe for code refactors).
Use --fresh to delete and recreate everything from scratch.
"""

import sys
import chromadb
from config import CHROMA_PATH, COLLECTIONS


def main():
    fresh = "--fresh" in sys.argv
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in client.list_collections()]
    collection_names = [c["name"] for c in COLLECTIONS.values()]

    for name in collection_names:
        if name in existing:
            if fresh:
                client.delete_collection(name)
                print(f"Deleted existing '{name}' collection.")
                client.create_collection(name=name)
                print(f"Created collection '{name}'.")
            else:
                count = client.get_collection(name).count()
                print(f"Collection '{name}' already exists ({count} documents). Skipping.")
        else:
            client.create_collection(name=name)
            print(f"Created collection '{name}'.")

    print(f"\nDatabase ready at: {CHROMA_PATH}/")
    if not fresh:
        print("Tip: use --fresh to delete and recreate all collections.")


if __name__ == "__main__":
    main()

"""
Fourth Umpire AI — RAG Query Pipeline

Retrieves relevant cricket rules from ChromaDB and generates
an authoritative ruling using Claude.

Usage:
    python query.py                  # Query both collections (default)
    python query.py --finer          # Query finer collection only
    python query.py --broader        # Query broader collection only
"""

import os
import sys
from dotenv import load_dotenv
import voyageai
import chromadb
from anthropic import Anthropic

from config import (
    CHROMA_PATH, COLLECTIONS, EMBEDDING_MODEL, LLM_MODEL,
    TOP_K, PROMPTS_DIR, DEFAULT_QUESTION,
)

load_dotenv(override=True)


def load_system_prompt():
    """Load the system prompt from prompts/system.md."""
    prompt_path = os.path.join(PROMPTS_DIR, "system.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_collections(chroma_client, mode):
    """Return list of (label, collection) tuples based on mode."""
    result = []
    if mode in ("finer", "both"):
        result.append(("finer", chroma_client.get_collection(COLLECTIONS["finer"]["name"])))
    if mode in ("broader", "both"):
        result.append(("broader", chroma_client.get_collection(COLLECTIONS["broader"]["name"])))
    return result


def embed_question(voyage_client, question):
    """Embed a question using Voyage AI. Call once, reuse for all collections."""
    result = voyage_client.embed([question], model=EMBEDDING_MODEL)
    return result.embeddings[0]


def retrieve(collection, query_embedding):
    """Query ChromaDB for top K results using a pre-computed embedding."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "law_number": meta.get("law_number"),
            "law_title": meta.get("law_title", ""),
            "section": meta.get("section", ""),
            "section_title": meta.get("section_title", ""),
        })
    return chunks


def format_chunk_label(chunk):
    """Format a chunk's law/section info for display."""
    law_num = chunk["law_number"]
    section = chunk["section"]
    section_title = chunk["section_title"]

    if law_num and law_num != -1:
        label = f"Law {law_num}"
        if section:
            label += f", Section {section}"
        if section_title:
            label += f" — {section_title}"
        label += f"  ({chunk['law_title']})"
    else:
        label = chunk["law_title"]
        if section:
            label += f" | {section}"
    return label


def build_context(all_chunks):
    """Build the context string from all retrieved chunks (deduplicated)."""
    seen = set()
    context_parts = []
    for chunk in all_chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            label = format_chunk_label(chunk)
            context_parts.append(f"[{label}]\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def generate_ruling(anthropic_client, system_prompt, question, context):
    """Send question + context to Claude and return the ruling."""
    message = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt.format(context=context),
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


def print_divider(title):
    print(f"\n── {title} {'─' * max(1, 50 - len(title))}")


def main():
    if "--finer" in sys.argv:
        mode = "finer"
    elif "--broader" in sys.argv:
        mode = "broader"
    else:
        mode = "both"

    system_prompt = load_system_prompt()
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    anthropic_client = Anthropic()
    collections = get_collections(chroma_client, mode)

    print("=" * 55)
    print("  FOURTH UMPIRE AI")
    print("  Ask any cricket rules question. Type 'quit' to exit.")
    print(f"  Mode: {mode} | Top {TOP_K} results per collection")
    print("=" * 55)

    while True:
        print(f"\nDefault: {DEFAULT_QUESTION}")
        user_input = input("\nYour question (press Enter for default): ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nStumps drawn. Good day!")
            break

        question = user_input if user_input else DEFAULT_QUESTION
        print_divider("Question")
        print(f"  {question}")

        query_embedding = embed_question(voyage_client, question)

        all_chunks = []
        for label, collection in collections:
            chunks = retrieve(collection, query_embedding)
            all_chunks.extend(chunks)

            print_divider(f"Retrieved Rules ({label})")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {format_chunk_label(chunk)}")

        context = build_context(all_chunks)
        print_divider("Fourth Umpire's Ruling")
        ruling = generate_ruling(anthropic_client, system_prompt, question, context)
        print(f"\n{ruling}")
        print("\n" + "─" * 55)


if __name__ == "__main__":
    main()

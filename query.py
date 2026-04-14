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
    RETRIEVAL_K, RERANK_K, RERANK_MODEL, PROMPTS_DIR, DEFAULT_QUESTION,
)
from tools import TOOL_DEFINITIONS, execute_tool
from query_expansion import expand_query

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


def retrieve(collection, query_embedding, n_results=RETRIEVAL_K):
    """Query ChromaDB for top results using a pre-computed embedding."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
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


def rerank_chunks(voyage_client, question, chunks, top_n=RERANK_K):
    """Rerank retrieved chunks using Voyage rerank model. Returns top_n best matches."""
    if not chunks or len(chunks) <= top_n:
        return chunks
    documents = [c["text"] for c in chunks]
    result = voyage_client.rerank(question, documents, model=RERANK_MODEL, top_k=top_n)
    reranked = [chunks[r.index] for r in result.results]
    return reranked


def hybrid_retrieve(voyage_client, question, chunks, cosine_k=RERANK_K, rerank_k=RERANK_K):
    """Union of cosine top-K and reranked top-K, deduplicated by chunk ID.

    Ensures neither cosine similarity nor the reranker can veto a chunk
    the other method ranked highly. Returns 5-10 unique chunks.
    """
    cosine_top = chunks[:cosine_k]
    reranked_top = rerank_chunks(voyage_client, question, chunks, top_n=rerank_k)

    # Union, preserving order: cosine first, then reranked additions
    seen = set()
    result = []
    for c in cosine_top + reranked_top:
        if c["id"] not in seen:
            seen.add(c["id"])
            result.append(c)
    return result


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


MAX_TOOL_ITERATIONS = 10


def _extract_text(response):
    """Safely extract text from a Claude response, handling empty/missing content."""
    if not response.content:
        return ""
    text_parts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(text_parts)


def _return_ruling(text, total_input_tokens, total_output_tokens, return_usage):
    """Helper to return ruling text with optional usage tracking."""
    if return_usage:
        return text, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
    return text


def _call_without_tools(anthropic_client, system, messages, verbose=False):
    """Fallback: call Claude without tools when the tools+context combination causes issues."""
    if verbose:
        print("  [Retrying without tools]")
    response = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=2048,
        system=system,
        messages=[messages[0]],  # Only send original user message
    )
    return response


def generate_ruling(anthropic_client, system_prompt, question, context, return_usage=False, verbose=False):
    """Send question + context to Claude, handling tool calls in an agentic loop."""
    messages = [{"role": "user", "content": question}]
    system = system_prompt.format(context=context)

    # Accumulate tokens across loop iterations
    total_input_tokens = 0
    total_output_tokens = 0
    last_response = None
    empty_content_retries = 0

    for _ in range(MAX_TOOL_ITERATIONS):
        response = anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        last_response = response

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Guard: empty content with tool_use stop_reason is a known API edge case.
        # Retry once without tools — this reliably produces a valid response.
        if response.stop_reason == "tool_use" and not response.content:
            empty_content_retries += 1
            if verbose:
                print(f"  [Empty content with tool_use stop_reason — retry #{empty_content_retries}]")
            if empty_content_retries <= 1:
                fallback = _call_without_tools(anthropic_client, system, messages, verbose)
                total_input_tokens += fallback.usage.input_tokens
                total_output_tokens += fallback.usage.output_tokens
                return _return_ruling(
                    _extract_text(fallback),
                    total_input_tokens, total_output_tokens, return_usage,
                )
            # If retry also fails, return empty
            return _return_ruling("", total_input_tokens, total_output_tokens, return_usage)

        # If Claude wants to use tools, execute them and continue the loop
        if response.stop_reason == "tool_use":
            # Execute each tool call and build tool_result messages
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    if verbose:
                        print(f"  [Tool: {block.name}({block.input})]  →  {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Guard: content blocks exist but none are tool_use — treat as done
            if not tool_results:
                return _return_ruling(
                    _extract_text(response),
                    total_input_tokens, total_output_tokens, return_usage,
                )

            # Add Claude's response and tool results to messages, then loop
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop_reason (end_turn, max_tokens, stop_sequence, etc.)
        # → extract text and return
        return _return_ruling(
            _extract_text(response),
            total_input_tokens, total_output_tokens, return_usage,
        )

    # Safety: if we hit max iterations, return whatever text we have
    ruling_text = _extract_text(last_response) if last_response else "Error: max tool iterations reached."
    return _return_ruling(ruling_text, total_input_tokens, total_output_tokens, return_usage)


def generate_ruling_stream(anthropic_client, system_prompt, question, context):
    """Stream a ruling token-by-token, handling tool calls in an agentic loop.

    Yields text chunks (str) as they arrive from Claude.
    Tool calls are executed between loop iterations — the user sees
    a status message like "[Using calculator(...)]" yielded inline.

    Designed for Streamlit's st.write_stream().
    The existing generate_ruling() stays for CLI and eval use.
    """
    messages = [{"role": "user", "content": question}]
    system = system_prompt.format(context=context)
    empty_content_retries = 0

    for _ in range(MAX_TOOL_ITERATIONS):
        with anthropic_client.messages.stream(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        ) as stream:
            # Stream text tokens to caller
            for event in stream:
                if (event.type == "content_block_delta"
                        and event.delta.type == "text_delta"):
                    yield event.delta.text

            # Get the complete response for tool handling
            response = stream.get_final_message()

        # Empty content edge case — fall back to non-streaming without tools
        if response.stop_reason == "tool_use" and not response.content:
            empty_content_retries += 1
            if empty_content_retries <= 1:
                fallback = _call_without_tools(anthropic_client, system, messages)
                yield _extract_text(fallback)
                return
            return

        # If Claude wants to use tools, execute them and continue
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    # Yield a visible status so user knows a tool was called
                    yield f"\n\n*Using {block.name}({block.input}) → {result}*\n\n"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if not tool_results:
                return

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn or any other stop_reason — we're done
        return


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
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"), max_retries=3)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    anthropic_client = Anthropic()
    collections = get_collections(chroma_client, mode)

    print("=" * 55)
    print("  FOURTH UMPIRE AI")
    print("  Ask any cricket rules question. Type 'quit' to exit.")
    print(f"  Mode: {mode} | Retrieve {RETRIEVAL_K} → Rerank to {RERANK_K}")
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

        # Expand query for better retrieval (formal MCC terminology)
        expanded_query, _ = expand_query(anthropic_client, question, verbose=True)
        query_embedding = embed_question(voyage_client, expanded_query)

        all_chunks = []
        for label, collection in collections:
            chunks = retrieve(collection, query_embedding)

            print_divider(f"Retrieved Rules ({label}) — {len(chunks)} candidates")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {format_chunk_label(chunk)}")

            # Hybrid retrieve: union of cosine top-5 + reranked top-5
            chunks = hybrid_retrieve(voyage_client, question, chunks)

            print_divider(f"After Hybrid Retrieval ({label}) — {len(chunks)} unique")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {format_chunk_label(chunk)}")

            all_chunks.extend(chunks)

        context = build_context(all_chunks)
        print_divider("Fourth Umpire's Ruling")
        ruling = generate_ruling(anthropic_client, system_prompt, question, context, verbose=True)
        print(f"\n{ruling}")
        print("\n" + "─" * 55)


if __name__ == "__main__":
    main()

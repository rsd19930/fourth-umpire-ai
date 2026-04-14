#!/usr/bin/env python3
"""
Fourth Umpire AI — Evaluation Runner

Runs the golden dataset through the full RAG pipeline and computes:
  1. Answer Relevance  (LLM judge, thinking mode)
  2. Faithfulness       (LLM judge, thinking mode)
  3. Context Recall     (set math)
  4. Context Precision  (set math)

Reports finer and broader chunk results SEPARATELY — never averaged.

Usage:
    python3 run_eval.py --note "Baseline run"
    python3 run_eval.py --finer --note "Testing finer only"
    python3 run_eval.py --broader --note "Testing broader only"
"""

import os
import sys
import json
import time
import re
import argparse
from datetime import datetime
from dotenv import load_dotenv
import voyageai
import chromadb
from anthropic import Anthropic

from config import (
    CHROMA_PATH, COLLECTIONS, EMBEDDING_MODEL, LLM_MODEL,
    EXPANSION_MODEL, RETRIEVAL_K, RERANK_K, RERANK_MODEL,
    BATCH_SIZE, WAIT_SECONDS,
)
from query import (
    load_system_prompt, get_collections, retrieve,
    rerank_chunks, hybrid_retrieve, build_context, generate_ruling,
)
from query_expansion import batch_expand_questions

load_dotenv(override=True)

DEFAULT_DATASET = "evals/golden_dataset.json"
RESULTS_DIR = "evals/results"
JUDGE_MODEL = "claude-haiku-4-5-20251001"


# ═══════════════════════════════════════════════════════════
# LLM JUDGE
# ═══════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are evaluating whether an AI cricket umpire's response is faithful to the retrieved context — meaning it only uses information present in the context.

## Retrieved Context
{context}

## AI Response
{response}

## Scoring
- 1.0 = FULLY FAITHFUL: Every claim is grounded in the context
- 0.5 = PARTIALLY FAITHFUL: Core ruling is grounded but some details are not in context
- 0.0 = NOT FAITHFUL: Introduces significant facts not in context or contradicts it

Respond in exactly this format:
SCORE: <1.0 or 0.5 or 0.0>
JUSTIFICATION: <1-2 sentences>"""

ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an AI cricket umpire gave the correct ruling.

## Question
{question}

## Expected Ruling
{expected_ruling}

## AI Response
{response}

## Scoring
- 1.0 = CORRECT: Ruling matches expected in substance (wording can differ)
- 0.5 = PARTIALLY CORRECT: Right direction but misses key details or cites wrong laws
- 0.0 = INCORRECT: Ruling contradicts expected ruling or fails to answer

Respond in exactly this format:
SCORE: <1.0 or 0.5 or 0.0>
JUSTIFICATION: <1-2 sentences>"""


def call_judge(client, prompt, max_retries=3):
    """Call Claude as LLM-as-judge to score a response. Returns (result, usage)."""
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens}
            return parse_judge_response(message.content[0].text), usage
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 5 * (2 ** attempt)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {"score": 0.0, "justification": f"API error: {e}"}, {"input_tokens": 0, "output_tokens": 0}
    return {"score": 0.0, "justification": "Failed after retries"}, {"input_tokens": 0, "output_tokens": 0}


def parse_judge_response(text):
    """Parse SCORE and JUSTIFICATION from judge response."""
    score = 0.0
    justification = text.strip()

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            score_str = line.split(":", 1)[1].strip()
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
        elif line.upper().startswith("JUSTIFICATION:"):
            justification = line.split(":", 1)[1].strip()

    return {"score": score, "justification": justification}


# ═══════════════════════════════════════════════════════════
# RETRIEVAL METRICS (set math)
# ═══════════════════════════════════════════════════════════

def compute_context_recall(relevant_ids, retrieved_ids):
    """What fraction of relevant chunks were retrieved?"""
    if not relevant_ids:
        return None  # Can't compute without ground truth
    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids)
    return len(relevant & retrieved) / len(relevant)


def compute_context_precision(relevant_ids, retrieved_ids):
    """What fraction of retrieved chunks were relevant?"""
    if not retrieved_ids:
        return None
    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids)
    return len(relevant & retrieved) / len(retrieved)


# ═══════════════════════════════════════════════════════════
# BATCH EMBEDDING
# ═══════════════════════════════════════════════════════════

def batch_embed_questions(voyage_client, questions):
    """Embed all questions in batches, respecting Voyage AI rate limits."""
    texts = [q["question"] for q in questions]
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = voyage_client.embed(batch, model=EMBEDDING_MODEL)
        all_embeddings.extend(result.embeddings)

        done = min(i + BATCH_SIZE, len(texts))
        print(f"  Embedded {done}/{len(texts)} questions")

        if done < len(texts):
            print(f"  Waiting {WAIT_SECONDS}s for rate limit...")
            time.sleep(WAIT_SECONDS)

    return all_embeddings


# ═══════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ═══════════════════════════════════════════════════════════

def compute_summary(per_question_results, questions, collection_key):
    """Compute aggregate metrics for a single collection."""
    metrics = ["answer_relevance", "faithfulness", "context_recall", "context_precision"]

    def avg(values):
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 3) if valid else None

    def extract_scores(results, metric):
        scores = []
        for r in results:
            val = r.get(collection_key, {}).get(metric)
            if isinstance(val, dict):
                scores.append(val.get("score"))
            else:
                scores.append(val)
        return scores

    # Overall
    overall = {}
    for m in metrics:
        overall[m] = avg(extract_scores(per_question_results, m))

    # By difficulty
    by_difficulty = {}
    for diff in ["easy", "hard", "edge_case"]:
        indices = [i for i, q in enumerate(questions) if q["difficulty"] == diff]
        subset = [per_question_results[i] for i in indices]
        if subset:
            entry = {"count": len(subset)}
            for m in metrics:
                entry[m] = avg(extract_scores(subset, m))
            by_difficulty[diff] = entry

    # By source
    by_source = {}
    for src in ["bcci_exam", "blog", "llm_generated", "scraped"]:
        indices = [i for i, q in enumerate(questions) if q["source"] == src]
        subset = [per_question_results[i] for i in indices]
        if subset:
            entry = {"count": len(subset)}
            for m in metrics:
                entry[m] = avg(extract_scores(subset, m))
            by_source[src] = entry

    return {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_source": by_source,
    }


# ═══════════════════════════════════════════════════════════
# SCORECARD & README
# ═══════════════════════════════════════════════════════════

def print_scorecard(summary, collections_used):
    """Print formatted scorecard to terminal."""
    print("\n" + "=" * 55)
    print("  SCORECARD")
    print("=" * 55)

    for key in collections_used:
        label = "FINER (717 chunks)" if key == "finer" else "BROADER (279 chunks)"
        s = summary[key]["overall"]
        print(f"\n  {label}:")
        print(f"    Answer Relevance:  {s.get('answer_relevance', 'N/A')}")
        print(f"    Faithfulness:      {s.get('faithfulness', 'N/A')}")
        print(f"    Context Recall:    {s.get('context_recall', 'N/A')}")
        print(f"    Context Precision: {s.get('context_precision', 'N/A')}")

        if summary[key].get("by_difficulty"):
            print(f"\n    By Difficulty:")
            for diff, vals in summary[key]["by_difficulty"].items():
                ar = vals.get("answer_relevance", "N/A")
                ff = vals.get("faithfulness", "N/A")
                cr = vals.get("context_recall", "N/A")
                cp = vals.get("context_precision", "N/A")
                print(f"      {diff:10s} (n={vals['count']:2d}): AR={ar}  F={ff}  CR={cr}  CP={cp}")

    print()


def fmt(val):
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def append_to_readme(summary, run_config, collections_used):
    """Append eval run summary to README.md."""
    readme_path = "README.md"
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Add Eval Runs section if it doesn't exist
    if "## Eval Runs" not in content:
        content += "\n\n## Eval Runs\n"

    date_str = run_config["timestamp"][:10]
    note = run_config.get("note", "No description provided")

    block = f"\n### Eval Run — {date_str}\n"

    for key in collections_used:
        label = "Finer Chunks (717)" if key == "finer" else "Broader Chunks (279)"
        s = summary[key]
        overall = s["overall"]

        block += f"\n**{label}:**\n\n"
        block += "| | Answer Relevance | Faithfulness | Context Recall | Context Precision |\n"
        block += "|---|---|---|---|---|\n"
        block += (
            f"| **Overall** | {fmt(overall.get('answer_relevance'))} "
            f"| {fmt(overall.get('faithfulness'))} "
            f"| {fmt(overall.get('context_recall'))} "
            f"| {fmt(overall.get('context_precision'))} |\n"
        )

        for diff in ["easy", "hard", "edge_case"]:
            if diff in s.get("by_difficulty", {}):
                d = s["by_difficulty"][diff]
                label_d = diff.replace("_", " ").title()
                block += (
                    f"| {label_d} ({d['count']}) "
                    f"| {fmt(d.get('answer_relevance'))} "
                    f"| {fmt(d.get('faithfulness'))} "
                    f"| {fmt(d.get('context_recall'))} "
                    f"| {fmt(d.get('context_precision'))} |\n"
                )

    block += (
        f"\n**Config:** Retrieve {run_config['retrieval_k']} → Rerank to {run_config['rerank_k']} ({run_config['rerank_model']}), "
        f"model={run_config['llm_model']}, "
        f"judge={run_config['judge_model']}\n"
    )

    # Runtime and token info
    elapsed = run_config.get("elapsed_seconds", 0)
    minutes, seconds = divmod(elapsed, 60)
    tokens = run_config.get("tokens", {})
    total_input_k = round(tokens.get("total_input", 0) / 1000, 1)
    total_output_k = round(tokens.get("total_output", 0) / 1000, 1)
    block += f"**Runtime:** {minutes}m {seconds}s | **Tokens:** {total_input_k}K input + {total_output_k}K output\n"

    block += f"**What changed:** {note}\n"
    block += f"**Results file:** `{run_config['results_file']}`\n"

    content += block

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  README.md updated with eval summary.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fourth Umpire AI — Evaluation Runner")
    parser.add_argument("--finer", action="store_true", help="Evaluate finer collection only")
    parser.add_argument("--broader", action="store_true", help="Evaluate broader collection only")
    parser.add_argument("--both", action="store_true", help="Evaluate both finer and broader collections")
    parser.add_argument("--note", type=str, default="No description provided",
                        help="Document what changed for this run")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Path to golden dataset JSON")
    args = parser.parse_args()

    if args.both:
        collections_to_eval = ["finer", "broader"]
    elif args.finer:
        collections_to_eval = ["finer"]
    else:
        collections_to_eval = ["broader"]

    # ── Setup ──
    print("=" * 55)
    print("  FOURTH UMPIRE AI — Evaluation Runner")
    print("=" * 55)

    anthropic_client = Anthropic()
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"), max_retries=3)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    system_prompt = load_system_prompt()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]

    print(f"  Mode: {'+'.join(collections_to_eval)} | Retrieve {RETRIEVAL_K} → Rerank to {RERANK_K} | Questions: {len(questions)}")
    print(f"  Note: {args.note}")

    # Get collection objects
    chroma_collections = {}
    for key in collections_to_eval:
        chroma_collections[key] = chroma_client.get_collection(COLLECTIONS[key]["name"])

    start_time = time.time()

    # ── Step 1: Expand questions (formal MCC terminology) ──
    print(f"\n── Step 1: Expanding Questions {'─' * 30}")
    expanded_texts, expansion_usage = batch_expand_questions(anthropic_client, questions)

    # ── Step 2: Embed expanded questions ──
    print(f"\n── Step 2: Embedding Questions {'─' * 30}")
    expanded_question_dicts = [{"question": text} for text in expanded_texts]
    embeddings = batch_embed_questions(voyage_client, expanded_question_dicts)

    # ── Step 3: Retrieve & Rerank chunks ──
    # Rerank API has same rate limits as embeddings (3 RPM free tier)
    # Process in batches of BATCH_SIZE with WAIT_SECONDS delays
    print(f"\n── Step 3: Retrieving & Reranking Chunks {'─' * 20}")
    retrieval_results = {}
    for key in collections_to_eval:
        retrieval_results[key] = []
        for i, (q, emb) in enumerate(zip(questions, embeddings)):
            # Retrieve wider pool (local ChromaDB — no rate limit)
            chunks = retrieve(chroma_collections[key], emb)
            # Hybrid retrieve: union of cosine top-5 + reranked top-5 (Voyage API — rate limited)
            try:
                chunks = hybrid_retrieve(voyage_client, q["question"], chunks)
            except Exception as e:
                # On rate limit or other error, keep original top-K without reranking
                print(f"    Hybrid retrieve failed for {q['id']}, using cosine top chunks: {e}")
                chunks = chunks[:RERANK_K]
            retrieval_results[key].append({
                "chunks": chunks,
                "retrieved_ids": [c["id"] for c in chunks],
                "context": build_context(chunks),
            })

            done = i + 1
            if done % BATCH_SIZE == 0 or done == len(questions):
                print(f"  {key}: retrieved & reranked {done}/{len(questions)} questions")
            # Rate limit: pause after every rerank call (3 RPM limit → 25s gap = ~2.4 RPM)
            if done < len(questions):
                time.sleep(WAIT_SECONDS)
        print(f"  {key}: done")

    # ── Token tracking ──
    token_counts = {
        "expansion_input": expansion_usage["input_tokens"],
        "expansion_output": expansion_usage["output_tokens"],
        "ruling_input": 0, "ruling_output": 0,
        "judge_input": 0, "judge_output": 0,
    }

    # ── Step 4: Generate rulings ──
    print(f"\n── Step 4: Generating Rulings {'─' * 30}")
    rulings = {}
    for key in collections_to_eval:
        rulings[key] = []
        for i, q in enumerate(questions):
            context = retrieval_results[key][i]["context"]
            try:
                ruling, usage = generate_ruling(anthropic_client, system_prompt, q["question"], context, return_usage=True)
            except Exception as e:
                print(f"    ERROR generating ruling for {q['id']}: {e}")
                ruling = f"Error: {e}"
                usage = {"input_tokens": 0, "output_tokens": 0}
            rulings[key].append(ruling)
            token_counts["ruling_input"] += usage["input_tokens"]
            token_counts["ruling_output"] += usage["output_tokens"]
            print(f"  [{key}] Ruling {i + 1}/{len(questions)} ({q['id']})")
            time.sleep(1)

    # ── Step 5: Context Recall & Precision ──
    print(f"\n── Step 5: Context Recall & Precision {'─' * 20}")
    # Initialize per-question results
    per_question = []
    for i, q in enumerate(questions):
        result = {"id": q["id"], "expanded_question": expanded_texts[i]}
        for key in collections_to_eval:
            relevant_ids = q.get(f"relevant_chunk_ids_{key}", [])
            retrieved_ids = retrieval_results[key][i]["retrieved_ids"]
            result[key] = {
                "context_recall": compute_context_recall(relevant_ids, retrieved_ids),
                "context_precision": compute_context_precision(relevant_ids, retrieved_ids),
                "retrieved_chunk_ids": retrieved_ids,
            }
            result[f"ai_ruling_{key}"] = rulings[key][i]
        per_question.append(result)
    print("  Computed for all collections (instant)")

    # ── Step 6: Judge Faithfulness ──
    print(f"\n── Step 6: Judging Faithfulness {'─' * 25}")
    for key in collections_to_eval:
        for i, q in enumerate(questions):
            context = retrieval_results[key][i]["context"]
            response = rulings[key][i]
            prompt = FAITHFULNESS_PROMPT.format(context=context, response=response)
            result, usage = call_judge(anthropic_client, prompt)
            per_question[i][key]["faithfulness"] = result
            token_counts["judge_input"] += usage["input_tokens"]
            token_counts["judge_output"] += usage["output_tokens"]
            print(f"  [{key}] Judged {i + 1}/{len(questions)} ({q['id']}): {result['score']}")
            time.sleep(1)

    # ── Step 7: Judge Answer Relevance ──
    print(f"\n── Step 7: Judging Answer Relevance {'─' * 20}")
    for key in collections_to_eval:
        for i, q in enumerate(questions):
            response = rulings[key][i]
            prompt = ANSWER_RELEVANCE_PROMPT.format(
                question=q["question"],
                expected_ruling=q["expected_ruling"],
                response=response,
            )
            result, usage = call_judge(anthropic_client, prompt)
            per_question[i][key]["answer_relevance"] = result
            token_counts["judge_input"] += usage["input_tokens"]
            token_counts["judge_output"] += usage["output_tokens"]
            print(f"  [{key}] Judged {i + 1}/{len(questions)} ({q['id']}): {result['score']}")
            time.sleep(1)

    # ── Compute summary ──
    summary = {}
    for key in collections_to_eval:
        summary[key] = compute_summary(per_question, questions, key)

    # ── Save results ──
    elapsed_seconds = round(time.time() - start_time)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")

    total_input = token_counts["expansion_input"] + token_counts["ruling_input"] + token_counts["judge_input"]
    total_output = token_counts["expansion_output"] + token_counts["ruling_output"] + token_counts["judge_output"]

    run_config = {
        "timestamp": datetime.now().isoformat(),
        "mode": "+".join(collections_to_eval),
        "retrieval_k": RETRIEVAL_K,
        "rerank_k": RERANK_K,
        "rerank_model": RERANK_MODEL,
        "llm_model": LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "expansion_model": EXPANSION_MODEL,
        "dataset": args.dataset,
        "total_questions": len(questions),
        "note": args.note,
        "results_file": results_file,
        "elapsed_seconds": elapsed_seconds,
        "tokens": {
            "expansion_input": token_counts["expansion_input"],
            "expansion_output": token_counts["expansion_output"],
            "ruling_input": token_counts["ruling_input"],
            "ruling_output": token_counts["ruling_output"],
            "judge_input": token_counts["judge_input"],
            "judge_output": token_counts["judge_output"],
            "total_input": total_input,
            "total_output": total_output,
            "total": total_input + total_output,
        },
    }

    output = {
        "run_metadata": run_config,
        "summary": summary,
        "per_question": per_question,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Print scorecard ──
    print_scorecard(summary, collections_to_eval)

    print(f"  Results saved: {results_file}")

    # ── Append to README ──
    append_to_readme(summary, run_config, collections_to_eval)

    print("=" * 55)


if __name__ == "__main__":
    main()

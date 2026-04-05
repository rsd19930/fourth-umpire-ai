#!/usr/bin/env python3
"""
Generate Golden Evaluation Dataset for Fourth Umpire AI.

Collects ~50 cricket umpiring questions from 4 sources:
  1. BCCI Umpire Exam PDF (~5-7 MCC-relevant questions)
  2. Cricket Umpiring Blog PDF (~15-17 Q&A pairs)
  3. LLM-generated from MCC Laws (~18 questions)
  4. Reddit/web (~10 real-world debate questions)

For each question, retrieves candidate chunks from ChromaDB
and suggests relevant chunk IDs for evaluation.

Output: evals/golden_dataset_draft.json

Usage:
    python3 generate_eval_dataset.py           # All sources
    python3 generate_eval_dataset.py --skip-reddit  # Skip web scraping
"""

import os
import sys
import json
import time
import re
from dotenv import load_dotenv
from anthropic import Anthropic
import PyPDF2
import voyageai
import chromadb

from config import (
    CHROMA_PATH, COLLECTIONS, EMBEDDING_MODEL, LLM_MODEL,
    TOP_K, BATCH_SIZE, WAIT_SECONDS,
)

load_dotenv(override=True)

EVAL_DIR = "evals"
OUTPUT_FILE = os.path.join(EVAL_DIR, "golden_dataset_draft.json")


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def extract_pdf_text(path):
    """Extract all text from a PDF file."""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def call_claude(client, prompt, system="You are a helpful assistant.", max_tokens=8192):
    """Call Claude API and return text response."""
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def parse_json_from_response(text):
    """Extract JSON array from Claude's response (handles markdown code blocks)."""
    # Try code block first
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from response. First 200 chars: {text[:200]}")


def law_ref_to_chunk_id(law_ref, available_ids):
    """
    Convert a law reference like 'Law 36.1.2' to chunk ID format 'law36_s36.1.2'.
    Falls back to parent section if exact match not found.
    """
    # Extract law number and section from references like "Law 36.1.2" or "36.1.2"
    match = re.match(r"(?:Law\s*)?(\d+)\.?([\d.]*)", str(law_ref))
    if not match:
        return None

    law_num = match.group(1)
    section_rest = match.group(2)

    if section_rest:
        full_section = f"{law_num}.{section_rest}"
        chunk_id = f"law{law_num}_s{full_section}"
    else:
        # Just a law number, try the first section
        chunk_id = f"law{law_num}_s{law_num}.1"

    # Check if it exists
    if chunk_id in available_ids:
        return chunk_id

    # Try parent section (remove last .N)
    parts = full_section.rsplit(".", 1) if section_rest else []
    if len(parts) == 2:
        parent_id = f"law{law_num}_s{parts[0]}"
        if parent_id in available_ids:
            return parent_id

    # Try just lawN_sN.N (broadest)
    if "." in (section_rest or ""):
        top_section = f"{law_num}.{section_rest.split('.')[0]}"
        top_id = f"law{law_num}_s{top_section}"
        if top_id in available_ids:
            return top_id

    return chunk_id  # Return anyway, user will verify


def load_laws_toc():
    """Build a table of contents from the finer chunks JSON."""
    with open(COLLECTIONS["finer"]["json_file"], "r", encoding="utf-8") as f:
        data = json.load(f)

    toc = {}
    for chunk in data["chunks"]:
        law_num = chunk["law_number"]
        if law_num is None:
            continue
        try:
            law_num = int(law_num)
        except (ValueError, TypeError):
            continue
        if law_num < 1:
            continue

        if law_num not in toc:
            toc[law_num] = {"title": chunk["law_title"], "sections": []}
        section = chunk["section"]
        section_title = chunk["section_title"]
        if section and section_title:
            toc[law_num]["sections"].append(f"{section}: {section_title}")
    return toc


def load_all_chunk_ids():
    """Load sets of all chunk IDs from both JSON files."""
    result = {}
    for key in ["finer", "broader"]:
        with open(COLLECTIONS[key]["json_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
        result[key] = set(c["id"] for c in data["chunks"])
    return result


def normalize_question(q, all_chunk_ids):
    """Ensure a question dict has all required fields with correct types."""
    # Required string fields
    for field in ["question", "expected_ruling", "expected_ruling_type",
                   "difficulty", "category", "id", "source", "notes"]:
        q.setdefault(field, "")

    # Source URL
    q.setdefault("source_url", None)

    # Law numbers
    q.setdefault("primary_law", 0)
    q.setdefault("laws_involved", [])
    if isinstance(q["primary_law"], str):
        try:
            q["primary_law"] = int(q["primary_law"])
        except (ValueError, TypeError):
            q["primary_law"] = 0

    # Explanation keywords
    q.setdefault("expected_explanation_keywords", [])

    # Chunk IDs — convert law references to chunk ID format if needed
    for key_suffix in ["finer", "broader"]:
        field = f"relevant_chunk_ids_{key_suffix}"
        ids = all_chunk_ids[key_suffix]

        if field not in q:
            # Try to derive from law references
            chunk_ids = []
            for ref in q.get("expected_explanation_keywords", []):
                cid = law_ref_to_chunk_id(ref, ids)
                if cid:
                    chunk_ids.append(cid)
            q[field] = chunk_ids

        # Also check for LLM-generated fields and rename
        alt_field = f"relevant_sections_{key_suffix}"
        if alt_field in q:
            q[field] = q.pop(alt_field)

        q.setdefault(f"candidate_chunks_{key_suffix}", [])

    return q


# ═══════════════════════════════════════════════════════════
# SOURCE 1: BCCI UMPIRE EXAM
# ═══════════════════════════════════════════════════════════

SYSTEM_UMPIRE = (
    "You are an expert cricket umpire and MCC Laws of Cricket specialist. "
    "You have deep knowledge of all 42 MCC Laws."
)


def extract_bcci_questions(client):
    """Extract MCC Laws-relevant questions from BCCI exam PDF."""
    print("\n" + "=" * 55)
    print("  Source 1: BCCI Umpire Exam")
    print("=" * 55)

    pdf_path = "Umpire Exam Sheet with Answers.pdf"
    if not os.path.exists(pdf_path):
        print(f"  Skipping: {pdf_path} not found")
        return []

    pdf_text = extract_pdf_text(pdf_path)
    print(f"  Extracted {len(pdf_text)} chars from PDF")

    prompt = f"""Below is text from a BCCI Umpires' Examination 2025. Many questions reference BCCI Playing Conditions for ODMs and T20s, which are NOT part of the universal MCC Laws.

Extract ONLY questions that can be answered using the MCC Laws of Cricket alone. Skip any question that:
- References BCCI Playing Conditions, ODM rules, T20 powerplay/free hit rules
- Is about over limits, DLS/VJD, over-rate penalties specific to formats
- Cannot be answered from MCC Laws alone

For each relevant question, rewrite it as a clean cricket scenario (remove BCCI-specific framing) and provide a JSON array of objects with these exact fields:
- "question": string - the scenario (clear, self-contained)
- "expected_ruling": string - the ruling (e.g., "Fair delivery", "OUT - Run out", "NOT OUT")
- "expected_ruling_type": "definitive" or "guidance"
- "expected_explanation_keywords": array of strings - MCC Law references (e.g., ["Law 21.1", "Law 36.1"])
- "difficulty": "easy", "hard", or "edge_case"
- "category": string - main topic (e.g., "fair_delivery", "run_out", "lbw", "caught", "penalty")
- "primary_law": integer - the main law number
- "laws_involved": array of integers - all law numbers involved

Return ONLY the JSON array. No markdown, no explanation.

EXAM TEXT:
{pdf_text}"""

    response = call_claude(client, prompt, system=SYSTEM_UMPIRE)
    questions = parse_json_from_response(response)

    for i, q in enumerate(questions):
        q["id"] = f"bcci_{i + 1:03d}"
        q["source"] = "bcci_exam"
        q["source_url"] = None
        q["notes"] = "From BCCI Umpires' Examination 2025 (MCC-relevant only)"

    print(f"  Extracted {len(questions)} MCC-relevant questions")
    return questions


# ═══════════════════════════════════════════════════════════
# SOURCE 2: CRICKET UMPIRING BLOG
# ═══════════════════════════════════════════════════════════

def extract_blog_questions(client):
    """Extract Q&A pairs from the cricket umpiring blog PDF."""
    print("\n" + "=" * 55)
    print("  Source 2: Cricket Umpiring Blog")
    print("=" * 55)

    pdf_path = "cricketumpiring-blog-page.pdf"
    if not os.path.exists(pdf_path):
        print(f"  Skipping: {pdf_path} not found")
        return []

    pdf_text = extract_pdf_text(pdf_path)
    print(f"  Extracted {len(pdf_text)} chars from PDF")

    prompt = f"""Below is text from a cricket umpiring blog containing user-submitted questions and expert answers about cricket rules.

Extract 15-17 of the BEST question-answer pairs that relate to MCC Laws of Cricket. Pick diverse scenarios:

Good picks:
- Clear scenarios with definitive rulings based on MCC Laws
- Questions about different types of dismissals
- Questions about scoring, penalties, fielding rules
- Unusual or tricky edge cases

Skip:
- Questions about T20/ODI-specific format rules not in MCC Laws
- Questions about ICC/BCCI playing conditions
- Vague discussion questions (not about specific rulings)
- Questions where the blog answer is unclear or incomplete
- Duplicate or near-duplicate scenarios

For each, provide a JSON array of objects with these exact fields:
- "question": string - rewrite as a clean, self-contained scenario
- "expected_ruling": string - the correct ruling
- "expected_ruling_type": "definitive" or "guidance"
- "expected_explanation_keywords": array of strings - MCC Law references cited
- "difficulty": "easy", "hard", or "edge_case"
- "category": string - main topic
- "primary_law": integer - main law number
- "laws_involved": array of integers
- "blog_context": string - brief note about original blog Q&A

Return ONLY the JSON array. No markdown, no explanation.

BLOG TEXT:
{pdf_text}"""

    response = call_claude(client, prompt, system=SYSTEM_UMPIRE, max_tokens=16384)
    questions = parse_json_from_response(response)

    for i, q in enumerate(questions):
        q["id"] = f"blog_{i + 1:03d}"
        q["source"] = "blog"
        q["source_url"] = "http://www.blog.cricketumpiring.com/p/blog-page_28.html"
        q["notes"] = q.pop("blog_context", "From cricket umpiring blog Q&A")

    print(f"  Extracted {len(questions)} questions")
    return questions


# ═══════════════════════════════════════════════════════════
# SOURCE 3: LLM-GENERATED FROM MCC LAWS
# ═══════════════════════════════════════════════════════════

def generate_llm_questions(client):
    """Generate diverse evaluation questions using Claude + MCC Laws knowledge."""
    print("\n" + "=" * 55)
    print("  Source 3: LLM-Generated")
    print("=" * 55)

    # Build table of contents from our chunk data
    toc = load_laws_toc()
    toc_text = ""
    for law_num in sorted(toc.keys()):
        info = toc[law_num]
        toc_text += f"\nLaw {law_num}: {info['title']}\n"
        for s in info["sections"]:
            toc_text += f"  - {s}\n"

    prompt = f"""You are generating an evaluation dataset for a RAG system that answers cricket umpiring questions using the MCC Laws of Cricket.

Generate EXACTLY 18 cricket scenario questions. Each must be answerable from the MCC Laws.

Distribution:
- 6 EASY: Straightforward application of a single law (clear-cut scenarios)
- 6 HARD: Require understanding interactions between 2-3 laws
- 6 EDGE CASE: Unusual/tricky scenarios that test boundaries of the laws

Coverage requirements (at minimum):
- 2+ questions about modes of dismissal (caught, bowled, lbw, run out, stumped, hit wicket, etc.)
- 2+ questions about scoring (runs, boundaries, extras, penalty runs, short runs)
- 2+ questions about bowling/delivery rules (no ball, wide, fair delivery, mode of delivery)
- 2+ questions about fielding (illegal fielding, protective equipment, substitutes)
- 2+ questions about match procedures (declarations, intervals, results, toss)
- 2+ questions about unfair play (ball tampering, time wasting, dangerous bowling)
- Remaining questions: any law area not yet covered

For each question, provide a JSON array of objects with these exact fields:
- "question": string - a clear, detailed scenario
- "expected_ruling": string - the definitive ruling
- "expected_ruling_type": "definitive" or "guidance"
- "expected_explanation_keywords": array of law section references (e.g., ["Law 33.2.2", "Law 20.1"])
- "difficulty": "easy", "hard", or "edge_case"
- "category": string - main topic
- "primary_law": integer - main law number
- "laws_involved": array of integers
- "relevant_sections_finer": array of chunk IDs in format "lawN_sN.N.N" (e.g., ["law33_s33.2.2", "law20_s20.1"])
- "relevant_sections_broader": array of chunk IDs in format "lawN_sN.N" (e.g., ["law33_s33.2", "law20_s20.1"])

Here is the complete table of contents of our MCC Laws chunk database (use these section numbers for chunk IDs):
{toc_text}

Return ONLY the JSON array. No markdown wrapping, no explanation outside the JSON."""

    response = call_claude(client, prompt, system=SYSTEM_UMPIRE, max_tokens=16384)
    questions = parse_json_from_response(response)

    for i, q in enumerate(questions):
        q["id"] = f"llm_{i + 1:03d}"
        q["source"] = "llm_generated"
        q["source_url"] = None
        q["notes"] = f"LLM-generated ({q.get('difficulty', 'unknown')})"

    print(f"  Generated {len(questions)} questions")
    return questions


# ═══════════════════════════════════════════════════════════
# SOURCE 4: REDDIT / WEB
# ═══════════════════════════════════════════════════════════

def scrape_reddit_questions(client):
    """Fetch cricket umpiring questions from Reddit, or generate realistic alternatives."""
    print("\n" + "=" * 55)
    print("  Source 4: Reddit / Web")
    print("=" * 55)

    import urllib.request
    import urllib.parse

    reddit_posts = []
    queries = [
        "umpire decision rules",
        "unusual dismissal law cricket",
        "cricket rules confusion",
    ]

    for query in queries:
        try:
            url = (
                f"https://old.reddit.com/r/Cricket/search.json"
                f"?q={urllib.parse.quote(query)}&restrict_sr=1&sort=relevance&t=all&limit=10"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "FourthUmpireAI/1.0 (Educational Research)"}
            )
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            for post in data.get("data", {}).get("children", []):
                pd = post.get("data", {})
                selftext = pd.get("selftext", "")
                if selftext and len(selftext) > 50:
                    reddit_posts.append({
                        "title": pd.get("title", ""),
                        "text": selftext[:1500],
                        "url": f"https://reddit.com{pd.get('permalink', '')}",
                    })
            print(f"  Fetched {len(reddit_posts)} posts for query: '{query}'")
            time.sleep(3)  # Respectful rate limiting
        except Exception as e:
            print(f"  Reddit fetch failed for '{query}': {e}")

    if len(reddit_posts) >= 5:
        print(f"  Processing {len(reddit_posts)} Reddit posts...")
        posts_text = "\n\n---\n\n".join(
            f"Title: {p['title']}\nURL: {p['url']}\nText: {p['text']}"
            for p in reddit_posts[:25]
        )

        prompt = f"""Below are posts from r/Cricket discussing umpiring questions and rule debates.

Extract up to 10 clear cricket umpiring scenarios that can be answered using MCC Laws.

For each, provide a JSON array of objects with:
- "question": string - clean scenario description based on the Reddit discussion
- "expected_ruling": string - correct ruling per MCC Laws
- "expected_ruling_type": "definitive" or "guidance"
- "expected_explanation_keywords": array of Law references
- "difficulty": "easy", "hard", or "edge_case"
- "category": string
- "primary_law": integer
- "laws_involved": array of integers
- "source_url": string - the Reddit URL

Return ONLY the JSON array.

REDDIT POSTS:
{posts_text}"""

        response = call_claude(client, prompt, system=SYSTEM_UMPIRE)
        questions = parse_json_from_response(response)

    else:
        # Fallback: generate realistic real-world debate questions
        print("  Reddit scraping didn't find enough posts.")
        print("  Generating real-world cricket debate questions instead...")

        prompt = """Generate exactly 10 cricket umpiring questions inspired by real-world cricket debates and controversial decisions.

These should feel like questions a cricket fan would post on Reddit or a forum - NOT textbook questions.

Focus on:
- Controversial decisions from real cricket (club or international)
- Scenarios where fans commonly misunderstand the rules
- Edge cases that even experienced players debate
- Unusual situations from amateur/club cricket

Make each scenario detailed and realistic, as if someone is describing what happened in a match.

For each, provide a JSON array of objects with:
- "question": string - detailed scenario written in casual style
- "expected_ruling": string - correct ruling per MCC Laws
- "expected_ruling_type": "definitive" or "guidance"
- "expected_explanation_keywords": array of Law references
- "difficulty": "hard" or "edge_case"
- "category": string
- "primary_law": integer
- "laws_involved": array of integers

Return ONLY the JSON array."""

        response = call_claude(client, prompt, system=SYSTEM_UMPIRE, max_tokens=16384)
        questions = parse_json_from_response(response)

    for i, q in enumerate(questions):
        q["id"] = f"web_{i + 1:03d}"
        q["source"] = "scraped"
        q.setdefault("source_url", None)
        q.setdefault("notes", "")

    print(f"  Got {len(questions)} questions")
    return questions


# ═══════════════════════════════════════════════════════════
# CANDIDATE CHUNK RETRIEVAL
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


def add_candidate_chunks(questions, voyage_client, chroma_client):
    """Retrieve candidate chunk IDs from ChromaDB for all questions."""
    print("\n" + "=" * 55)
    print("  Retrieving Candidate Chunks from ChromaDB")
    print("=" * 55)

    # Batch embed all questions
    embeddings = batch_embed_questions(voyage_client, questions)

    # Query ChromaDB for each embedding (local, no rate limit)
    for key in ["finer", "broader"]:
        try:
            collection = chroma_client.get_collection(COLLECTIONS[key]["name"])
        except Exception as e:
            print(f"  Warning: {key} collection not found: {e}")
            for q in questions:
                q[f"candidate_chunks_{key}"] = []
            continue

        print(f"  Querying {key} collection...")
        for i, (q, emb) in enumerate(zip(questions, embeddings)):
            results = collection.query(
                query_embeddings=[emb],
                n_results=TOP_K,
                include=["metadatas"],
            )
            q[f"candidate_chunks_{key}"] = results["ids"][0]

    print(f"  Done: added candidates for {len(questions)} questions")
    return questions


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    skip_reddit = "--skip-reddit" in sys.argv

    print("=" * 55)
    print("  FOURTH UMPIRE AI — Golden Dataset Generator")
    print("=" * 55)

    # Setup clients
    anthropic_client = Anthropic()
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    os.makedirs(EVAL_DIR, exist_ok=True)

    # Load chunk ID sets for validation
    all_chunk_ids = load_all_chunk_ids()

    all_questions = []

    # ── Source 1: BCCI Exam ──
    try:
        bcci = extract_bcci_questions(anthropic_client)
        all_questions.extend(bcci)
        print(f"  Running total: {len(all_questions)} questions")
    except Exception as e:
        print(f"  ERROR in BCCI extraction: {e}")

    time.sleep(2)

    # ── Source 2: Blog ──
    try:
        blog = extract_blog_questions(anthropic_client)
        all_questions.extend(blog)
        print(f"  Running total: {len(all_questions)} questions")
    except Exception as e:
        print(f"  ERROR in blog extraction: {e}")

    time.sleep(2)

    # ── Source 3: LLM-Generated ──
    try:
        llm = generate_llm_questions(anthropic_client)
        all_questions.extend(llm)
        print(f"  Running total: {len(all_questions)} questions")
    except Exception as e:
        print(f"  ERROR in LLM generation: {e}")

    time.sleep(2)

    # ── Source 4: Reddit/Web ──
    if not skip_reddit:
        try:
            reddit = scrape_reddit_questions(anthropic_client)
            all_questions.extend(reddit)
            print(f"  Running total: {len(all_questions)} questions")
        except Exception as e:
            print(f"  ERROR in Reddit scraping: {e}")
    else:
        print("\n  Skipping Reddit (--skip-reddit flag)")

    # ── Normalize all questions ──
    print(f"\n  Total raw questions: {len(all_questions)}")
    for q in all_questions:
        normalize_question(q, all_chunk_ids)

    # ── Add candidate chunks from ChromaDB ──
    all_questions = add_candidate_chunks(all_questions, voyage_client, chroma_client)

    # ── Build and save dataset ──
    source_counts = {}
    difficulty_counts = {}
    category_counts = {}
    for q in all_questions:
        src = q.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        diff = q.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        cat = q.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    dataset = {
        "metadata": {
            "version": "draft_v1",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(all_questions),
            "sources": source_counts,
            "difficulty_distribution": difficulty_counts,
            "category_distribution": category_counts,
            "status": "DRAFT - needs manual review and verification",
            "instructions": (
                "Review each question. Verify: (1) expected_ruling is correct, "
                "(2) relevant_chunk_ids_finer/broader contain the RIGHT chunks "
                "(not just what retrieval found). Once verified, save as "
                "evals/golden_dataset.json"
            ),
        },
        "questions": all_questions,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  DATASET GENERATED")
    print("=" * 55)
    print(f"  Output:     {OUTPUT_FILE}")
    print(f"  Total:      {len(all_questions)} questions")
    print(f"  Sources:    {json.dumps(source_counts)}")
    print(f"  Difficulty: {json.dumps(difficulty_counts)}")
    print(f"  Status:     DRAFT — review and verify before using")
    print()
    print("  Next steps:")
    print("  1. Open evals/golden_dataset_draft.json")
    print("  2. Review each question's expected_ruling")
    print("  3. Verify relevant_chunk_ids match the actual needed chunks")
    print("  4. Save verified version as evals/golden_dataset.json")
    print("=" * 55)


if __name__ == "__main__":
    main()

# Fourth Umpire AI

An agentic chatbot that umpires cricket. Fourth Umpire AI blends retrieval-augmented generation over the MCC Laws of Cricket with tool-calling for match arithmetic (run rates, overs↔balls), conversation memory so follow-up questions keep their scenario context, a three-way intent classifier that routes greetings, cricket-chat, and off-topic inputs away from the ruling engine, and hybrid retrieval (cosine ∪ reranker) that keeps multi-law scenarios grounded. Responses stream in real time with the model's private reasoning filtered out, every ruling cites the specific Law number, and a thumbs-up/down feedback loop captures disagreement for later review.

**Try it live:** [fourth-umpire-ai.streamlit.app](https://fourth-umpire-ai.streamlit.app/)

## System Architecture

```
                        DATA PIPELINE (run once)
                        ========================

  ┌──────────────┐
  │  MCC Laws    │
  │  of Cricket  │     ┌─────────────────────────────────────┐
  │  (PDF)       │────▶│          ingest_pdf.py               │
  └──────────────┘     │                                      │
                       │  Extracts text from PDF pages        │
                       │  Splits into Laws, Appendices        │
                       │  Chunks by sub-law sections          │
                       │                                      │
                       │  --broader  → N.N level only         │
                       │  (default)  → N.N + N.N.N levels     │
                       │                                      │
                       │  Fixes duplicate IDs (e.g. 1947)     │
                       │  Preserves parent context in chunks  │
                       └──────────────┬──────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │         JSON Chunk Files              │
                       │                                       │
                       │  broader: 279 chunks (larger, fewer)  │
                       │  finer:   717 chunks (smaller, more)  │
                       │                                       │
                       │  Each chunk has:                      │
                       │    id, text, law_number, law_title,   │
                       │    section, section_title              │
                       └──────────────┬───────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │        ingest_to_db.py                │
                       │                                       │
                       │  For each chunk:                      │
                       │    1. Compute MD5 hash of text        │
                       │    2. Check if chunk exists in DB     │
                       │    3. Compare hash → skip if same     │
                       │    4. Embed via Voyage AI only if     │
                       │       new or changed                  │
                       │    5. Upsert into ChromaDB            │
                       │                                       │
                       │  Rate limiting: 20 chunks/batch,      │
                       │  25s between batches (3 RPM free tier)│
                       │  Auto-retry on rate limit errors      │
                       │                                       │
                       │  --fresh  → force re-embed all        │
                       └──────────────┬───────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │      ChromaDB  (chroma_storage/)      │
                       │                                       │
                       │  mcc_rules_broader  (279 vectors)     │
                       │  mcc_rules_finer    (717 vectors)     │
                       │                                       │
                       │  Each entry stores:                   │
                       │    embedding, document text,          │
                       │    metadata (law info + text_hash)    │
                       └──────────────────────────────────────┘


                        QUERY PIPELINE (every question)
                        ===============================

  ┌──────────────┐
  │  User asks   │     ┌──────────────────────────────────────┐
  │  a cricket   │────▶│      query_expansion.py               │
  │  question    │     │                                       │
  └──────────────┘     │  1. Rewrite question using formal     │
                       │     MCC Laws terminology (Haiku)      │
                       │                                       │
                       │  2. Three-way intent classifier:      │
                       │     [GREETING]          → friendly    │
                       │                           reply       │
                       │     [CRICKET_OFF_TOPIC] → cricket     │
                       │                           chat, no RAG│
                       │     [OFF_TOPIC]         → polite      │
                       │                           refusal     │
                       │     else                → continue    │
                       │                                       │
                       │  Follow-up turns: Haiku summarises    │
                       │  prior scenario context (no rulings)  │
                       └──────────────┬───────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │            query.py                   │
                       │                                       │
                       │  3. Embed expanded query via Voyage   │
                       │     AI (single embedding, reused)     │
                       │                                       │
                       │  4. Retrieve top 10 chunks from       │
                       │     broader collection (cosine sim)   │
                       │                                       │
                       │  5. Hybrid retrieval: union of        │
                       │     cosine top-5 + reranked top-5     │
                       │     (deduplicated, capped at 6 via    │
                       │     HYBRID_FINAL_K)                   │
                       │                                       │
                       │  6. Build context from unique chunks  │
                       │                                       │
                       │  7. Send to Claude Sonnet with        │
                       │     Fourth Umpire persona + tools     │
                       │     (calculator, overs↔balls)         │
                       │                                       │
                       │  8. Stream response token-by-token    │
                       │     (Streamlit) or return full (CLI)  │
                       └──────────────┬───────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │      Fourth Umpire's Ruling           │
                       │                                       │
                       │  - Thinking process (analysis)        │
                       │  - Definitive ruling with law cites   │
                       │  - Detailed explanation               │
                       │  - Expandable citations (Streamlit)   │
                       │                                       │
                       │  If laws don't cover the scenario:    │
                       │  → States this clearly                │
                       │  → Provides guidance from closest     │
                       │    related laws                       │
                       └──────────────┬───────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │      User Feedback (optional)         │
                       │                                       │
                       │  Thumbs up/down on each ruling        │
                       │  → Google Sheets (production)         │
                       │  → Local JSONL (development)          │
                       └──────────────────────────────────────┘


                        SHARED INFRASTRUCTURE
                        =====================

                       ┌──────────────────────────────────────┐
                       │  config.py          (single source    │
                       │                      of truth)        │
                       │   Models, paths, collections,         │
                       │   rate limits, RAG settings            │
                       ├──────────────────────────────────────┤
                       │  prompts/system.md  (AI persona)      │
                       │   Editable without touching code      │
                       │   Structured: Rules, Tools, Format    │
                       ├──────────────────────────────────────┤
                       │  tools.py           (calculations)    │
                       │   calculator, overs_to_balls,         │
                       │   balls_to_overs                      │
                       ├──────────────────────────────────────┤
                       │  init_db.py         (DB setup)        │
                       │   Safe by default — preserves data    │
                       │   --fresh to reset                    │
                       └──────────────────────────────────────┘
```

## Project Structure

```
fourth_umpire_ai/
├── app.py                                     # Streamlit web app (chat UI, streaming, feedback)
├── query.py                                   # RAG pipeline (retrieval, hybrid rerank, streaming, tool loop)
├── query_expansion.py                         # Query rewriting + off-topic rejection (Haiku)
├── tools.py                                   # Cricket calculation tools (calculator, overs/balls)
├── config.py                                  # Shared settings (models, paths, collections)
├── prompts/
│   └── system.md                              # Fourth Umpire AI persona prompt
├── ingest_pdf.py                              # PDF → JSON chunks
├── init_db.py                                 # Create ChromaDB collections
├── ingest_to_db.py                            # Embed chunks → ChromaDB (smart re-embedding)
├── run_eval.py                                # Evaluation runner with LLM judge
├── graph_rag.py                               # Optional one-hop cross-ref expansion (eval-only, not on Streamlit)
├── generate_eval_dataset.py                   # Golden dataset generator
├── evals/
│   ├── golden_dataset.json                    # 51-question eval set with expected answers
│   └── results/                               # Eval run results (JSON)
├── real_cricket_rules_broader_chunking.json   # 279 chunks (N.N level)
├── real_cricket_rules_finer_chunking.json     # 717 chunks (N.N + N.N.N level)
├── chroma_storage/                            # ChromaDB vector database (committed for deployment)
├── feedback/                                  # Local feedback logs (dev only, gitignored)
├── requirements.txt                           # Python dependencies
└── .env                                       # API keys (not in repo)
```

## Setup

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Voyage AI API key](https://dash.voyageai.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/rsd19930/fourth-umpire-ai.git
cd fourth-umpire-ai

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your-key-here
VOYAGE_API_KEY=your-key-here
EOF
```

### Load the Data

```bash
# 1. Generate chunks from the PDF (downloads PDF automatically)
python3 ingest_pdf.py              # Finer chunks (717)
python3 ingest_pdf.py --broader    # Broader chunks (279)

# 2. Initialize the database
python3 init_db.py

# 3. Embed and load into ChromaDB (~15 min on free tier)
python3 ingest_to_db.py
```

## Usage

**Live app:** [fourth-umpire-ai.streamlit.app](https://fourth-umpire-ai.streamlit.app/)

```bash
# Web app (primary — chat UI with streaming, citations, feedback)
streamlit run app.py

# CLI (development/testing)
python3 query.py                   # Both collections
python3 query.py --broader         # Broader collection only (production default)
python3 query.py --finer           # Finer collection only
```

### Example

```
Your question: The batter hits the ball and it strikes a fielder's
helmet lying on the ground behind the wicket-keeper. How many penalty
runs are awarded?

Ruling: 5 Penalty Runs awarded to the Batting Side

Explanation: Under Law 28.3.2, when the ball in play strikes a
protective helmet placed behind the wicket-keeper (Law 28.3.1),
the ball immediately becomes dead and 5 Penalty runs are awarded
to the batting side...
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Query Expansion | Claude Haiku 4.5 (Anthropic) |
| Embeddings | Voyage AI (voyage-4-lite) |
| Reranking | Voyage AI (rerank-2.5-lite) |
| Vector Database | ChromaDB (persistent) |
| Web App | Streamlit |
| Feedback Storage | Google Sheets (production) / JSONL (development) |
| PDF Extraction | PyPDF2 |
| Language | Python 3.13 |

## Key Design Decisions

- **Broader chunks won**: Evaluated two chunking strategies — broader (279 chunks at N.N level) vs finer (717 chunks at N.N + N.N.N level). Broader outperformed on all metrics and is used in production
- **Hybrid retrieval**: Fetch 10 chunks via cosine similarity, then take the union of cosine top-5 + reranked top-5 (deduplicated by chunk ID). Neither method can veto a chunk the other ranked highly — capped at 6 final chunks via `HYBRID_FINAL_K` to keep context tight
- **Query expansion**: Haiku rewrites casual user language to formal MCC Laws terminology before embedding (e.g., "batter" to "striker"). The original question is still used for ruling generation — expanded version is only for retrieval
- **Three-way intent classifier**: query expansion doubles as a gatekeeper — `[GREETING]` gets a short friendly reply, `[CRICKET_OFF_TOPIC]` (cricket chat that isn't a rules question) gets a light conversational reply, `[OFF_TOPIC]` gets a polite refusal. Only "real" rules questions hit the retrieval + ruling pipeline, saving API calls on the other three paths
- **Conversation memory**: on follow-up turns, Haiku summarises the prior scenario context — scenarios only, never past rulings or law citations — so context is preserved without the model retroactively revising its earlier ruling
- **Tool calling**: Claude has access to calculation tools (calculator, overs_to_balls, balls_to_overs) for match arithmetic — only invoked when the question involves numbers
- **Streaming with thinking-tag filter**: token-by-token response via Claude's streaming API, integrated with Streamlit's `st.write_stream()`. A 30-char lookahead filter strips `<Thinking>…</Thinking>` blocks in real time so the user sees only the final ruling
- **Feedback loop**: Thumbs up/down on each ruling flows to Google Sheets in production, local JSONL in development
- **Smart re-embedding**: MD5 hash per chunk — only re-embeds when text actually changes, saving API costs
- **Separated AI persona**: `prompts/system.md` can be edited without touching code
- **Centralized config**: all settings in `config.py` — no duplicated values across scripts

## Feedback

User feedback is collected via thumbs up/down on each ruling in the Streamlit app. Each feedback entry captures: timestamp, question asked, ruling given, retrieved chunk IDs, and positive/negative rating.

**Feedback data:** [Google Sheets](https://docs.google.com/spreadsheets/d/1m_TypklqIS8TFzRzUjGR3E4Cp6DZ4sXatdx9Y5j6SA0/edit?usp=sharing)

## Learnings

### 1. Golden eval datasets are tedious but essential

Our 51-question eval set required manually verifying every answer, difficulty label, and expected chunk ID against the source laws. There's no shortcut — auto-generated evals give meaningless metrics. But done well, the dataset serves as both a quality gate (catching hallucinations and missing context before users see them) and a compass for improvement (showing exactly where retrieval or reasoning breaks down).

### 2. Graph RAG v1 sounded right but didn't pan out

A single cricket scenario can span 3-4 interconnected laws (e.g., an overthrow after a no-ball touches Law 21, 19, 18, and 24). Vector similarity retrieves chunks independently with no understanding of cross-references. The hypothesis: parse "See Law 21.8" / "Appendix A" style references out of retrieved chunks and append the referenced chunks in one hop. We built this as an **optional, eval-only** pipeline (`graph_rag.py`, enabled via `run_eval.py --graph-rag`) so Streamlit stays on the standard path.

The 2026-04-18 eval falsified v1: Context Precision collapsed from 0.57 → 0.40, Context Recall barely moved (0.61 → 0.62), and Answer Relevance dipped (0.68 → 0.65). Appending up to 5 cross-referenced chunks added noise faster than it added signal — the reranker had already filtered for a reason. A production Graph RAG would need a smarter edge model (filter expansions by relevance to the current question, not just by citation presence) rather than naive one-hop append.

### 3. Rerankers improve faithfulness but can hurt accuracy

Adding a reranker (fetch 10 → keep 5) improved faithfulness from 0.70 to 0.81 but left answer relevance flat — 7 questions improved, 6 regressed, nearly cancelling out. The reranker surfaces more semantically similar chunks (so Claude stays grounded), but it demoted chunks critical for multi-law reasoning that "looked" less relevant. Example: a caught-behind question lost the umpire consultation law (Law 31.7) because the reranker preferred sections mentioning "caught" more directly. Fix: hybrid retrieval (union of cosine top-5 + reranked top-5) so neither method can veto the other.


## Journey

Each step below corresponds to a row in the Eval Runs table. Numbers are `AR / F / CR / CP` (Answer Relevance / Faithfulness / Context Recall / Context Precision) on broader chunks unless noted.

### Step 0 — Baseline (2026-04-11)
**Config:** top-5 cosine similarity, both finer (717) and broader (279) collections, Claude Sonnet 4.6 for both answer generation and as judge.
**Result:** Finer `0.56 / 0.78 / 0.81 / 0.97`, Broader `0.62 / 0.84 / 0.85 / 0.97`.
**Why & verdict:** Needed a starting point and a head-to-head between chunking strategies. Broader beat finer on every metric — the larger sub-law chunks carried enough context on their own, while the finer split fragmented rulings across chunks. **Kept:** broader-only from here on.

### Step 1 — Cheaper judge + tighter gen (2026-04-11)
**Change:** Dropped finer collection, switched judge to Claude Haiku 4.5, set `max_tokens=2048` for generation, cleaned up the system prompt.
**Result:** `0.62 / 0.72 / 0.85 / 0.97` (15m 0s, ~260K tokens).
**Why & verdict:** The sonnet-as-judge cost was hard to justify on a learning project, and the generator was rambling past where the ruling actually ended. Haiku judge saved ~3× on eval cost with scores that held up. Faithfulness dipped (0.84→0.72) because shorter generations sometimes dropped citations. **Kept:** haiku judge, tighter token budget. The faithfulness regression became a target for later steps.

### Step 2 — Query expansion + tool calling (2026-04-12)
**Change:** Added Haiku query expansion (rewrites user language to formal MCC terms before embedding) and Claude tool-calling for calculator / overs↔balls.
**Result:** `0.70 / 0.70 / 0.66 / 0.75`.
**Why & verdict:** Users ask "batter", the laws say "striker" — embeddings were missing obvious hits. And numeric questions ("what's the required run rate if...") needed real arithmetic, not vibes. AR jumped +0.08, which is what we hoped for. But recall and precision both dropped — the expanded query was pulling in plausible-but-wrong chunks. **Kept:** both. The retrieval regression pushed us toward reranking.

### Step 3 — Reranker (2026-04-12)
**Change:** Fetch top-10 by cosine, rerank with Voyage `rerank-2.5-lite`, keep top-5.
**Result:** `0.70 / 0.81 / 0.55 / 0.63` (46m 13s — reranking is the slow step).
**Why & verdict:** Step 2's precision drop suggested we needed a second-stage filter. Faithfulness recovered (0.70→0.81) because the reranker picked chunks that actually matched the question semantically. But recall fell further (0.66→0.55) — the reranker was demoting chunks critical for multi-law reasoning that "looked" less relevant (e.g., dropping Law 31.7 on a caught-behind question because it mentioned "umpire consultation" instead of "caught"). **Partially kept:** the reranker stayed, but we clearly needed something to stop it from vetoing important chunks.

### Step 4 — Hybrid retrieval + anti-hedging prompt (2026-04-14)
**Change:** Take the union of cosine top-5 and reranker top-5 (deduped by chunk ID), so neither method can veto the other. Added anti-hedging rules to the system prompt.
**Result:** `0.66 / 0.84 / 0.72 / 0.63`.
**Why & verdict:** Directly addressed Step 3's recall hole. CR bounced back 0.55→0.72 while faithfulness stayed high. AR regressed slightly (0.70→0.66), which we suspected was prompt-related, not retrieval-related. **Kept:** hybrid retrieval — the single most important architectural decision in the project.

### Step 5 — Hybrid cap + prompt hardening (2026-04-17)
**Change:** Added `HYBRID_FINAL_K=6` to cap the hybrid union (unions could balloon to 10 chunks, diluting context). Added anti-hedging Rule 3 and a thorough-reading Rule 7 to the system prompt. Commit `ab5c5ec`.
**Result:** `0.68 / 0.78 / 0.61 / 0.57`.
**Why & verdict:** Pre-cap, too many marginal chunks were making it into context and faithfulness was taking hits when Claude leaned on borderline ones. AR ticked back up (0.66→0.68) from the prompt tweaks; CR and CP drifted down as expected given the tighter context. Net trade-off we were willing to take for more decisive rulings. **Kept:** this is the current production architecture.

### Step 6 — Graph RAG v1 (2026-04-18, eval-only)
**Change:** One-hop cross-reference expansion. Parse "See Law 21.8" / "Appendix A" references out of retrieved chunks and append up to 5 referenced chunks. Isolated in `graph_rag.py`, gated behind `run_eval.py --graph-rag` — Streamlit never touches it.
**Result:** `0.65 / 0.77 / 0.62 / 0.40`.
**Why & verdict:** Cricket scenarios span multiple laws (overthrow → Laws 21, 19, 18, 24). Vector search can't follow those edges. The hypothesis was that naive one-hop expansion would lift Context Recall materially. It didn't. CR barely moved (+0.01), Context Precision collapsed (−0.17), and AR dipped too — the appended chunks added noise faster than signal. **Rejected.** The code stays in the repo as an honest negative result; production Graph RAG would need a smarter edge-selection model, not just citation-presence.

## Eval Runs

### Eval Run — 2026-04-11

**Finer Chunks (717):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.56 | 0.78 | 0.81 | 0.97 |
| Easy (13) | 0.65 | 0.81 | 0.85 | 1.00 |
| Hard (26) | 0.52 | 0.81 | 0.78 | 0.96 |
| Edge Case (12) | 0.54 | 0.67 | 0.84 | 0.95 |

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.62 | 0.84 | 0.85 | 0.97 |
| Easy (13) | 0.77 | 0.96 | 0.88 | 1.00 |
| Hard (26) | 0.52 | 0.77 | 0.84 | 0.97 |
| Edge Case (12) | 0.67 | 0.88 | 0.82 | 0.93 |

**Config:** TOP_K=5, model=claude-sonnet-4-6, judge=claude-sonnet-4-6
**What changed:** Baseline run with default settings
**Results file:** `evals/results/eval_20260411_134614.json`

### Eval Run — 2026-04-11

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.62 | 0.72 | 0.85 | 0.97 |
| Easy (13) | 0.69 | 0.85 | 0.88 | 1.00 |
| Hard (26) | 0.54 | 0.65 | 0.84 | 0.97 |
| Edge Case (12) | 0.71 | 0.71 | 0.82 | 0.93 |

**Config:** TOP_K=5, model=claude-sonnet-4-6, judge=claude-haiku-4-5-20251001
**Runtime:** 15m 0s | **Tokens:** 223.0K input + 37.2K output
**What changed:** Optimised: broader-chunks-only, haiku as judge, max_tokens=2048 for generating answer, improved system prompt
**Results file:** `evals/results/eval_20260411_165711.json`

### Eval Run — 2026-04-12

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.70 | 0.70 | 0.66 | 0.75 |
| Easy (13) | 0.69 | 0.85 | 0.70 | 0.80 |
| Hard (26) | 0.71 | 0.69 | 0.64 | 0.74 |
| Edge Case (12) | 0.67 | 0.54 | 0.64 | 0.72 |

**Config:** TOP_K=5, model=claude-sonnet-4-6, judge=claude-haiku-4-5-20251001
**Runtime:** 24m 26s | **Tokens:** 435.3K input + 74.6K output
**What changed:** Query expansion with Haiku as pre-processor + tool calling
**Results file:** `evals/results/eval_20260412_110958.json`

### Eval Run — 2026-04-12

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.70 | 0.81 | 0.55 | 0.63 |
| Easy (13) | 0.69 | 0.85 | 0.58 | 0.66 |
| Hard (26) | 0.71 | 0.77 | 0.56 | 0.64 |
| Edge Case (12) | 0.67 | 0.88 | 0.51 | 0.57 |

**Config:** Retrieve 10 → Rerank to 5 (rerank-2.5-lite), model=claude-sonnet-4-6, judge=claude-haiku-4-5-20251001
**Runtime:** 46m 13s | **Tokens:** 426.3K input + 64.4K output
**What changed:** Added reranker (rerank-2.5-lite, fetch 10 → keep 5), fixed empty-content retry, improved tool call prompt
**Results file:** `evals/results/eval_20260412_215120.json`

### Eval Run — 2026-04-14

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.66 | 0.84 | 0.72 | 0.63 |
| Easy (13) | 0.65 | 0.89 | 0.73 | 0.66 |
| Hard (26) | 0.67 | 0.85 | 0.72 | 0.62 |
| Edge Case (12) | 0.62 | 0.79 | 0.72 | 0.60 |

**Config:** Retrieve 10 → Rerank to 5 (rerank-2.5-lite), model=claude-sonnet-4-6, judge=claude-haiku-4-5-20251001
**Runtime:** 49m 2s | **Tokens:** 524.6K input + 70.7K output
**What changed:** Hybrid retrieval (union cosine+reranked top-5) + anti-hedging prompt rules
**Results file:** `evals/results/eval_20260414_193249.json`

### Eval Run — 2026-04-17

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.68 | 0.78 | 0.61 | 0.57 |
| Easy (13) | 0.69 | 0.89 | 0.61 | 0.55 |
| Hard (26) | 0.67 | 0.71 | 0.60 | 0.58 |
| Edge Case (12) | 0.67 | 0.83 | 0.64 | 0.60 |

**Config:** Retrieve 10 → Rerank to 5 (rerank-2.5-lite), model=claude-sonnet-4-6, judge=claude-haiku-4-5-20251001
**Runtime:** 49m 30s | **Tokens:** 486.6K input + 75.4K output
**What changed:** Hybrid cap HYBRID_FINAL_K=6 + anti-hedging Rule 3 + thorough-reading Rule 7 (commit ab5c5ec)
**Results file:** `evals/results/eval_20260417_185128.json`

### Eval Run — 2026-04-18

**Broader Chunks (279):**

| | Answer Relevance | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|---|
| **Overall** | 0.65 | 0.77 | 0.62 | 0.40 |
| Easy (13) | 0.73 | 0.96 | 0.67 | 0.42 |
| Hard (26) | 0.60 | 0.65 | 0.57 | 0.39 |
| Edge Case (12) | 0.67 | 0.79 | 0.69 | 0.41 |

**Config:** Retrieve 10 → Rerank to 5 (rerank-2.5-lite), model=claude-sonnet-4-6, judge=claude-haiku-4-5-20251001
**Runtime:** 52m 5s | **Tokens:** 735.5K input + 80.4K output
**What changed:** Graph RAG v1: one-hop cross-ref expansion (cap 5 new chunks)
**Results file:** `evals/results/eval_20260418_121410_graph_rag.json`

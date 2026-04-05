# Fourth Umpire AI

AI-powered cricket umpiring assistant that uses RAG (Retrieval-Augmented Generation) over the latest MCC Laws of Cricket to help answer complex umpiring decisions with accuracy and rule references.

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
  │  a cricket   │────▶│            query.py                   │
  │  question    │     │                                       │
  └──────────────┘     │  1. Embed question ONCE via Voyage AI │
                       │     (reuses same vector for both      │
                       │      collections — saves API calls)   │
                       │                                       │
                       │  2. Retrieve top 5 chunks from each   │
                       │     collection via cosine similarity  │
                       │                                       │
                       │  3. Deduplicate chunks across         │
                       │     collections (by chunk ID)         │
                       │                                       │
                       │  4. Build context from retrieved laws │
                       │                                       │
                       │  5. Send to Claude with Fourth Umpire │
                       │     persona (prompts/system.md)       │
                       └──────────────┬───────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────────┐
                       │      Fourth Umpire's Ruling           │
                       │                                       │
                       │  - Thinking process (analysis)        │
                       │  - Definitive ruling with law cites   │
                       │  - Detailed explanation               │
                       │                                       │
                       │  If laws don't cover the scenario:    │
                       │  → States this clearly                │
                       │  → Provides guidance from closest     │
                       │    related laws                       │
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
                       │   Structured: Rules, Format, Context  │
                       ├──────────────────────────────────────┤
                       │  init_db.py         (DB setup)        │
                       │   Safe by default — preserves data    │
                       │   --fresh to reset                    │
                       └──────────────────────────────────────┘
```

## Project Structure

```
fourth_umpire_ai/
├── config.py                                  # Shared settings (models, paths, collections)
├── prompts/
│   └── system.md                              # Fourth Umpire AI persona prompt
├── ingest_pdf.py                              # PDF → JSON chunks
├── init_db.py                                 # Create ChromaDB collections
├── ingest_to_db.py                            # Embed chunks → ChromaDB (smart)
├── query.py                                   # Interactive RAG query pipeline
├── real_cricket_rules_broader_chunking.json   # 279 chunks (N.N level)
├── real_cricket_rules_finer_chunking.json     # 717 chunks (N.N + N.N.N level)
├── requirements.txt                           # Python dependencies
├── .env                                       # API keys (not in repo)
└── chroma_storage/                            # Local vector DB (not in repo)
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

```bash
# Start the interactive umpire
python3 query.py                   # Compare both collections
python3 query.py --finer           # Finer collection only
python3 query.py --broader         # Broader collection only
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
| LLM | Claude Sonnet (Anthropic) |
| Embeddings | Voyage AI (voyage-4-lite) |
| Vector Database | ChromaDB (local, persistent) |
| PDF Extraction | PyPDF2 |
| Language | Python 3.13 |

## Key Design Decisions

- **Two chunking strategies** for evaluation: broader (279 chunks at N.N level) vs finer (717 chunks at N.N + N.N.N level)
- **Smart re-embedding**: MD5 hash per chunk — only re-embeds when text actually changes, saving API costs
- **Single question embedding**: embeds the user's question once and reuses the vector across both collections
- **Separated AI persona**: `prompts/system.md` can be edited without touching code
- **Centralized config**: all settings in `config.py` — no duplicated values across scripts

## Learnings

### 1. Golden evaluation datasets are hard to build — but worth every hour

Building a golden dataset (our 51-question evaluation set) is one of the most tedious parts of any RAG project. Every question needs a verified correct answer, the right difficulty label, and — most critically — the exact chunk IDs that should be retrieved. There's no shortcut: you have to read the source material, understand the laws, and manually verify that each tagged chunk is truly relevant. It's tempting to skip this step or auto-generate everything, but a sloppy eval set means your metrics are meaningless. Done well, a golden dataset serves two purposes simultaneously: it pushes for a better user experience (by measuring whether the system actually retrieves the right context and gives faithful answers), and it limits company risk (by catching hallucinations, missing context, and wrong rulings before users ever see them). The eval dataset is the quality gate — it's only as good as the manual effort behind it.

### 2. Graph RAG would be a better fit — but not for a learning project

Cricket laws are deeply interconnected. A single scenario (like an overthrow that reaches the boundary after a no-ball) can involve 3-4 laws that reference each other: Law 21 (No Ball), Law 19 (Boundaries), Law 18 (Scoring Runs), and Law 24 (Fielder Absent or Leaving the Field). A traditional vector-similarity RAG retrieves the top-K most similar chunks independently — it has no understanding of how laws link together. Graph RAG, where laws and sub-sections are nodes connected by edges representing cross-references, would naturally capture these relationships and retrieve entire clusters of related rules in one hop. However, building a knowledge graph requires significant data cleaning: extracting every cross-reference, resolving ambiguous citations, handling edge cases in the PDF parsing, and maintaining the graph as laws update. For this learning project, the effort-to-insight ratio didn't justify it — but for a production cricket umpiring tool, Graph RAG would be the clear next step.

"""
Shared configuration for the Fourth Umpire AI project.
All scripts import from here — single source of truth.
"""

import os

# Paths
CHROMA_PATH = "chroma_storage"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Models
EMBEDDING_MODEL = "voyage-4-lite"
LLM_MODEL = "claude-sonnet-4-6"
EXPANSION_MODEL = "claude-haiku-4-5-20251001"

# RAG settings
RETRIEVAL_K = 10    # Fetch from ChromaDB (wider net)
RERANK_K = 5        # Keep after reranking (focused context)
HYBRID_FINAL_K = 6  # Max chunks after hybrid retrieval
RERANK_MODEL = "rerank-2.5-lite"

# Voyage AI rate limits (free tier)
BATCH_SIZE = 20
WAIT_SECONDS = 25

# Collections
COLLECTIONS = {
    "broader": {
        "name": "mcc_rules_broader",
        "json_file": "real_cricket_rules_broader_chunking.json",
    },
    "finer": {
        "name": "mcc_rules_finer",
        "json_file": "real_cricket_rules_finer_chunking.json",
    },
}

DEFAULT_QUESTION = (
    "The batter hits the ball and it strikes a fielder's helmet "
    "that is lying on the ground behind the wicket-keeper. "
    "How many penalty runs are awarded and to which team?"
)

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

# RAG settings
TOP_K = 5

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

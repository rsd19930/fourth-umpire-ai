"""
Fourth Umpire AI — Streamlit Web App

Chat-based cricket umpiring assistant powered by RAG.
Retrieves relevant MCC Laws of Cricket and generates authoritative rulings.

Usage:
    streamlit run app.py
"""

import os
import json
from datetime import datetime
import streamlit as st
import voyageai
import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

from config import CHROMA_PATH, COLLECTIONS
from query import (
    load_system_prompt, embed_question, retrieve,
    hybrid_retrieve, build_context, generate_ruling,
    generate_ruling_stream, format_chunk_label,
)
from query_expansion import expand_query

# Load .env for local development (Streamlit Cloud uses st.secrets)
load_dotenv(override=True)


# ─── Helper: get secret from st.secrets or env ───────────────────────────────

def get_secret(key):
    """Get a secret from Streamlit secrets (cloud) or environment variables (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        value = os.getenv(key)
        if not value:
            st.error(f"Missing secret: {key}. Set it in .streamlit/secrets.toml or .env")
            st.stop()
        return value


# ─── Initialize clients (cached — runs once) ─────────────────────────────────

@st.cache_resource
def init_clients():
    """Initialize API clients and load resources. Cached across reruns."""
    anthropic_client = Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    voyage_client = voyageai.Client(api_key=get_secret("VOYAGE_API_KEY"), max_retries=3)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTIONS["broader"]["name"])
    system_prompt = load_system_prompt()
    return anthropic_client, voyage_client, collection, system_prompt


# ─── Citations helper ─────────────────────────────────────────────────────────

def render_citations(chunks):
    """Display retrieved law chunks as expandable citation blocks."""
    if not chunks:
        return
    st.markdown("---")
    st.markdown("**Laws Referenced:**")
    for chunk in chunks:
        label = format_chunk_label(chunk)
        with st.expander(label):
            st.markdown(chunk["text"])


# ─── Feedback helpers ─────────────────────────────────────────────────────────

FEEDBACK_LOG = "feedback/feedback_log.jsonl"


@st.cache_resource
def get_gsheet():
    """Connect to Google Sheets for feedback logging."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    return client.open("Fourth Umpire AI - RAG Feedback").sheet1


def log_feedback(msg_id, question, ruling, chunks, score):
    """Log feedback to Google Sheets (production) or local file (development)."""
    feedback_value = "positive" if score == 1 else "negative"
    chunk_ids = ", ".join(c["id"] for c in chunks) if chunks else ""

    # Try Google Sheets first (production)
    try:
        sheet = get_gsheet()
        sheet.append_row([
            datetime.now().isoformat(),
            question,
            ruling[:500],
            chunk_ids,
            feedback_value,
        ])
        return
    except Exception:
        pass  # Fall through to local file

    # Local fallback
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "msg_id": msg_id,
        "question": question,
        "ruling": ruling,
        "retrieved_chunk_ids": [c["id"] for c in chunks] if chunks else [],
        "feedback": feedback_value,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def render_feedback(msg):
    """Display thumbs up/down feedback widget for an assistant message."""
    msg_id = msg.get("msg_id")
    if msg_id is None:
        return  # No feedback for off-topic rejections

    feedback_key = f"feedback_{msg_id}"
    score = st.feedback("thumbs", key=feedback_key)

    # If user just gave feedback (score is not None) and we haven't logged it yet
    logged_key = f"feedback_logged_{msg_id}"
    if score is not None and not st.session_state.get(logged_key):
        log_feedback(
            msg_id=msg_id,
            question=msg.get("question", ""),
            ruling=msg.get("content", ""),
            chunks=msg.get("chunks", []),
            score=score,
        )
        st.session_state[logged_key] = True
        st.toast("Thanks for your feedback!", icon="✅")


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fourth Umpire AI",
    page_icon="🏏",
    layout="centered",
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Fourth Umpire AI 🏏")
    st.markdown(
        "Your AI Cricket Umpire — powered by the **MCC Laws of Cricket**.\n\n"
        "Ask any question about cricket rules, laws, or match situations."
    )
    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Your question is analyzed and expanded\n"
        "2. Relevant laws are retrieved from the MCC rulebook\n"
        "3. AI generates an authoritative ruling with citations"
    )
    st.divider()
    st.caption(
        "**Disclaimer:** Fourth Umpire AI can make mistakes. "
        "Rulings are based on the MCC Laws of Cricket and should not be "
        "treated as legal advice. Always verify with official sources "
        "before applying in real-world scenarios."
    )


# ─── Main chat area ──────────────────────────────────────────────────────────

st.header("Fourth Umpire AI 🏏")
st.caption("Ask any question about cricket rules and laws")

# Initialize clients
anthropic_client, voyage_client, collection, system_prompt = init_clients()

# Initialize chat history and feedback counter
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            render_citations(msg["chunks"])
        if msg["role"] == "assistant" and msg.get("msg_id") is not None:
            render_feedback(msg)

# Chat input
if question := st.chat_input("Ask a cricket rules question..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Process the question through the RAG pipeline
    with st.chat_message("assistant"):
        with st.status("Analyzing your question...", expanded=True) as status:

            # Step 1: Query expansion + off-topic check
            status.update(label="Understanding your question...")
            try:
                expanded, _ = expand_query(anthropic_client, question)
            except Exception as e:
                expanded = question  # Fallback to original on error

            # Off-topic rejection
            if expanded == "[OFF_TOPIC]":
                status.update(label="Off-topic detected", state="error", expanded=False)
                response_text = (
                    "I'm the Fourth Umpire AI — I can only help with cricket rules, "
                    "laws, and match situations. Please ask a cricket-related question!"
                )
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.stop()

            # Step 2: Embed + Retrieve + Hybrid rerank
            status.update(label="Retrieving relevant cricket laws...")
            try:
                query_embedding = embed_question(voyage_client, expanded)
                chunks = retrieve(collection, query_embedding)
                chunks = hybrid_retrieve(voyage_client, question, chunks)
            except Exception as e:
                status.update(label="Error retrieving laws", state="error", expanded=False)
                response_text = f"Sorry, I encountered an error retrieving the relevant laws. Please try again. ({e})"
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.stop()

            # Step 3: Build context
            context = build_context(chunks)

            status.update(label="Done!", state="complete", expanded=False)

        # Stream the ruling token-by-token
        try:
            ruling = st.write_stream(
                generate_ruling_stream(
                    anthropic_client, system_prompt, question, context
                )
            )
        except Exception as e:
            ruling = f"Sorry, I encountered an error generating the ruling. Please try again. ({e})"
            st.markdown(ruling)

        # Display citations
        render_citations(chunks)

        # Save to chat history (with chunks, msg_id, and question for feedback)
        msg_id = st.session_state.msg_counter
        st.session_state.msg_counter += 1
        msg_data = {
            "role": "assistant",
            "content": ruling,
            "chunks": chunks,
            "msg_id": msg_id,
            "question": question,
        }
        st.session_state.messages.append(msg_data)

        # Display feedback widget
        render_feedback(msg_data)

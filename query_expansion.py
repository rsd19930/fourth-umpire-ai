"""
Fourth Umpire AI — Query Expansion

Bridges the semantic gap between casual user language and formal MCC Laws
vocabulary. Rewrites the user's question using official cricket terminology
before embedding, so vector similarity retrieval finds more relevant chunks.

The expanded query is used ONLY for embedding/retrieval.
The original query still goes to Claude for ruling generation.
"""

import time
from anthropic import Anthropic
from config import EXPANSION_MODEL


EXPANSION_PROMPT = """You are a Cricket Terminology Translator. Rewrite the following cricket question using formal MCC Laws of Cricket terminology.

Rules:
- FIRST CHECK: If the question is NOT about cricket rules, laws, match situations, or cricket calculations, respond with exactly: [OFF_TOPIC]
- This includes: general knowledge questions, requests to write code, toxic/offensive content, and anything unrelated to cricket
- Replace casual/colloquial cricket language with official MCC Laws vocabulary (e.g., "batter" → "striker", "swats the ball" → "struck the ball a second time", "bowler chucks" → "bowler with a suspected illegal bowling action")
- Do NOT add legal conclusions, assumptions about which laws apply, or extra context
- Do NOT change the meaning or add information that is not in the original question
- If the question already uses formal terminology, return it unchanged
- If the question is about math or calculations (run rates, strike rates, etc.), return it unchanged
- Output ONLY the rewritten question, nothing else"""


def expand_query(anthropic_client, question, verbose=False):
    """
    Rewrite a casual cricket question using formal MCC terminology.

    Used for embedding only — the original question still goes to Claude
    for ruling generation.

    Returns: (expanded_text: str, usage: dict)
    """
    try:
        response = anthropic_client.messages.create(
            model=EXPANSION_MODEL,
            max_tokens=512,
            system=EXPANSION_PROMPT,
            messages=[{"role": "user", "content": question}],
        )

        # Validate response has text content
        if response.content and hasattr(response.content[0], "text"):
            expanded_text = response.content[0].text.strip()
            # Check for off-topic rejection
            if expanded_text == "[OFF_TOPIC]":
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                return "[OFF_TOPIC]", usage
        else:
            # Fallback to original if response is empty or unexpected
            expanded_text = question

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as e:
        # On any API error, fall back to original question
        print(f"  [QUERY EXPANSION WARNING: Failed for question, using original. Error: {e}]")
        expanded_text = question
        usage = {"input_tokens": 0, "output_tokens": 0}

    if verbose:
        print(f"  [QUERY EXPANSION: Original -> Rewritten: {expanded_text}]")

    return expanded_text, usage


def batch_expand_questions(anthropic_client, questions, verbose=False):
    """
    Expand all questions in a list for batch eval.

    Args:
        anthropic_client: Anthropic client instance
        questions: list of question dicts (each has q["question"])
        verbose: if True, print each expansion

    Returns: (expanded_texts: list[str], total_usage: dict)
    """
    expanded_texts = []
    total_input = 0
    total_output = 0

    for i, q in enumerate(questions):
        expanded, usage = expand_query(anthropic_client, q["question"], verbose=verbose)
        expanded_texts.append(expanded)
        total_input += usage["input_tokens"]
        total_output += usage["output_tokens"]
        print(f"  Expanded {i + 1}/{len(questions)} questions")
        if i < len(questions) - 1:
            time.sleep(1)

    total_usage = {"input_tokens": total_input, "output_tokens": total_output}
    return expanded_texts, total_usage

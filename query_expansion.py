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


EXPANSION_PROMPT = """You are a cricket assistant input handler. Your job has two steps.

STEP 1 — CLASSIFY the user input. If it falls into any of these three categories, output ONLY the token (no other text):

  [GREETING] — social pleasantries, small talk, or meta-questions about you.
    Examples: "hi", "hello", "hey there", "how are you", "thanks", "good morning",
    "who are you", "what can you do"

  [CRICKET_OFF_TOPIC] — the input IS about cricket but NOT about rules, laws,
    umpiring decisions, or match calculations.
    Examples: "which team is best", "who is the greatest batsman of all time",
    "teach me cricket", "when was the first cricket match", "tell me about Sachin",
    "what do you think of T20 cricket"

  [OFF_TOPIC] — the input is NOT about cricket at all. This includes general
    knowledge, code requests, toxic content, or random chatter. DO NOT use this
    token for greetings (use [GREETING]) or cricket-but-not-rules questions
    (use [CRICKET_OFF_TOPIC]).
    Examples: "write me a Python function", "what's the capital of France",
    "tell me a joke", "2 + 2 = ?"

STEP 2 — Only if NONE of the three tokens above apply, the input is a genuine
cricket rules/laws/match question. Rewrite it using formal MCC Laws of Cricket
terminology and output ONLY the rewritten question:
  - Replace casual terms with official ones (e.g., "batter" → "striker",
    "swats the ball" → "struck the ball a second time", "bowler chucks" →
    "bowler with a suspected illegal bowling action").
  - Do NOT add legal conclusions, assumed laws, or extra context.
  - Do NOT change the meaning.
  - If the question already uses formal terminology, return it unchanged.
  - If the question is about math/calculations (run rates, strike rates, etc.),
    return it unchanged.

Output rule: output EITHER one of the three tokens OR the rewritten question —
nothing else, no preamble, no explanation."""


CLASSIFICATION_TOKENS = ("[GREETING]", "[CRICKET_OFF_TOPIC]", "[OFF_TOPIC]")


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
            # Check for any classification token. We compare leniently so minor
            # model-side formatting variations (trailing period, quotes, newline)
            # do not cause us to miss a classification and fall through to
            # treating a greeting as a genuine question.
            token_candidate = expanded_text.strip(" \t\n.,!?:;'\"")
            if token_candidate in CLASSIFICATION_TOKENS:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                return token_candidate, usage
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

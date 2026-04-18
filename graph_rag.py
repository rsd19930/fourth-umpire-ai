"""
Fourth Umpire AI — Optional Graph RAG expansion (eval-only).

Given a set of retrieved chunks, parse cross-references out of their text
("See Law 21.8", "under Law 28.3", "See Appendix A", etc.) and append the
referenced chunks to the set for one-hop graph expansion.

This module is imported ONLY by run_eval.py under the --graph-rag flag.
app.py and query.py must never import from here — the Streamlit production
path stays on the standard pipeline.
"""

import json
import os
import re

from config import COLLECTIONS


CROSS_REF_RE = re.compile(
    r"\b(Law|Appendix)\s+(\d{1,2}|[A-D])(\.\d+(?:\.\d+)*)?",
    re.IGNORECASE,
)

_LOOKUP_CACHE = {}


def _load_lookup(collection_name):
    """Build a (law_number, section) -> chunk dict from the source JSON.

    Cached per-process so repeated calls across 51 questions stay cheap.
    Keys:
      (law_number, section)   — exact match, e.g. (28, "28.3")
      (law_number, None)      — law-level fallback, e.g. "Law 28" with no subsection
    `law_number` is int for numbered laws, str ("A"–"D") for appendices.
    """
    if collection_name in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[collection_name]

    json_file = next(
        c["json_file"] for c in COLLECTIONS.values() if c["name"] == collection_name
    )
    here = os.path.dirname(__file__)
    with open(os.path.join(here, json_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]

    lookup = {}
    for ch in chunks:
        law_num = ch.get("law_number")
        if law_num is None:
            continue
        section = ch.get("section")
        lookup[(law_num, section)] = ch
        if section is None:
            lookup[(law_num, None)] = ch

    # Fall back for bare "Law N" refs when no whole-law header chunk exists:
    # point (N, None) at the chunk with the lowest-numbered section. Most laws
    # in the broader collection have no header chunk, so this keeps "Law 34"
    # style refs resolvable without bloating context with every subsection.
    def _section_sort_key(s):
        if not s:
            return ()
        parts = s.split(".")
        out = []
        for p in parts:
            out.append(int(p) if p.isdigit() else p)
        return tuple(out)

    by_law = {}
    for ch in chunks:
        law_num = ch.get("law_number")
        if law_num is None or ch.get("section") is None:
            continue
        by_law.setdefault(law_num, []).append(ch)
    for law_num, subs in by_law.items():
        if (law_num, None) in lookup:
            continue
        subs_sorted = sorted(subs, key=lambda c: _section_sort_key(c.get("section")))
        lookup[(law_num, None)] = subs_sorted[0]

    _LOOKUP_CACHE[collection_name] = lookup
    return lookup


def extract_cross_refs(chunk_text):
    """Return a list of (law_number, section_or_None) tuples found in chunk_text.

    Deduplicates while preserving order. `law_number` is int for numbered laws
    (0–42) or uppercase str ("A"–"D") for appendices. Bare sibling references
    without the "Law" / "Appendix" prefix (e.g. "see 1.2") are deliberately
    not matched — deferred to v2.
    """
    refs = []
    for m in CROSS_REF_RE.finditer(chunk_text):
        kind = m.group(1).lower()
        num = m.group(2)
        dotted = m.group(3)

        if kind == "law":
            law_num = int(num) if num.isdigit() else num.upper()
        else:
            law_num = num.upper()

        section = None
        if dotted:
            section = f"{num}{dotted}"

        ref = (law_num, section)
        if ref not in refs:
            refs.append(ref)
    return refs


def graph_expand(retrieved_chunks, collection_name, max_new=5):
    """One-hop graph expansion: append cross-referenced chunks to the set.

    Looks at each retrieved chunk's text, extracts "Law N.M" / "Appendix X"
    references, resolves them via the JSON lookup, and appends any that are
    not already in `retrieved_chunks`. Capped at `max_new` new chunks per call.

    Returns:
        (expanded_chunks, log)
        log is a dict:
          refs_extracted        total refs found (including duplicates across chunks)
          refs_resolved         refs that resolved to a known chunk
          refs_already_present  resolved but already in retrieved set (skipped)
          refs_added            newly appended to the set
          added_ids             list of chunk ids that were appended
    """
    lookup = _load_lookup(collection_name)
    retrieved_ids = {c["id"] for c in retrieved_chunks}
    out = list(retrieved_chunks)
    log = {
        "refs_extracted": 0,
        "refs_resolved": 0,
        "refs_already_present": 0,
        "refs_added": 0,
        "added_ids": [],
    }

    for ch in retrieved_chunks:
        for ref in extract_cross_refs(ch["text"]):
            log["refs_extracted"] += 1

            target = lookup.get(ref)
            if target is None and ref[1] is not None:
                target = lookup.get((ref[0], None))
            if target is None:
                continue

            log["refs_resolved"] += 1

            if target["id"] in retrieved_ids:
                log["refs_already_present"] += 1
                continue

            if log["refs_added"] >= max_new:
                break

            out.append(target)
            retrieved_ids.add(target["id"])
            log["refs_added"] += 1
            log["added_ids"].append(target["id"])

        if log["refs_added"] >= max_new:
            break

    return out, log

"""
Context Resolver: Anchor-based passage retrieval and quote verification.

1. The LLM is asked to produce a short *anchor* (5-10 exact words from the paper)
   rather than a full 30-50 word verbatim quote.
2. This module finds that anchor in the raw source text using exact match first,
   then falling back to a fuzzy sliding-window search.
3. If the anchor is found, the surrounding passage (default ±300 chars) is
   returned as the verified context. If not found, the item is flagged as
   unverified so it can be reviewed or discarded rather than silently trusted.
"""

import re
import logging
from typing import Optional
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Minimum fuzzy match score (0.0-1.0) to accept an anchor as found.
# 0.82 catches minor OCR/whitespace differences while rejecting clear misses.
FUZZY_MATCH_THRESHOLD = 0.80


def find_anchor_in_text(
    anchor: str,
    source_text: str,
    context_window: int = 300,
) -> dict:
    """
    Locate an anchor phrase inside source_text and return the surrounding passage.

    Strategy:
      1. Case-insensitive exact substring match  (score = 1.0)
      2. Fuzzy sliding-window search via rapidfuzz (score = best ratio found)

    Parameters
    ----------
    anchor : str
        Short phrase (ideally 5-10 words) that the LLM identified as a locator.
    source_text : str
        Full raw text of the paper.
    context_window : int
        Number of characters to include on each side of the matched anchor.

    Returns
    -------
    dict with keys:
        found          (bool)   – whether a match above threshold was found
        score          (float)  – match quality 0.0–1.0
        context        (str|None) – the extracted passage, or None if not found
        char_start     (int)    – character offset of match start, or -1
    """
    if not anchor or not source_text:
        return {"found": False, "score": 0.0, "context": None, "char_start": -1}

    anchor_clean = anchor.strip()

    # 1. Exact match
    lower_text = source_text.lower()
    lower_anchor = anchor_clean.lower()
    idx = lower_text.find(lower_anchor)
    if idx != -1:
        context = _extract_window(source_text, idx, len(anchor_clean), context_window)
        return {"found": True, "score": 1.0, "context": context, "char_start": idx}

    # 2. Fuzzy sliding-window fallback 
    anchor_len = len(anchor_clean)

    # Step by ~20% of anchor length so we don't miss a match that straddles
    # a step boundary, without being so slow we scan every character.
    step = max(1, anchor_len // 5)

    best_score = 0.0
    best_idx = -1

    for i in range(0, max(0, len(source_text) - anchor_len + 1), step):
        window = source_text[i: i + anchor_len]
        score = fuzz.ratio(lower_anchor, window.lower()) / 100.0
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= FUZZY_MATCH_THRESHOLD:
        context = _extract_window(source_text, best_idx, anchor_len, context_window)
        return {
            "found": True,
            "score": round(best_score, 3),
            "context": context,
            "char_start": best_idx,
        }

    logger.debug(
        f"Anchor not found (best score {best_score:.2f} < {FUZZY_MATCH_THRESHOLD}): "
        f'"{anchor_clean[:60]}..."'
    )
    return {"found": False, "score": round(best_score, 3), "context": None, "char_start": -1}


def _extract_window(text: str, match_start: int, match_len: int, window: int) -> str:
    """Extract text around a match, trimming to sentence/word boundaries."""
    start = max(0, match_start - window)
    end = min(len(text), match_start + match_len + window)
    snippet = text[start:end].strip()
    # Prefix/suffix ellipsis so the reader knows the passage was trimmed
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


# ---------------------------------------------------------------------------
# Per-extractor resolver helpers
# ---------------------------------------------------------------------------

def resolve_design_strategy_contexts(
    results: dict,
    source_text: str,
    context_window: int = 300,
) -> dict:
    """
    Replace each design strategy's raw LLM anchor with a verified context passage.

    Modifies results in-place and returns it.
    Adds 'anchor_verified' (bool) and 'anchor_match_score' (float) to each strategy.
    'context' becomes the resolved passage (str) or None if unverified.
    """
    strategies = results.get("design_strategies", [])
    verified_count = 0
    unverified_count = 0

    for strategy in strategies:
        anchor = strategy.pop("anchor", None)  # remove raw anchor field
        strategy["anchor_text"] = anchor       # keep it for debugging

        if not anchor:
            strategy["context"] = None
            strategy["anchor_verified"] = False
            strategy["anchor_match_score"] = 0.0
            unverified_count += 1
            continue

        result = find_anchor_in_text(anchor, source_text, context_window)
        strategy["context"] = result["context"]
        strategy["anchor_verified"] = result["found"]
        strategy["anchor_match_score"] = result["score"]

        if result["found"]:
            verified_count += 1
        else:
            unverified_count += 1
            logger.warning(
                f"[Design Strategy] Anchor not verified "
                f"(score={result['score']:.2f}) for '{strategy.get('name', '?')}': "
                f'"{anchor[:60]}"'
            )

    logger.info(
        f"Design strategy context resolution: "
        f"{verified_count} verified, {unverified_count} unverified "
        f"out of {len(strategies)} total."
    )
    return results


def resolve_ecosystem_service_contexts(
    results: dict,
    source_text: str,
    context_window: int = 300,
) -> dict:
    """
    Replace each ecosystem service's raw LLM anchor with a verified context passage.

    Same contract as resolve_design_strategy_contexts.
    """
    services = results.get("ecosystem_services", [])
    verified_count = 0
    unverified_count = 0

    for service in services:
        anchor = service.pop("anchor", None)
        service["anchor_text"] = anchor

        if not anchor:
            service["context"] = None
            service["anchor_verified"] = False
            service["anchor_match_score"] = 0.0
            unverified_count += 1
            continue

        result = find_anchor_in_text(anchor, source_text, context_window)
        service["context"] = result["context"]
        service["anchor_verified"] = result["found"]
        service["anchor_match_score"] = result["score"]

        if result["found"]:
            verified_count += 1
        else:
            unverified_count += 1
            logger.warning(
                f"[Ecosystem Service] Anchor not verified "
                f"(score={result['score']:.2f}) for '{service.get('name', '?')}': "
                f'"{anchor[:60]}"'
            )

    logger.info(
        f"Ecosystem service context resolution: "
        f"{verified_count} verified, {unverified_count} unverified "
        f"out of {len(services)} total."
    )
    return results


def resolve_entity_contexts(
    results: dict,
    source_text: str,
    context_window: int = 300,
) -> dict:
    """
    Verify the context snippets already embedded in each entity field dict.

    The entity extractor returns fields in the form:
        {"value": "Rio de Janeiro", "context": "situated in the city of Rio de Janeiro"}

    These short snippets (max 30 words) act as anchors directly — this function
    searches for each one in the source text and adds two sibling keys:
        "context_verified"    (bool)  – True if the snippet was found
        "context_match_score" (float) – fuzzy match quality 0.0–1.0

    The "value" and "context" keys are never modified, so the entity extractor's
    own generate_report() continues to work without any changes.

    Fields whose value is null or whose context is null are skipped silently.
    """
    entities = results.get("entities", [])
    verified_count = 0
    unverified_count = 0
    skipped_count = 0

    for entity in entities:
        for field_name, field_data in entity.items():
            # Only process the nested {"value": ..., "context": ...} dicts
            if not isinstance(field_data, dict) or "context" not in field_data:
                continue

            snippet = field_data.get("context")
            value   = field_data.get("value")

            # Nothing to verify if either is absent
            if not snippet or value is None:
                field_data["context_verified"] = None
                field_data["context_match_score"] = None
                skipped_count += 1
                continue

            result = find_anchor_in_text(snippet, source_text, context_window)
            field_data["context_verified"]    = result["found"]
            field_data["context_match_score"] = result["score"]

            if result["found"]:
                verified_count += 1
            else:
                unverified_count += 1
                logger.warning(
                    f"[Entity / {field_name}] Context not verified "
                    f"(score={result['score']:.2f}): \"{snippet[:60]}\""
                )

    logger.info(
        f"Entity context verification: "
        f"{verified_count} verified, {unverified_count} unverified, "
        f"{skipped_count} skipped (null) "
        f"across {len(entities)} entities."
    )
    return results

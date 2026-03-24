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
    lower_anchor = anchor_clean.lower()
    lower_text   = source_text.lower()

    # 1. Exact match
    idx = lower_text.find(lower_anchor)
    if idx != -1:
        context = _extract_window(source_text, idx, len(anchor_clean))
        return {"found": True, "score": 1.0, "context": context, "char_start": idx}

    # 2. partial_ratio fuzzy match
    # fuzz.partial_ratio(short, long) finds the best-aligning substring of
    # `long` that matches `short`, handling OCR noise and minor whitespace
    # differences without the alignment errors of a manual sliding window.
    score = fuzz.partial_ratio(lower_anchor, lower_text) / 100.0

    if score < FUZZY_MATCH_THRESHOLD:
        logger.debug(
            f"Anchor not found (partial_ratio={score:.3f} < {FUZZY_MATCH_THRESHOLD}): "
            f'"{anchor_clean[:60]}"'
        )
        return {"found": False, "score": round(score, 3), "context": None, "char_start": -1}

    # partial_ratio confirms a match exists but doesn't return its position,
    # so we do a two-pass scan to locate it for window extraction.
    # Pass 1 (coarse): step = anchor_len // 10 to find approximate region.
    # Pass 2 (fine):   step = 1 within +/-coarse_step of the best coarse position.
    anchor_len   = len(anchor_clean)
    coarse_step  = max(1, anchor_len // 10)
    best_ratio   = 0.0
    best_idx     = 0

    for i in range(0, max(1, len(source_text) - anchor_len + 1), coarse_step):
        s = fuzz.ratio(lower_anchor, lower_text[i: i + anchor_len]) / 100.0
        if s > best_ratio:
            best_ratio = s
            best_idx   = i

    refine_start = max(0, best_idx - coarse_step)
    refine_end   = min(len(source_text) - anchor_len + 1, best_idx + coarse_step + 1)
    for i in range(refine_start, refine_end):
        s = fuzz.ratio(lower_anchor, lower_text[i: i + anchor_len]) / 100.0
        if s > best_ratio:
            best_ratio = s
            best_idx   = i

    context = _extract_window(source_text, best_idx, anchor_len)
    return {
        "found":      True,
        "score":      round(score, 3),   # report partial_ratio (true match quality)
        "context":    context,
        "char_start": best_idx,
    }

def _extract_window(text: str, match_start: int, match_len: int) -> str:
    """Extract paragraph containing a match."""
    # Walk backward to paragraph start
    para_start = text.rfind("\n\n", 0, match_start)
    para_start = 0 if para_start == -1 else para_start + 2  # skip the \n\n itself

    # Walk forward to paragraph end
    para_end = text.find("\n\n", match_start + match_len)
    para_end = len(text) if para_end == -1 else para_end

    snippet = text[para_start:para_end].strip()

    if para_start > 0:
        snippet = "…" + snippet
    if para_end < len(text):
        snippet = snippet + "…"

    return snippet


# ---------------------------------------------------------------------------
# Per-extractor resolver helpers
# ---------------------------------------------------------------------------

def resolve_design_strategy_contexts(
    results: dict,
    source_text: str,
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

        result = find_anchor_in_text(anchor, source_text)
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

        result = find_anchor_in_text(anchor, source_text)
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

            result = find_anchor_in_text(snippet, source_text)
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

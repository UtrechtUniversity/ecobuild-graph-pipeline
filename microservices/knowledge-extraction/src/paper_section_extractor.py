"""
Paper section extractor — markdown-heading-first, pluggable classification.

Flow
----
1. Parse ATX headings (# / ## / ###) out of the pymupdf4llm markdown string.
2. Slice the markdown into raw sections keyed by heading.
3. Classify headings using either:
   a. Any LlamaIndex ``BaseEmbedding`` (e.g. OllamaEmbedding) — encodes
      headings and canonical labels, picks the closest label by cosine
      similarity.
   b. (Not in use) An LLM prompt fallback when no embed_model is supplied to PaperSectionExtractor.
4. Merge slices that share a canonical name and return
   {canonical_category: text}.

Canonical categories
--------------------
abstract · introduction · methods · results · discussion ·
conclusion · references · keywords · unclassified
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document

from .llama_index_interface import LlamaIndexInterface

logger = logging.getLogger(__name__)

# ── Public constants ──────────────────────────────────────────────────────────

CANONICAL_SECTIONS: dict[str, str] = {
    "abstract": "summary overview of the study objectives findings and conclusions",
    "introduction": "background context motivation research gap and objectives",
    "methods": "methodology data collection experimental setup study design procedure",
    "results": "findings measurements data analysis outcomes performance assessment",
    "discussion": "interpretation of results comparison with literature implications",
    "conclusion": "summary of findings limitations future work recommendations",
    "references": "bibliography citations list of sources",
    "keywords": "index terms key concepts",
    "unclassified": "miscellaneous content that does not fit other sections",
}

# Regex that matches any ATX heading (# … ####) at the start of a line.
_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_markdown_formatting(text: str) -> str:
    """Remove markdown syntax that adds noise for LLM extraction.

    Strips bold/italic markers, table rows, link syntax (keeping the label),
    and collapses runs of blank lines to at most two newlines.
    Call this on body text only — never on the raw heading string.
    """
    # Remove "ommitted image" paragraphs
    text = re.sub(r'^\*\*==>.*?(?:\n\s*\n|\Z)', '', text, flags=re.MULTILINE | re.DOTALL)
    # Bold / italic: **text** or *text* → text
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)
    # Markdown table rows: | cell | cell | → stripped entirely
    text = re.sub(r'\|.+\|', '', text)
    # Inline links: [label](url) → label
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    # Collapse 3+ consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

## Prompt for LLM fallback
def _build_classification_prompt(heading_context: dict[str, str]) -> str:
    canonical = ", ".join(CANONICAL_SECTIONS)
    header_list = "\n".join(f"- Header: {h}\n  Context: {ctx}" for h, ctx in heading_context.items())
    return (
        f"""
        You are classifying academic paper section headers into canonical categories.
        Canonical categories: {canonical}
        For each header below, assign exactly one canonical category based on the header and its context.
        Reply ONLY with a JSON object mapping each header to its category. No preamble.

        Headers and Context:
        {header_list}

        JSON output:
        """
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Embedding-based classifier (any LlamaIndex BaseEmbedding) ────────────────

class _EmbeddingClassifier:
    """
    Wraps any LlamaIndex ``BaseEmbedding`` (e.g. OllamaEmbedding) to perform
    section-heading classification via cosine similarity against the canonical
    label set.  No generative model is involved — results are deterministic.
    """

    def __init__(self, embed_model: BaseEmbedding) -> None:
        self._embed_model = embed_model
        # Pre-compute embeddings for all canonical labels once at startup.
        self._label_embeddings: dict[str, list[float]] = {
            label: self._embed_model.get_text_embedding(description)
            for label, description in CANONICAL_SECTIONS.items()
        }
        logger.info(
            "EmbeddingClassifier ready (%s canonical labels pre-embedded).",
            len(self._label_embeddings),
        )

    def classify(self, heading_context: dict[str, str]) -> dict[str, str]:
        """Return ``{heading: canonical_category}`` using cosine similarity."""
        result: dict[str, str] = {}
        for heading, context in heading_context.items():
            try:
                text_to_embed = f"{heading}\n{context}" if context else heading
                h_emb = self._embed_model.get_text_embedding(text_to_embed)
                best_label = max(
                    self._label_embeddings,
                    key=lambda label: _cosine_similarity(h_emb, self._label_embeddings[label]),
                )
                result[heading] = best_label
            except Exception as exc:
                logger.debug("Embedding classify failed for %r: %s", heading, exc)
                result[heading] = "unclassified"
        return result


# ── Main class ────────────────────────────────────────────────────────────────


class PaperSectionExtractor:
    """
    Converts a pymupdf4llm markdown string into a canonical-section dict.

    Classification strategy (resolved at construction time):

    1. If ``embed_model`` is supplied → embedding-based cosine-similarity
       classification (deterministic, no free-form generation).
    2. Otherwise → single LLM prompt call through ``LlamaIndexInterface``.

    Parameters
    ----------
    llm:
        Any LlamaIndex ``LLM`` instance. Used as the classifier when no
        embed_model is supplied.
    embed_model:
        Optional LlamaIndex ``BaseEmbedding`` (e.g. ``OllamaEmbedding``).
        When provided, headings are classified via cosine similarity against
        pre-embedded canonical labels — no LLM call is made for classification.
    """

    def __init__(self, llm: LLM, embed_model: Optional[BaseEmbedding] = None) -> None:
        self.llm_interface = LlamaIndexInterface(llm)
        self._embed_classifier: _EmbeddingClassifier | None = None

        if embed_model is not None:
            self._embed_classifier = _EmbeddingClassifier(embed_model)
            logger.info("Section classification: embedding cosine similarity (OllamaEmbedding).")
        else:
            logger.info("Section classification: LLM prompt (LlamaIndex).")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_sections_from_markdown(self, md_text: str) -> dict[str, str]:
        """
        Parse *md_text* (pymupdf4llm output) and return a section dict.

        Returns
        -------
        dict[str, str]
            ``{canonical_category: concatenated_body_text}``
            Unrecognised or heading-less text lands in ``"unclassified"``.
        """
        matches = list(_HEADING_RE.finditer(md_text))
        
        nodes = []
        if matches and matches[0].start() > 0:
            pre_text = md_text[0:matches[0].start()].strip()
            if pre_text:
                nodes.append({"heading": "", "text": pre_text})
        elif not matches:
             nodes.append({"heading": "", "text": md_text})
                
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(md_text)
            text = md_text[start_pos:end_pos].strip()
            nodes.append({"heading": heading, "text": text})

        # Collect unique non-empty headings and their context.
        heading_context: dict[str, str] = {}
        for node in nodes:
            heading = node["heading"]
            if heading and heading not in heading_context:
                clean_text = _strip_markdown_formatting(node["text"]).strip()
                context = clean_text.split('\n\n')[0] if clean_text else ""
                heading_context[heading] = context

        heading_map = self._classify_headings(heading_context) if heading_context else {}
        
        # Merge bodies under their canonical name.
        merged: dict[str, list[str]] = {c: [] for c in CANONICAL_SECTIONS}

        for node in nodes:
            heading = node["heading"]
            canonical = heading_map.get(heading, "unclassified") if heading else "unclassified"
            body = _strip_markdown_formatting(node["text"])
            if not body:
                continue
            merged[canonical].append(body)

        # Join and drop empty canonicals.
        result: dict[str, str] = {}
        for canonical, parts in merged.items():
            joined = "\n\n".join(parts).strip()
            if joined:
                result[canonical] = joined

        logger.info(
            "Extracted %d canonical sections: %s",
            len(result),
            list(result.keys()),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _classify_headings(self, heading_context: dict[str, str]) -> dict[str, str]:
        """
        Dispatch to embedding-based classifier or LLM depending on config.

        Returns ``{raw_heading: canonical}`` for every entry in *heading_context*.
        """
        if self._embed_classifier is not None:
            logger.debug("Classifying %d headings via embedding cosine similarity.", len(heading_context))
            return self._embed_classifier.classify(heading_context)

        logger.debug("Classifying %d headings via LLM prompt.", len(heading_context))
        return self._classify_headings_via_llm(heading_context)

    def _classify_headings_via_llm(self, heading_context: dict[str, str]) -> dict[str, str]:
        """
        Single LLM call: returns ``{raw_heading: canonical}``.
        Invalid categories are coerced to ``"unclassified"``.
        """
        prompt = _build_classification_prompt(heading_context)
        raw_response = self.llm_interface.query(prompt)
        mapping: dict = self.llm_interface.extract_json(raw_response)

        validated: dict[str, str] = {}
        for heading in heading_context:
            raw_cat = mapping.get(heading, "unclassified")
            canonical = raw_cat.strip().lower() if isinstance(raw_cat, str) else "unclassified"
            if canonical not in CANONICAL_SECTIONS:
                logger.debug(
                    "LLM returned unknown category %r for heading %r → unclassified",
                    canonical,
                    heading,
                )
                canonical = "unclassified"
            validated[heading] = canonical

        logger.debug("LLM heading classification: %s", validated)
        return validated
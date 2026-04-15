"""
PaperLabeler
============
Scans a preprocessed academic paper and labels whether each of a fixed set of
study-type categories is present.

Two strategies are provided:

  LLMLabeler       - single structured LLM call per paper (recommended)
  EmbeddingLabeler - cosine-similarity scan over chunks (fast pre-filter)

Both implement the same interface:
    labeler.label(preprocessed: dict) -> LabelResult

where ``preprocessed`` is the dict returned by PaperPreprocessor.preprocess_pdf().

The LLM strategy uses the anchor-resolution infrastructure from context_resolver
to verify every YES answer before returning it, so hallucinated citations are
caught and downgraded to UNVERIFIED rather than silently trusted.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from .context_resolver import find_anchor_in_text
from .llama_index_interface import LlamaIndexInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label registry
# ---------------------------------------------------------------------------

class Label(str, Enum):
    """
    Canonical study-type labels.
    Adjust display names here; keys are used as dict keys throughout.
    """
    MODELING_CASE_BUILDING      = "modeling_case_study_building"
    EMPIRICAL_CASE_BUILDING     = "empirical_case_study_building"
    GOVERNANCE                  = "governance"
    EMPIRICAL_URBAN             = "empirical_urban_environment"
    MODELING_URBAN              = "modeling_urban_environment"
    EMPIRICAL_NON_URBAN         = "empirical_non_urban_environment"
    MODELING_NON_URBAN          = "modeling_non_urban_environment"


# Human-readable descriptions used in LLM prompts AND as embedding queries.
LABEL_DESCRIPTIONS: dict[Label, str] = {
    Label.MODELING_CASE_BUILDING: (
        "The paper presents a computational or simulation model applied to a specific or theoretical building as a case study."
    ),
    Label.EMPIRICAL_CASE_BUILDING: (
        "The paper presents measured, monitored, or survey-based empirical data collected from a specific building or small set of buildings as a case study."
    ),
    Label.GOVERNANCE: (
        "The paper discusses governance aspects: barriers, opportunities, organisational changes, stakeholder roles, policy instruments," 
        "or urban planning frameworks related to the built environment or environmental sustainability."
    ),
    Label.EMPIRICAL_URBAN: (
        "The paper presents empirical data (measurements, surveys, remote sensing, "
        "statistics) collected at the urban or city scale."
    ),
    Label.MODELING_URBAN: (
        "The paper presents a computational or simulation model applied at the urban "
        "or city scale."
    ),
    Label.EMPIRICAL_NON_URBAN: (
        "The paper presents empirical data collected in non-urban environments: "
        "rural areas, forests, agricultural land, wetlands, peri-urban fringe, etc."
    ),
    Label.MODELING_NON_URBAN: (
        "The paper presents a computational or simulation model applied to "
        "non-urban environments: rural, natural, agricultural, or peri-urban settings."
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    YES        = "YES"
    NO         = "NO"
    UNVERIFIED = "UNVERIFIED"   # LLM returned YES but anchor was not confirmed; hallucination insurance


@dataclass
class LabelDecision:
    label:        Label
    verdict:      Verdict
    anchor_text:  Optional[str]   = None   # raw anchor phrase from LLM
    context:      Optional[str]   = None   # resolved passage from source text
    match_score:  float           = 0.0    # anchor resolution quality
    rationale:    Optional[str]   = None   # brief LLM explanation (LLM strategy only)
    chunk_score:  float           = 0.0    # embedding cosine similarity (embedding strategy only)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"]   = self.label.value
        d["verdict"] = self.verdict.value
        return d


@dataclass
class LabelResult:
    paper_path: str
    decisions:  list[LabelDecision] = field(default_factory=list)
    strategy:   str = "unknown"

    def to_dict(self) -> dict:
        return {
            "paper_path": self.paper_path,
            "strategy":   self.strategy,
            "decisions":  [d.to_dict() for d in self.decisions],
        }

    def summary(self) -> dict[str, str]:
        """Quick {label_value: verdict} mapping for logging."""
        return {d.label.value: d.verdict.value for d in self.decisions}


# ---------------------------------------------------------------------------
# LLM Labeler
# ---------------------------------------------------------------------------

def _build_prompt(paper_text: str, label_descriptions: dict[Label, str]) -> str:
    """
    Build a single self-contained prompt (system instructions + paper text).
    Returns a JSON *object* keyed by label — compatible with LlamaIndexInterface's
    extract_json() which matches {.*} rather than [...].
    Output shape:
    {
      "<label_key>": {"verdict": "YES"|"NO", "anchor": "<phrase>"|null, "rationale": "<sentence>"|null},
      ...
    }
    """
    cats = "\n".join(
        f'  "{lbl.value}": {desc}'
        for lbl, desc in label_descriptions.items()
    )
 
    # Concrete output skeleton so the model knows exact keys and order
    skeleton_lines = []
    for i, lbl in enumerate(label_descriptions):
        comma = "," if i < len(label_descriptions) - 1 else ""
        skeleton_lines.append(
            f'  "{lbl.value}": {{"verdict": "YES"|"NO", "anchor": "<5-10 word phrase>"|null, "rationale": "<one sentence>"|null}}{comma}'
        )
    skeleton = "{\n" + "\n".join(skeleton_lines) + "\n}"
 
    return textwrap.dedent(f"""\
        You are a precise academic-paper classifier.
        Classify the paper below against each category. Return ONLY valid JSON - no prose, no markdown fences.
 
        RULES:
        - Default verdict is NO. Only mark YES if the evidence is clear and direct.
        - For every YES you MUST copy a short verbatim phrase (5-10 words) from the paper as the anchor.
        - For NO answers set anchor and rationale to null.
        - verdict must be exactly "YES" or "NO".
        - Every label key must appear in the output object.
 
        CATEGORIES:
        {cats}
 
        REQUIRED OUTPUT FORMAT:
        {skeleton}
 
        PAPER TEXT:
        {paper_text}
    """)


class LLMLabeler:
    """
    Classifies a paper by sending its full text to an LLM in a single call,
    then resolves every YES anchor against the source text.

    Parameters
    ----------
    llm:
        Any LlamaIndex-compatible LLM (OpenAI, Anthropic, local via Ollama, …).
    labels:
        Subset of Label enums to classify. Defaults to all seven.
    fuzzy_threshold:
        Minimum anchor resolution score to accept a YES as verified.
        Passed through to find_anchor_in_text (which uses its own internal
        threshold, so this acts as an *additional* post-hoc filter).
    """

    def __init__(
        self,
        llm: LLM,
        labels: Optional[list[Label]] = None,
        fuzzy_threshold: float = 0.80,
    ) -> None:
        self._interface = LlamaIndexInterface(llm)
        self.llm = llm                              # Keeping for now, for HybridLabeler compatibility
        self.labels = labels or list(Label)
        self.fuzzy_threshold = fuzzy_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def label(self, preprocessed: dict, section_name: str = "unknown") -> LabelResult:
        """
        Parameters
        ----------
        preprocessed:
            Dict returned by PaperPreprocessor.preprocess_pdf().
            Must contain 'raw_text_path' and 'pdf_path'.
        """
        if isinstance(preprocessed, str):
            paper_path  = section_name
            source_text = preprocessed
        else:
            paper_path  = preprocessed.get("pdf_path", "unknown")
            source_text = self._load_source_text(preprocessed)
        active_descs = {lbl: LABEL_DESCRIPTIONS[lbl] for lbl in self.labels}
        if len(source_text) < 50:
            logger.info("[LLMLabeler] Skipping '%s' (%d chars) - too short", paper_path, len(source_text))
            return LabelResult(
                paper_path=paper_path,
                decisions=[],
                strategy="llm",
            )

        logger.info("[LLMLabeler] Labeling '%s' (%d chars)", paper_path, len(source_text))

        raw_decisions = self._call_llm(source_text, active_descs)
        decisions     = self._resolve_anchors(raw_decisions, source_text)

        result = LabelResult(
            paper_path=paper_path,
            decisions=decisions,
            strategy="llm",
        )
        logger.info("[LLMLabeler] Done. Summary: %s", result.summary())
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_source_text(self, preprocessed: dict) -> str:
        """Load the plain-text version saved by PaperPreprocessor."""
        raw_path = preprocessed.get("raw_text_path")
        if raw_path:
            try:
                from pathlib import Path
                return Path(raw_path).read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not read raw_text_path (%s): %s", raw_path, exc)
        # Fall back to the LlamaIndex Document text
        doc = preprocessed.get("document")
        if doc:
            return doc.text
        return ""

    def _call_llm(
        self,
        source_text: str,
        label_descriptions: dict[Label, str],
    ) -> dict[str, dict]:
        """Send the classification prompt and parse the JSON response."""
        prompt = _build_prompt(source_text, label_descriptions)
        raw_text = self._interface.query(prompt)
        parsed   = self._interface.extract_json(raw_text)
 
        if not parsed:
            logger.error("[LLMLabeler] extract_json returned empty — defaulting all to NO.")
            return {
                lbl.value: {"verdict": "NO", "anchor": None, "rationale": None}
                for lbl in label_descriptions
            }
 
        return parsed

    def _resolve_anchors(
        self,
        raw_decisions: dict[str, dict],
        source_text: str,
    ) -> list[LabelDecision]:
        """
        For each YES decision, locate the anchor in the source text.
        Downgrades to UNVERIFIED if the anchor cannot be found above threshold.
        """
        # The LLM output is already keyed by the label string. Fall back gracefully if it hallucinates a list.
        if isinstance(raw_decisions, list):
            by_label_key = {d.get("label", ""): d for d in raw_decisions if isinstance(d, dict)}
        else:
            by_label_key = raw_decisions

        decisions: list[LabelDecision] = []

        for lbl in self.labels:
            raw = by_label_key.get(lbl.value) or {}
            verdict_str = (raw.get("verdict") or "NO").strip().upper()
            anchor      = raw.get("anchor") or None
            rationale   = raw.get("rationale") or None

            if verdict_str != "YES" or not anchor:
                decisions.append(LabelDecision(
                    label=lbl,
                    verdict=Verdict.NO,
                    rationale=rationale,
                ))
                continue

            # Resolve anchor → context passage
            resolution = find_anchor_in_text(anchor, source_text)
            verified   = resolution["found"] and resolution["score"] >= self.fuzzy_threshold

            if verified:
                verdict = Verdict.YES
            else:
                verdict = Verdict.UNVERIFIED
                logger.warning(
                    "[LLMLabeler] YES for '%s' could not be verified "
                    "(score=%.2f, anchor='%s')",
                    lbl.value, resolution["score"], anchor[:60],
                )

            decisions.append(LabelDecision(
                label=lbl,
                verdict=verdict,
                anchor_text=anchor,
                context=resolution["context"],
                match_score=resolution["score"],
                rationale=rationale,
            ))

        return decisions


# ---------------------------------------------------------------------------
# Embedding Labeler  (fast pre-filter / standalone)
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _cosine(a: list[float], b: list[float]) -> float:
    if not _NUMPY_AVAILABLE:
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x * x for x in a) ** 0.5
        nb  = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)
    import numpy as np
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))


class EmbeddingLabeler:
    """
    Classifies a paper by comparing label-description embeddings against
    chunk embeddings from the LlamaIndex nodes stored in the preprocessed dict.

    A label is marked YES when the maximum cosine similarity across all chunks
    exceeds *threshold*.  The best-matching chunk is used as the context.

    Parameters
    ----------
    embed_model:
        Any LlamaIndex BaseEmbedding.
    threshold:
        Cosine similarity above which a label is considered present.
        Tune this per-label if needed (see *per_label_thresholds*).
    chunk_size / chunk_overlap:
        Character-level chunking used when LlamaIndex nodes are absent.
    per_label_thresholds:
        Optional overrides, e.g. {Label.GOVERNANCE: 0.45}.
    """

    DEFAULT_THRESHOLD = 0.42   # calibrate on a labelled sample

    def __init__(
        self,
        embed_model: BaseEmbedding,
        labels: Optional[list[Label]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        per_label_thresholds: Optional[dict[Label, float]] = None,
    ) -> None:
        self.embed_model          = embed_model
        self.labels               = labels or list(Label)
        self.threshold            = threshold
        self.chunk_size           = chunk_size
        self.chunk_overlap        = chunk_overlap
        self.per_label_thresholds = per_label_thresholds or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def label(self, preprocessed: dict) -> LabelResult:
        paper_path = preprocessed.get("pdf_path", "unknown")
        logger.info("[EmbeddingLabeler] Classifying '%s'", paper_path)

        # 1. Get text chunks
        chunks = self._get_chunks(preprocessed)
        if not chunks:
            logger.warning("[EmbeddingLabeler] No chunks found; returning all NO.")
            return LabelResult(
                paper_path=paper_path,
                decisions=[LabelDecision(label=lbl, verdict=Verdict.NO) for lbl in self.labels],
                strategy="embedding",
            )

        # 2. Embed all chunks (batch for efficiency)
        chunk_embeddings = self.embed_model.get_text_embedding_batch(chunks)

        # 3. Embed all label descriptions
        label_queries = {lbl: LABEL_DESCRIPTIONS[lbl] for lbl in self.labels}
        query_texts   = list(label_queries.values())
        query_embeds  = self.embed_model.get_text_embedding_batch(query_texts)

        # 4. Score and decide
        decisions: list[LabelDecision] = []
        for lbl, q_emb in zip(self.labels, query_embeds):
            scores     = [_cosine(q_emb, c_emb) for c_emb in chunk_embeddings]
            best_idx   = max(range(len(scores)), key=lambda i: scores[i])
            best_score = scores[best_idx]
            cutoff     = self.per_label_thresholds.get(lbl, self.threshold)

            if best_score >= cutoff:
                best_chunk = chunks[best_idx]
                decisions.append(LabelDecision(
                    label=lbl,
                    verdict=Verdict.YES,
                    context=best_chunk,
                    chunk_score=round(best_score, 4),
                ))
            else:
                decisions.append(LabelDecision(
                    label=lbl,
                    verdict=Verdict.NO,
                    chunk_score=round(best_score, 4),
                ))

        result = LabelResult(paper_path=paper_path, decisions=decisions, strategy="embedding")
        logger.info("[EmbeddingLabeler] Done. Summary: %s", result.summary())
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_chunks(self, preprocessed: dict) -> list[str]:
        """
        Prefer LlamaIndex nodes if available (already split by the preprocessor).
        Falls back to simple character-level chunking of the plain text.
        """
        nodes = preprocessed.get("nodes")   # set this key if you store nodes upstream
        if nodes:
            return [n.get_content() for n in nodes if n.get_content().strip()]

        # Character-level fallback
        from pathlib import Path
        raw_path = preprocessed.get("raw_text_path")
        if raw_path:
            try:
                text = Path(raw_path).read_text(encoding="utf-8")
            except Exception:
                doc = preprocessed.get("document")
                text = doc.text if doc else ""
        else:
            doc = preprocessed.get("document")
            text = doc.text if doc else ""

        return list(self._char_chunk(text))

    def _char_chunk(self, text: str):
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            yield text[start:end]
            start += self.chunk_size - self.chunk_overlap


# ---------------------------------------------------------------------------
# Hybrid: Embedding pre-filter → LLM confirmation
# ---------------------------------------------------------------------------

class HybridLabeler:
    """
    Two-stage pipeline:
      1. EmbeddingLabeler quickly identifies *candidate* labels (low threshold).
      2. LLMLabeler is called only for candidate labels, saving tokens on clear NO's.

    Parameters
    ----------
    llm_labeler:        A configured LLMLabeler instance.
    embedding_labeler:  A configured EmbeddingLabeler with a *low* threshold
                        (e.g. 0.35) so it acts as a liberal gate, not a classifier.
    """

    def __init__(self, llm_labeler: LLMLabeler, embedding_labeler: EmbeddingLabeler) -> None:
        self.llm_labeler       = llm_labeler
        self.embedding_labeler = embedding_labeler

    def label(self, preprocessed: dict) -> LabelResult:
        paper_path = preprocessed.get("pdf_path", "unknown")

        emb_result  = self.embedding_labeler.label(preprocessed)
        candidates  = [d.label for d in emb_result.decisions if d.verdict == Verdict.YES]
        hard_nos    = [d for d in emb_result.decisions if d.verdict != Verdict.YES]

        logger.info(
            "[HybridLabeler] Embedding stage: %d candidates, %d hard NO's",
            len(candidates), len(hard_nos),
        )

        if not candidates:
            return LabelResult(paper_path=paper_path, decisions=hard_nos, strategy="hybrid")

        # Run LLM only on candidates
        restricted_labeler = LLMLabeler(
            llm=self.llm_labeler.llm,
            labels=candidates,
            fuzzy_threshold=self.llm_labeler.fuzzy_threshold,
        )
        llm_result = restricted_labeler.label(preprocessed)

        all_decisions = hard_nos + llm_result.decisions
        # Restore original label order
        order = {lbl: i for i, lbl in enumerate(Label)}
        all_decisions.sort(key=lambda d: order.get(d.label, 99))

        return LabelResult(paper_path=paper_path, decisions=all_decisions, strategy="hybrid")

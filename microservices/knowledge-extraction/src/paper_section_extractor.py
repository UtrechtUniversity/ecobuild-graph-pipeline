from __future__ import annotations
 
import re
import collections
from typing import Optional
 
# PDFPlumber — used only for header detection (reliable font metadata per char)
import pdfplumber
 
# PyMuPDF — used only for clean text extraction
import fitz
 
from rapidfuzz import fuzz, process
import statistics
 
 
SECTION_ALIASES: dict[str, list[str]] = {
    "metadata":     [],
    "abstract":     ["abstract", "summary", "executive summary", "synopsis",
                     "description"],
    "keywords":     ["keywords", "key words", "index terms", "key terms"],
    "introduction": ["introduction", "background", "motivation",
                     "background and motivation", "background & motivation",
                     "overview"],
    "related_work": ["related work", "related works", "literature review",
                     "prior work", "previous work"],
    "methods":      ["methods", "method", "methodology", "materials and methods",
                     "materials & methods", "proposed method", "proposed approach",
                     "approach", "model", "framework", "system", "architecture"],
    "results":      ["results", "evaluation", "findings", "empirical results",
                     "benchmarks", "results and discussion"],
    "discussion":   ["discussion", "analysis", "discussion and analysis",
                     "discussion and conclusion", "discussion & conclusion"],
    "conclusion":   ["conclusion", "conclusions", "concluding remarks",
                     "summary and conclusion", "future work",
                     "conclusion and future work"],
    "acknowledgements": ["acknowledgements", "acknowledgments", "acknowledgement",
                         "acknowledgment"],
    "references":   ["references", "bibliography", "notes and references",
                     "works cited", "citations", "literature",
                     "reference list", "bibliographic references"],
    "appendix":     ["appendix", "appendices", "supplementary material",
                     "supplementary", "supplemental material", "supplemental materials",
                     "additional material", "online supplement"],
}
 
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in SECTION_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canon
 
_ALL_ALIASES: list[str] = list(_ALIAS_TO_CANONICAL.keys())
 
 
def _normalize(text: str) -> str:
    t = text.strip()
    # collapse spaced letters: "A B S T R A C T" -> "abstract"
    if re.fullmatch(r"(?:[A-Za-z]\s+){2,}[A-Za-z]", t):
        t = t.replace(" ", "")
    # strip leading section numbers: "1.", "2.3 ", "A. ", "IV. "
    t = re.sub(r"^\s*(?:\d+\.)+\s*|^[a-zA-Z]{1,4}\.\s+", "", t)
    t = re.sub(r"\s+", " ", t).casefold()
    return t.strip()
 
 
def _chars_to_lines(chars: list[dict]) -> list[list[dict]]:
    """
    Group pdfplumber char dicts into lines by rounding their y0 coordinate.
    Characters whose y0 values are within 1 pt of each other are the same line.
    Returns lines sorted top-to-bottom (descending y0 in PDF space).
    """
    buckets: dict[int, list[dict]] = {}
    for ch in chars:
        key = round(ch["y0"])
        buckets.setdefault(key, []).append(ch)
    # sort each line left-to-right, then return lines top-to-bottom
    return [
        sorted(line_chars, key=lambda c: c["x0"])
        for _, line_chars in sorted(buckets.items(), reverse=True)
    ]
 
 
def _line_font_info(line_chars: list[dict]) -> tuple[float, bool]:
    """Return (dominant_font_size, is_bold) for a list of pdfplumber char dicts."""
    sizes = [c["size"] for c in line_chars if c.get("size")]
    if not sizes:
        return 0.0, False
    bold_count = sum(
        1 for c in line_chars
        if any(w in (c.get("fontname") or "").lower() for w in ("bold", "heavy", "black", ",b"))
    )
    return statistics.median(sizes), bold_count > len(line_chars) * 0.5
 
 
def _detect_headers(pdf_path: str, min_fuzzy_score: float = 70.0) -> list[tuple[int, str, str]]:
    """
    Use PDFPlumber to find section headers.
 
    Returns a list of (page_number, raw_text, canonical_name), sorted by
    page order. Duplicate canonical names are deduplicated (first wins).
    """
    all_sizes: list[float] = []
    page_lines: list[tuple[int, list[dict]]] = []  # (page_no, line_chars)
 
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages):
            chars = [c for c in (page.chars or []) if c.get("text", "").strip()]
            for line_chars in _chars_to_lines(chars):
                size, _ = _line_font_info(line_chars)
                if size > 0:
                    all_sizes.append(size)
                page_lines.append((page_no, line_chars))
 
    if not all_sizes:
        return []
 
    body_size = statistics.median(all_sizes)
 
    headers: list[tuple[int, str, str]] = []
    seen_canonical: set[str] = set()
 
    for page_no, line_chars in page_lines:
        raw_text = "".join(c["text"] for c in line_chars).strip()
        if not raw_text or len(raw_text) < 3 or len(raw_text.split()) > 10:
            continue
 
        size, is_bold = _line_font_info(line_chars)
        if size < body_size * 1.1 and not is_bold:
            continue
 
        norm = _normalize(raw_text)
        if not norm:
            continue
 
        result = process.extractOne(norm, _ALL_ALIASES, scorer=fuzz.token_sort_ratio)
        if result is None:
            continue
        best_alias, score, _ = result
        if score < min_fuzzy_score:
            continue
 
        canonical = _ALIAS_TO_CANONICAL[best_alias]
        if canonical in seen_canonical:
            continue
 
        seen_canonical.add(canonical)
        headers.append((page_no, raw_text, canonical))
 
    return headers
 
 
class PaperSectionExtractor:
    def __init__(self, min_fuzzy_score: float = 72.0):
        self.min_fuzzy_score = min_fuzzy_score
 
    def extract_sections(self, pdf_path: str) -> dict:
        # ── Step 1: detect headers via PDFMiner ──────────────────────────────
        headers = _detect_headers(pdf_path, self.min_fuzzy_score)
 
        # ── Step 2: extract clean full text via PyMuPDF ──────────────────────
        doc = fitz.open(pdf_path)
        pages_text: list[str] = [page.get_text("text") for page in doc]
        doc.close()
        full_text = "\n\n".join(pages_text)
 
        # ── Step 3: find each header's position in the full text and slice ───
        # We search for the raw header string inside the PyMuPDF text.
        # This sidesteps any coordinate-system mismatch between the two libs.
        boundaries: list[tuple[int, str]] = []  # (char_offset, canonical)
 
        for _page_no, raw_text, canonical in headers:
            pos = full_text.lower().find(raw_text.lower())
            if pos == -1:
                # Try the normalised form as a fallback
                norm = _normalize(raw_text)
                pos = full_text.lower().find(norm.lower())
            if pos == -1:
                continue  # couldn't anchor this header; skip
            boundaries.append((pos, canonical))
 
        # Sort by position and deduplicate (string search may re-find earlier hit)
        boundaries.sort(key=lambda x: x[0])
        seen: set[str] = set()
        deduped: list[tuple[int, str]] = []
        for pos, canonical in boundaries:
            if canonical not in seen:
                seen.add(canonical)
                deduped.append((pos, canonical))
        boundaries = deduped
 
        # ── Step 4: slice full_text into sections ────────────────────────────
        sections: dict[str, str] = {}
 
        if not boundaries:
            sections["undefined"] = full_text
            return sections
 
        # Text before the first header
        sections["undefined"] = full_text[: boundaries[0][0]].strip()
 
        for i, (pos, canonical) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            sections[canonical] = full_text[pos:end].strip()
 
        return sections
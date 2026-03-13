"""
paper_section_extractor.py
──────────────────────────
Extracts named sections (abstract, introduction, methods, results,
discussion, conclusion, references, metadata) from research-paper PDFs.

Strategy
────────
1. Use PyMuPDF's span-level font metadata to detect candidate section
   headers (larger font / bold / all-caps on a short line).
2. Fuzzy-match each candidate against a vocabulary of known section names.
3. Merge / deduplicate the matched headers, then slice the raw text into
   sections.
4. (Optional) fall back to an Anthropic LLM call for papers whose headers
   are detected but not confidently classified.

Dependencies
────────────
    pip install pymupdf rapidfuzz anthropic
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from rapidfuzz import fuzz, process
from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SECTION VOCABULARY
#     Each canonical key maps to a list of aliases (lower-case).
#     Add more as you encounter them in your corpus.
# ─────────────────────────────────────────────────────────────────────────────

SECTION_ALIASES: dict[str, list[str]] = {
    "metadata":     [],          # filled in programmatically – text before abstract
    "abstract":     ["abstract", "summary", "executive summary", "synopsis",
                     "preface"],
    "keywords":     ["keywords", "key words", "index terms"],
    "introduction": ["introduction", "background", "motivation",
                     "background and motivation", "background & motivation",
                     "overview"],
    "related_work": ["related work", "related works", "literature review",
                     "prior work", "previous work"],
    "methods":      ["methods", "method", "methodology", "materials and methods",
                     "materials & methods", "proposed method", "proposed approach", 
                     "approach", "model", "framework", "system", "architecture",
                     "technical approach"],
    "results":      ["results", "experimental results",
                     "evaluation", "findings", "empirical results"],
    "discussion":   ["discussion", "analysis", "discussion and analysis",
                     "discussion and conclusion", "discussion & conclusion"],
    "conclusion":   ["conclusion", "conclusions", "concluding remarks",
                     "summary and conclusion", "future work",
                     "conclusion and future work"],
    "acknowledgements": ["acknowledgements", "acknowledgments", "acknowledgement",
                         "acknowledgment"],
    "references":   ["references", "bibliography", "notes and references",
                     "works cited", "citations", "literature",
                     "reference list"],
    "appendix":     ["appendix", "appendices", "supplementary material",
                     "supplementary", "supplemental material",
                     "additional material", "online supplement"],
}

# Flattened lookup: alias → canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in SECTION_ALIASES.items():
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a] = _canon

# Ordered list of all aliases (for rapidfuzz queries)
_ALL_ALIASES: list[str] = list(_ALIAS_TO_CANONICAL.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateHeader:
    """A text span that looks like a section header."""
    text: str                    # raw text of the header
    normalized: str              # lower-case, stripped, no numbering
    char_offset: int             # character offset in the concatenated full text
    page: int
    font_size: float
    is_bold: bool
    canonical: Optional[str] = None   # assigned after fuzzy matching
    score: float = 0.0                # match confidence 0-100

@dataclass
class PaperSections:
    """Holds the extracted sections of one paper."""
    path: str
    metadata: str = ""
    abstract: str = ""
    keywords: str = ""
    introduction: str = ""
    related_work: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    acknowledgements: str = ""
    references: str = ""
    appendix: str = ""
    unclassified: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "unclassified"}
        d["unclassified"] = self.unclassified
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PDF → FULL TEXT  +  CANDIDATE HEADER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lower-case, strip leading numbers/bullets/dots, collapse whitespace."""
    t = unicodedata.normalize("NFKC", text).lower().strip()
    # Remove leading section numbers like "1.", "1.2", "A.", "I."
    t = re.sub(r"^[\divxIVXA-Za-z]{1,4}[.\s]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_text_and_headers(pdf_path: str, 
                              font_size_ratio: float = 1.15,
                              min_header_score: float = 72.0
                              ) -> tuple[str, list[CandidateHeader]]:
    """
    Open *pdf_path* with PyMuPDF and return:
      - full_text : the entire paper as a single string (pages joined by \\n\\n)
      - candidates: list of CandidateHeader objects with char_offset into full_text

    Parameters
    ──────────
    font_size_ratio
        A text block is considered a *potential* header if its font size is at
        least this multiple of the median body-text font size on the page.
        1.15 means "at least 15 % larger than body text".
    min_header_score
        Fuzzy-match score threshold (0-100) below which a candidate is kept as
        unclassified rather than discarded or mis-labelled.
    """
    doc = fitz.open(pdf_path)

    full_pages: list[str] = []
    candidates: list[CandidateHeader] = []
    running_offset = 0  # character offset into the full concatenated text

    for page_no, page in enumerate(doc):
        # ── collect all spans on this page with their font metrics ──────────
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        span_data: list[dict] = []
        page_text_chars: list[str] = []  # we rebuild page text from spans
        page_char_offsets: list[int] = []  # offset of each span's first char

        for block in blocks:
            if block["type"] != 0:          # 0 = text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    txt = span["text"]
                    if not txt.strip():
                        continue
                    span_data.append({
                        "text": txt,
                        "size": span["size"],
                        "flags": span["flags"],   # bold = flags & 2^4
                        "offset": len("".join(page_text_chars)),
                    })
                    page_text_chars.append(txt)
                page_text_chars.append("\n")

        page_text = "".join(page_text_chars)
        full_pages.append(page_text)

        # ── compute median font size for this page (≈ body text) ────────────
        sizes = [s["size"] for s in span_data if len(s["text"].strip()) > 3]
        if not sizes:
            running_offset += len(page_text) + 2  # +2 for the \n\n separator
            continue
        sizes.sort()
        median_size = sizes[len(sizes) // 2]

        # ── find candidate headers ───────────────────────────────────────────
        # Group consecutive spans that share the same line (same y-position);
        # a header is typically a single short line with larger / bolder text.
        for span in span_data:
            txt = span["text"].strip()
            if not txt or len(txt) > 120:          # too long to be a heading
                continue
            if len(txt.split()) > 12:              # too many words
                continue

            is_big   = span["size"] >= median_size * font_size_ratio
            is_bold  = bool(span["flags"] & 16)    # bit 4 = bold in PDF flags
            is_caps  = txt.isupper() and len(txt) > 2

            if not (is_big or is_bold or is_caps):
                continue

            norm = _normalize(txt)
            if not norm:
                continue

            # fuzzy match against vocabulary
            match = process.extractOne(norm, _ALL_ALIASES,
                                       scorer=fuzz.token_sort_ratio)
            if match is None:
                continue

            best_alias, score, _ = match
            canonical = _ALIAS_TO_CANONICAL[best_alias] if score >= min_header_score else None

            char_offset = running_offset + span["offset"]
            candidates.append(CandidateHeader(
                text=txt,
                normalized=norm,
                char_offset=char_offset,
                page=page_no + 1,
                font_size=span["size"],
                is_bold=is_bold,
                canonical=canonical,
                score=score,
            ))

        running_offset += len(page_text) + 2  # for the \n\n between pages

    doc.close()
    full_text = "\n\n".join(full_pages)
    return full_text, candidates


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DEDUPLICATE & SELECT BEST HEADER PER SECTION
# ─────────────────────────────────────────────────────────────────────────────

def _select_headers(candidates: list[CandidateHeader]) -> list[CandidateHeader]:
    """
    Given raw candidates (possibly multiple per section), keep the *first*
    occurrence of each canonical section, choosing the highest-confidence hit
    when multiple candidates point to the same section.

    Also preserves unclassified candidates (canonical=None) for LLM fallback.
    """
    # Group by canonical name
    by_canon: dict[str, list[CandidateHeader]] = {}
    for c in candidates:
        key = c.canonical or f"__unknown_{c.char_offset}"
        by_canon.setdefault(key, []).append(c)

    selected: list[CandidateHeader] = []
    for key, group in by_canon.items():
        if key.startswith("__unknown"):
            # keep all unknowns for possible LLM classification
            selected.extend(group)
        else:
            # keep earliest occurrence (lowest char_offset) among high-confidence
            best = min(group, key=lambda c: (c.char_offset, -c.score))
            selected.append(best)

    selected.sort(key=lambda c: c.char_offset)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OPTIONAL LLM FALLBACK  
# ─────────────────────────────────────────────────────────────────────────────

def _classify_with_llm(
    unknown_headers: list[CandidateHeader],
    llm_host: str = "http://localhost:11434",
    llm_model: str = "llama3.2",
) -> list[CandidateHeader]:
    """
    Ask a local LLM (via Ollama's OpenAI-compatible /v1/chat/completions
    endpoint) to classify unrecognised section headers.

    Returns the same list with .canonical filled in where possible.

    Parameters
    ──────────
    unknown_headers : headers whose .canonical is still None
    llm_host        : base URL of the Ollama server
                      (e.g. "http://localhost:11434" or "http://ollama:11434")
    llm_model       : model tag to use (e.g. "llama3", "mistral", "phi3")
    """
    import urllib.request

    candidates_json = json.dumps(
        [{"text": h.text, "normalized": h.normalized} for h in unknown_headers],
        indent=2,
    )
    canonical_list = list(SECTION_ALIASES.keys())

    prompt = (
        f"You are classifying section headers from a research paper.\n\n"
        f"Canonical section names: {canonical_list}\n\n"
        f"For each header below, output a JSON array with the same order, "
        f"where each element is either the best matching canonical name "
        f"(a string from the list above) or null if it does not correspond "
        f"to any standard section.\n"
        f"Return ONLY the JSON array — no explanation, no markdown fences.\n\n"
        f"Headers to classify:\n{candidates_json}"
    )

    payload = json.dumps({
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()

    url = llm_host.rstrip("/") + "/v1/chat/completions"

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[LLM fallback] Request to {url} failed: {exc}")
        return unknown_headers

    try:
        raw = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        print(f"[LLM fallback] Unexpected response shape: {exc}\n{body}")
        return unknown_headers

    # Strip markdown fences if the model added them anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")

    try:
        labels: list[Optional[str]] = json.loads(raw)
        for header, label in zip(unknown_headers, labels):
            if label and label in SECTION_ALIASES:
                header.canonical = label
                header.score = 90.0  # LLM-assigned confidence
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"[LLM fallback] Could not parse model response: {exc}\nRaw: {raw}")

    return unknown_headers


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SLICE FULL TEXT INTO SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _slice_text(full_text: str,
                headers: list[CandidateHeader]) -> dict[str, str]:
    """
    Given the full paper text and an ordered list of detected headers,
    slice the text into named sections.

    Text *before* the first recognised header is assigned to 'metadata'.
    """
    sections: dict[str, str] = {}

    # Only keep headers that have been classified
    classified = [h for h in headers if h.canonical and not h.canonical.startswith("__")]
    classified.sort(key=lambda h: h.char_offset)

    if not classified:
        # No headers found – return everything as metadata
        return {"metadata": full_text}

    # Text before the first header → metadata
    first = classified[0]
    sections["metadata"] = full_text[: first.char_offset].strip()

    for i, header in enumerate(classified):
        start = header.char_offset
        end   = classified[i + 1].char_offset if i + 1 < len(classified) else len(full_text)
        chunk = full_text[start:end].strip()
        canon = header.canonical

        # If the same canonical section appears twice (e.g. two "methods" subsections),
        # append rather than overwrite
        if canon in sections:
            sections[canon] = sections[canon] + "\n\n" + chunk
        else:
            sections[canon] = chunk

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TOP-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

@tool
def extract_sections(pdf_path: str,
                     use_llm_fallback: bool = False,
                     llm_host: str = "http://localhost:11434",
                     llm_model: str = "llama3",
                     font_size_ratio: float = 1.15,
                     min_header_score: float = 72.0,
                     verbose: bool = False) -> PaperSections:
    """
    Main entry point.

    Parameters
    ──────────
    pdf_path          : path to the PDF file
    use_llm_fallback  : send unclassified headers to an Ollama LLM for
                        classification when fuzzy matching isn't confident enough
    llm_host          : Ollama server base URL
                        (default "http://localhost:11434"; use "http://ollama:11434"
                        inside Docker)
    llm_model         : Ollama model tag to use (e.g. "llama3", "mistral", "phi3")
    font_size_ratio   : see extract_text_and_headers
    min_header_score  : fuzzy-match threshold (0-100)
    verbose           : print detected headers to stdout

    Returns
    ───────
    PaperSections dataclass with one attribute per section.
    """
    full_text, raw_candidates = extract_text_and_headers(
        pdf_path, font_size_ratio, min_header_score
    )

    selected = _select_headers(raw_candidates)

    if verbose:
        print(f"\n── Detected headers in '{Path(pdf_path).name}' ──")
        for h in selected:
            status = h.canonical or "UNCLASSIFIED"
            print(f"  p{h.page:>3}  [{h.score:5.1f}]  {status:<20}  '{h.text}'")

    if use_llm_fallback:
        unknowns = [h for h in selected if h.canonical is None]
        if unknowns:
            _classify_with_llm(unknowns, llm_host=llm_host, llm_model=llm_model)

    slices = _slice_text(full_text, selected)

    ps = PaperSections(path=pdf_path)
    canonical_fields = set(PaperSections.__dataclass_fields__) - {"path", "unclassified"}

    for canon, text in slices.items():
        if canon in canonical_fields:
            setattr(ps, canon, text)
        else:
            ps.unclassified[canon] = text

    return ps


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI  (python paper_section_extractor.py paper.pdf)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Extract sections from a research PDF.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM fallback for unclassified headers")
    parser.add_argument("--llm-host", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--llm-model", default="llama3",
                        help="Ollama model tag (default: llama3)")
    parser.add_argument("--ratio", type=float, default=1.15,
                        help="Font size ratio to flag headers (default 1.15)")
    parser.add_argument("--score", type=float, default=72.0,
                        help="Fuzzy match threshold 0-100 (default 72)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of pretty-print")
    args = parser.parse_args()

    result = extract_sections(
        args.pdf,
        use_llm_fallback=args.llm,
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        font_size_ratio=args.ratio,
        min_header_score=args.score,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        d = result.to_dict()
        for section, text in d.items():
            if section == "unclassified":
                continue
            if text:
                preview = text[:200].replace("\n", " ")
                print(f"\n{'═'*60}")
                print(f"  {section.upper()}")
                print(f"{'═'*60}")
                print(preview + ("…" if len(text) > 200 else ""))
        if result.unclassified:
            print(f"\n── Unclassified sections ──")
            for k, v in result.unclassified.items():
                print(f"  {k}: {v[:100]}…")

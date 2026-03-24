"""
Preprocessing adapter for the knowledge-extraction pipeline.

Converts a PDF → pymupdf4llm Markdown → LlamaIndex Document
and partitions the document into canonical sections via PaperSectionExtractor.
"""

import json
import logging
from pathlib import Path
from typing import Optional
import re

import pymupdf
import pymupdf4llm
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from .paper_section_extractor import PaperSectionExtractor

logger = logging.getLogger(__name__)


class PaperPreprocessor:
    """
    Ingests a PDF and returns:
    - A LlamaIndex ``Document`` (the full markdown text + file metadata)
    - A ``sections`` dict mapping canonical section names → section text
    - The path to the saved raw markdown file (used for anchor resolution)
    """

    def __init__(self, llm: LLM, embed_model: Optional[BaseEmbedding] = None) -> None:
        self.llm = llm
        self.section_extractor = PaperSectionExtractor(llm, embed_model=embed_model)
        logger.info("Initialized PaperPreprocessor (pymupdf4llm + LlamaIndex)")

    # ── Core ingestion ────────────────────────────────────────────────────────

    def _pdf_to_markdown(self, pdf_path: Path) -> str:
        """Convert a PDF to Markdown using pymupdf4llm."""
        doc = pymupdf.Document(str(pdf_path))
        return pymupdf4llm.to_markdown(doc, footer=False, header=False)

    def _strip_markdown_formatting(self, text: str) -> str:
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

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess_pdf(self, pdf_path: str, output_dir: str | None = None) -> dict:
        """
        Preprocess a single PDF.

        Parameters
        ----------
        pdf_path:   Path to the PDF file.
        output_dir: Directory for saved outputs (default: ``<pdf_dir>/preprocessed``).

        Returns
        -------
        dict with keys:
            ``pdf_path``       – original PDF path (str)
            ``raw_text_path``  – path to the saved Markdown file (str)
            ``document``       – LlamaIndex Document (full markdown + metadata)
            ``sections``       – dict[str, str] of canonical sections
        On error, returns ``{"error": <message>}``.
        """
        pdf_path = Path(pdf_path)

        if output_dir is None:
            output_dir = pdf_path.parent / "preprocessed"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = pdf_path.stem
        logger.info("Preprocessing PDF: %s", pdf_path.name)

        # ── Step 1: PDF → Markdown ────────────────────────────────────────────
        logger.info("  → Converting PDF to Markdown via pymupdf4llm …")
        try:
            md_text = self._pdf_to_markdown(pdf_path)
        except Exception as exc:
            logger.error("  ✗ PDF conversion failed: %s", exc)
            return {"error": str(exc)}

        # ── Step 2: Wrap in a LlamaIndex Document ────────────────────────────
        li_document = Document(
            text=md_text,
            metadata={"file_name": pdf_path.name, "file_path": str(pdf_path)},
        )

        # ── Step 3: Save raw Markdown (for identifying headers) ──────────
        raw_md_path = output_dir / f"{base_name}_raw.md"
        raw_md_path.write_text(md_text, encoding="utf-8")
        logger.info("  ✓ Saved raw Markdown: %s", raw_md_path)

        # ── Step 3.5: Save clean text (no formatting) ───────────────────
        stripped_text = self._strip_markdown_formatting(md_text)  
        stripped_path = output_dir / f"{base_name}_plain.txt"
        stripped_path.write_text(stripped_text, encoding="utf-8")
        logger.info("  ✓ Saved clean text: %s", stripped_path)

        # ── Step 4: Extract canonical sections ───────────────────────────────
        # Heading detection runs on raw md_text; body stripping happens inside extract_sections_from_markdown after slicing.
        logger.info("  → Extracting sections …")
        try:
            sections = self.section_extractor.extract_sections_from_markdown(md_text)
        except Exception as exc:
            logger.error("  ✗ Section extraction failed: %s", exc)
            sections = {}

        # Optionally persist sections for debugging
        sections_path = output_dir / f"{base_name}_sections.json"
        sections_path.write_text(
            json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("  ✓ Saved sections: %s", sections_path)

        return {
            "pdf_path": str(pdf_path),
            "raw_text_path": str(stripped_path),
            "document": li_document,
            "sections": sections,
        }

    def batch_preprocess(self, input_dir: str, output_dir: str | None = None) -> list[dict]:
        """
        Preprocess all PDFs in *input_dir*.

        Returns a list of result dicts (same shape as ``preprocess_pdf``).
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error("Input directory not found: %s", input_dir)
            return []

        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", input_dir)
            return []

        logger.info("Found %d PDF files to preprocess", len(pdf_files))

        results: list[dict] = []
        for i, pdf in enumerate(pdf_files, 1):
            logger.info("\n[%d/%d] Processing: %s", i, len(pdf_files), pdf.name)
            try:
                result = self.preprocess_pdf(str(pdf), output_dir)
            except Exception as exc:
                logger.error("  ✗ Error: %s", exc, exc_info=True)
                result = {"pdf_path": str(pdf), "error": str(exc)}
            results.append(result)

        logger.info("\n✓ Preprocessed %d files", len(results))
        return results

"""
Preprocessing adapter for Neo4j pipeline
Extracts text from PDF and prepares it for processing
"""
import os
import sys
import logging
from pathlib import Path

from .document_converter import DocumentConverter
from .paper_section_extractor import extract_sections, PaperSections
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Sections that constitute the "body" of a paper (passed to LLM extractors).
# References, metadata, and acknowledgements are excluded by default.
BODY_SECTIONS = [
    "abstract",
    "keywords",
    "introduction",
    "related_work",
    "methods",
    "results",
    "discussion",
    "conclusion",
]

class BuildingPreprocessor:
    """
    Preprocesses PDFs: extracts text and splits into named sections
    (abstract, introduction, methods, results, discussion, conclusion,
    references, metadata) using font-metadata detection + fuzzy matching.
    """
    def __init__(self, ollama_host: str = None, ollama_model: str = None):
        """
        Args:
            ollama_host:  Ollama base URL (e.g. "http://ollama:11434").
                          Used for the LLM fallback in section classification.
            ollama_model: Ollama model tag (e.g. "llama3").
        """
        self.converter  = DocumentConverter()
        self.llm_host   = ollama_host  or "http://localhost:11434"
        self.llm_model  = ollama_model or "llama3"
        logger.info("Initialized BuildingPreprocessor (section-aware PDF extractor)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_pdf(self, pdf_path: str, output_dir: str = None) -> dict:
        """
        Preprocess a single PDF: extract text, split into sections, save files.

        Args:
            pdf_path:   Path to the PDF file.
            output_dir: Directory to write outputs (default: <pdf_dir>/preprocessed).

        Returns:
            dict with keys:
              - pdf_path        : original PDF path (str)
              - raw_text_path   : path to the saved full body text  (str)
              - sections        : PaperSections dataclass
              - section_paths   : dict mapping section name → saved .txt path
        """
        pdf_path = Path(pdf_path)

        if output_dir is None:
            output_dir = pdf_path.parent / "preprocessed"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = pdf_path.stem
        logger.info(f"Preprocessing PDF: {pdf_path.name}")

        # ── 1. Section-aware extraction ──────────────────────────────────────
        logger.info("  → Extracting and classifying sections...")
        try:
            # extract_sections is now a @tool, so we must use .invoke()
            sections_data = extract_sections.invoke({
                "pdf_path": str(pdf_path),
                "use_llm_fallback": False,
                "llm_host": self.llm_host,
                "llm_model": self.llm_model,
                "verbose": False,
            })
            sections = sections_data  # PaperSections dataclass
        except Exception as e:
            logger.error(f"  ✗ Section extraction failed: {e}")
            return {"error": str(e)}

        self._log_section_summary(sections)

        # ── 2. Build "body text" (all content except metadata / references) ──
        body_parts = []
        for name in BODY_SECTIONS:
            text = getattr(sections, name, "")
            if text:
                body_parts.append(text)
        # Also include any unclassified sections so nothing is silently dropped
        body_parts.extend(sections.unclassified.values())

        body_text = "\n\n".join(body_parts).strip()

        if not body_text:
            # Edge case: section detection found nothing — fall back to full text
            logger.warning("  ⚠ No body sections detected; falling back to full-document text.")
            try:
                body_text = self.converter.convert_to_text(str(pdf_path))
                body_text = self.converter.preprocess_text(body_text)
            except Exception as e:
                logger.error(f"  ✗ Fallback text extraction failed: {e}")
                return {"error": str(e)}

        # ── 3. Save full body text (backwards-compatible raw_text_path) ──────
        raw_text_path = output_dir / f"{base_name}_raw.txt"
        raw_text_path.write_text(body_text, encoding="utf-8")
        logger.info(f"  ✓ Body text saved: {raw_text_path}")

        # ── 4. Save individual section files ─────────────────────────────────
        section_paths: dict[str, str] = {}
        all_section_names = list(PaperSections.__dataclass_fields__.keys())
        all_section_names = [s for s in all_section_names if s not in ("path", "unclassified")]

        for name in all_section_names:
            text = getattr(sections, name, "")
            if text:
                section_file = output_dir / f"{base_name}_{name}.txt"
                section_file.write_text(text, encoding="utf-8")
                section_paths[name] = str(section_file)

        for name, text in sections.unclassified.items():
            if text:
                section_file = output_dir / f"{base_name}_unclassified_{name}.txt"
                section_file.write_text(text, encoding="utf-8")
                section_paths[f"unclassified_{name}"] = str(section_file)

        return {
            "pdf_path":      str(pdf_path),
            "raw_text_path": str(raw_text_path),   # body text (no refs/metadata)
            "sections":      sections,              # PaperSections dataclass
            "section_paths": section_paths,         # name → file path
        }

    def batch_preprocess(self, input_dir: str, output_dir: str = None) -> list:
        """
        Preprocess all PDFs in a directory.

        Args:
            input_dir:  Directory containing PDFs.
            output_dir: Directory to save outputs.

        Returns:
            List of result dicts (one per PDF).
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return []

        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to preprocess")

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            try:
                result = self.preprocess_pdf(str(pdf_path), output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"  ✗ Error: {e}", exc_info=True)
                results.append({"pdf_path": str(pdf_path), "error": str(e)})

        logger.info(f"\n✓ Preprocessed {len(results)} files")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_section_summary(sections: PaperSections) -> None:
        found, missing = [], []
        for name in BODY_SECTIONS:
            (found if getattr(sections, name, "") else missing).append(name)
        if sections.references:
            found.append("references")
        if sections.metadata:
            found.append("metadata")
        logger.info(f"  ✓ Sections found:   {found}")
        if missing:
            logger.info(f"  ⚠ Sections missing: {missing}")
        if sections.unclassified:
            logger.info(f"  ? Unclassified:     {list(sections.unclassified.keys())}")


@tool
def preprocess_building_document(pdf_path: str, output_dir: str = None) -> dict:
    """
    Preprocess a PDF document to extract text and prepare it for building information extraction.
    Returns a dictionary with paths to the generated files.
    """
    preprocessor = BuildingPreprocessor()
    return preprocessor.preprocess_pdf(pdf_path, output_dir)

# ──────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess PDFs: extract and classify sections"
    )
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    preprocessor = BuildingPreprocessor()
    input_path = Path(args.input)

    if input_path.is_file():
        result = preprocessor.preprocess_pdf(str(input_path), args.output)
        if "error" not in result:
            print(f"\n✓ Done!  Body text → {result['raw_text_path']}")
            print(f"  Sections saved: {list(result['section_paths'].keys())}")

    elif input_path.is_dir():
        results = preprocessor.batch_preprocess(str(input_path), args.output)
        print(f"\n✓ Batch complete — {len(results)} files processed")

    else:
        print(f"Error: {args.input} is not a file or directory")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
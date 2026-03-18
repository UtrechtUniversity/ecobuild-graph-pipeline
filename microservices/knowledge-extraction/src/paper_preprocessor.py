"""
Preprocessing adapter for Neo4j pipeline
Extracts text from PDF and prepares it for processing
"""

import os
import sys
import logging
from pathlib import Path
import re
import fitz
import statistics
import json
from llama_index.core.llms import LLM

from .document_converter import DocumentConverter
# from .paper_section_extractor import PaperSectionExtractor

logger = logging.getLogger(__name__)

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class PaperPreprocessor:
    """
    Preprocesses PDFs to extract text
    """
    def _remove_metadata(self, text: str) -> str:
        """
        Remove initial metadata section from processed text
        """
        metadata = re.search(r"\n(Key Words|Keywords|Abstract|A B S T R A C T)", text, flags=re.IGNORECASE)
        if metadata:
            return text[metadata.start():]
        return text

    def _remove_references(self, text: str) -> str:
        """
        Remove references section from processed text
        """
        ref_start = re.search(r"\n(References|Bibliography)\n", text, flags=re.IGNORECASE)
        if ref_start:
            return text[:ref_start.start()]
        return text
        
    
    def __init__(self, llm: LLM):
        """
        Initialize preprocessor
        
        Args:
            llm: LlamaIndex LLM instance
        """
        self.llm = llm
        self.converter = DocumentConverter()
        logger.info("Initialized PaperPreprocessor (PDF-to-text only)")
        # self.section_extractor = PaperSectionExtractor(
        #     model=ollama_model, 
        #     base_url=ollama_host
        # )
        # logger.info("Initialized section extractor")
    
    def preprocess_pdf(self, pdf_path: str, output_dir: str = None) -> dict:
        """
        Preprocess a PDF: extract text and save it
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs (default: same as PDF)
        
        Returns:
            Dict with paths to generated files
        """
        pdf_path = Path(pdf_path)
        
        if output_dir is None:
            output_dir = pdf_path.parent / "preprocessed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = pdf_path.stem
        
        logger.info(f"Preprocessing PDF: {pdf_path.name}")
              
        # Convert PDF to text
        logger.info("  → Converting PDF to text...")
        try:
            # raw_text = self.converter.convert_to_text(str(pdf_path))
            raw_text = self.converter.pdf_to_markdown(str(pdf_path))
            # raw_text = self.converter.preprocess_text(raw_md_text)
            # raw_text = self._remove_references(raw_text)
            # logger.info("  → WARNING: References removed")
            # raw_text = self._remove_metadata(raw_text)
            # logger.info("  → WARNING: Metadata removed")
            # raw_text = self._extract_abstract(raw_text)
        except Exception as e:
            logger.error(f"  ✗ Failed to convert PDF: {e}")
            return {"error": str(e)}
        
        # Save raw text
        raw_text_path = output_dir / f"{base_name}_raw.md"
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        logger.info(f"  ✓ Saved raw text: {raw_text_path}")

        # # Extract sections
        # logger.info("  → Extracting sections...")
        # try:
        #     sections = self.section_extractor.extract_sections(raw_text)
        # except Exception as e:
        #     logger.error(f"  ✗ Failed to extract sections: {e}")
        #     return {"error": str(e)}

        # # Save sections
        # sections_path = output_dir / f"{base_name}_sections.json"
        # with open(sections_path, 'w', encoding='utf-8') as f:
        #     json.dump(sections, f, indent=2)
        # logger.info(f"  ✓ Saved sections: {sections_path}")
        
        return {
            "pdf_path": str(pdf_path),
            "raw_text_path": str(raw_text_path),
            # "sections_path": str(sections_path)
        }
    
    def batch_preprocess(self, input_dir: str, output_dir: str = None) -> list:
        """
        Preprocess all PDFs in a directory
        
        Args:
            input_dir: Directory containing PDFs
            output_dir: Directory to save outputs
        
        Returns:
            List of result dictionaries
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return []
        
        # Find all PDFs
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


def main():
    """Standalone preprocessing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess PDFs: Extract text only")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize preprocessor (no params needed now)
    preprocessor = PaperPreprocessor()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        result = preprocessor.preprocess_pdf(str(input_path), args.output)
        if "error" not in result:
            print(f"\n✓ Preprocessing complete!")
            print(f"  Text file: {result.get('raw_text_path')}")
    
    elif input_path.is_dir():
        # Process directory
        results = preprocessor.batch_preprocess(str(input_path), args.output)
        print(f"\n✓ Batch preprocessing complete! Processed {len(results)} files")
    
    else:
        print(f"Error: {args.input} is not a file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

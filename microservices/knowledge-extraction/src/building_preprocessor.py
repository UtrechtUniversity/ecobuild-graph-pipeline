"""
Preprocessing adapter for Neo4j pipeline
Extracts text from PDF and prepares it for processing
"""

import os
import sys
import logging
from pathlib import Path

# Import only the converter
from .document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class BuildingPreprocessor:
    """
    Preprocesses PDFs to extract text
    """
    
    def __init__(self, ollama_host: str = None, ollama_model: str = None):
        """
        Initialize preprocessor
        
        Args:
            ollama_host: Unused (kept for compatibility)
            ollama_model: Unused (kept for compatibility)
        """
        self.converter = DocumentConverter()
        logger.info("Initialized BuildingPreprocessor (PDF-to-text only)")
    
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
            raw_text = self.converter.convert_to_text(str(pdf_path))
            raw_text = self.converter.preprocess_text(raw_text)
        except Exception as e:
            logger.error(f"  ✗ Failed to convert PDF: {e}")
            return {"error": str(e)}
        
        # Save raw text
        raw_text_path = output_dir / f"{base_name}_raw.txt"
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        logger.info(f"  ✓ Saved raw text: {raw_text_path}")
        
        return {
            "pdf_path": str(pdf_path),
            "raw_text_path": str(raw_text_path)
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
    preprocessor = BuildingPreprocessor()
    
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

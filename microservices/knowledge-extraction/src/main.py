"""
Modified Neo4j pipeline with building information preprocessing
Now extracts both entities AND design strategies
"""

from requests.models import HTTPError
from io import BytesIO
import requests
from time import sleep
from neo4j import GraphDatabase
import os
import asyncio
import logging
import json

# Import the preprocessor and extractors
from .building_preprocessor import BuildingPreprocessor
from .entity_extractor import EntityInformationExtractor
from .design_strategy_extractor import DesignStrategyExtractor

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def download_paper_pdf(url: str) -> BytesIO | None:
    """Downloads the pdf of the paper from the provided url"""
    logger.info(f"Attempting to download pdf from {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as e:
        logger.info(f"HTTPError for url: {url} - Error: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.info(f"ConnectionError for url: {url} - Error: {e}")
        return None
    except Exception as e:
        logger.info(f"Unexpected error for url: {url} - Error: {e}")
        return None
    else:
        content = response.content
        if content[:5] == b'%PDF-':
            pdf_bytes = BytesIO(response.content)
            return pdf_bytes
        else:
            logger.info(f"URL {url} does not contain a pdf.")
            return None


async def main():
    """
    Current extraction pipeline:
    1. Convert PDFs to text
    2. Extract entities (buildings)
    3. Extract design strategies
    4. Combine results into single JSON
    """
    
    # Initialize the building preprocessor
    logger.info("Initializing building information preprocessor...")
    preprocessor = BuildingPreprocessor(
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_LLM_MODEL
    )
    
    # Initialize extractors
    logger.info("Initializing entity extractor...")
    entity_extractor = EntityInformationExtractor(
        model=OLLAMA_LLM_MODEL, 
        base_url=OLLAMA_HOST
    )
    
    logger.info("Initializing design strategy extractor...")
    design_extractor = DesignStrategyExtractor(
        model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_HOST
    )
    
    # Define paths
    test_papers_dir_path = "/app/test_papers/test_papers"
    preprocessed_output_dir = "/app/test_papers/preprocessed"
    
    logger.info(f"Looking for test papers in: {test_papers_dir_path}")
    
    # Find PDF files
    pdf_files_to_process = []
    if os.path.exists(test_papers_dir_path) and os.path.isdir(test_papers_dir_path):
        for filename in os.listdir(test_papers_dir_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(test_papers_dir_path, filename)
                pdf_files_to_process.append(full_path)
        
        if not pdf_files_to_process:
            logger.warning(f"No PDF files found in {test_papers_dir_path}. Please add some test PDFs.")
    else:
        logger.error(f"Test papers directory not found: {test_papers_dir_path}")
        pdf_files_to_process = []
    
    if not pdf_files_to_process:
        logger.info("No PDFs to process. Exiting.")
        return
    
    # Process each PDF
    for i, pdf_path in enumerate(pdf_files_to_process):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing PDF {i+1}/{len(pdf_files_to_process)}: {os.path.basename(pdf_path)}")
        logger.info(f"{'='*80}")
        
        try:
            # ========================================
            # STEP 1: Convert PDF to Text
            # ========================================
            logger.info("\nSTEP 1: Converting PDF to text...")
            preprocess_result = preprocessor.preprocess_pdf(
                pdf_path=pdf_path,
                output_dir=preprocessed_output_dir
            )
            
            if "error" in preprocess_result:
                logger.error(f"  ✗ Preprocessing failed: {preprocess_result['error']}")
                continue
            
            logger.info(f"  ✓ Text extracted: {preprocess_result['raw_text_path']}")
            
            # Load the extracted text
            with open(preprocess_result['raw_text_path'], 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # ========================================
            # STEP 2: Extract Entities (Buildings)
            # ========================================
            logger.info("\nSTEP 2: Extracting entities (buildings)...")
            entity_results = entity_extractor.extract_from_text(raw_text, verbose=False)
            
            entities = entity_results.get('entities', [])
            
            # Log entity summary
            if entities:
                logger.info(f"  ✓ Found {len(entities)} entities")
                for ent in entities:
                    name = ent.get('name', {}).get('value') if isinstance(ent.get('name'), dict) else ent.get('name')
                    city = ent.get('city', {}).get('value') if isinstance(ent.get('city'), dict) else ent.get('city')
                    country = ent.get('country', {}).get('value') if isinstance(ent.get('country'), dict) else ent.get('country')
                    logger.info(f"    - {name or 'Unnamed'} ({city or 'Unknown'}, {country or 'Unknown'})")
            else:
                logger.info("  ✗ No entities identified")
            
            # ========================================
            # STEP 3: Extract Design Strategies
            # ========================================
            logger.info("\nSTEP 3: Extracting design strategies...")
            design_results = design_extractor.extract_from_text(raw_text, verbose=False)
            
            strategies = design_results.get('design_strategies', [])
            
            # Log strategy summary
            if strategies:
                logger.info(f"  ✓ Found {len(strategies)} design strategies")
                for strategy in strategies[:3]:  # Show first 3
                    name = strategy.get('name', 'Unnamed')
                    num_mentions = len(strategy.get('context', []))
                    logger.info(f"    - {name} ({num_mentions} mentions)")
            else:
                logger.info("  ✗ No design strategies identified")
            
            # ========================================
            # STEP 4: Combine Results into Single JSON
            # ========================================
            logger.info("\nSTEP 4: Combining results...")
            
            # Create combined results dictionary
            combined_results = {
                'entities': entity_results.get('entities', []),
                'design_strategies': design_results.get('design_strategies', [])
            }
            
            # Save combined results to single JSON file
            pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            json_path = os.path.join(preprocessed_output_dir, f"{pdf_base_name}_extraction.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  ✓ Combined results saved to: {json_path}")
            
            # ========================================
            # STEP 5: Generate Combined Report
            # ========================================
            logger.info("\nSTEP 5: Generating report...")
            
            # Generate combined report
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"EXTRACTION REPORT: {pdf_base_name}")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # Entities section
            report_lines.append("### ENTITIES (BUILDINGS) ###")
            if entities:
                for idx, ent in enumerate(entities, 1):
                    report_lines.append(f"\nEntity {idx}:")
                    for key, data in ent.items():
                        if isinstance(data, dict) and 'value' in data:
                            value = data.get('value')
                            context = data.get('context')
                        else:
                            value = data
                            context = None
                        
                        if value is not None:
                            report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
                            if context and context != 'Extracted via QA':
                                report_lines.append(f"    Context: \"{context}\"")
            else:
                report_lines.append("  No entities found.")
            
            report_lines.append("\n")
            
            # Design strategies section
            report_lines.append("### DESIGN STRATEGIES ###")
            if strategies:
                for idx, strategy in enumerate(strategies, 1):
                    name = strategy.get('name', 'Unnamed')
                    contexts = strategy.get('context', [])
                    
                    report_lines.append(f"\nStrategy {idx}: {name}")
                    report_lines.append(f"  Mentions: {len(contexts)}")
                    
                    for j, context in enumerate(contexts, 1):
                        report_lines.append(f"\n  Quote {j}:")
                        report_lines.append(f'    "{context}"')
            else:
                report_lines.append("  No design strategies found.")
            
            report_lines.append("\n" + "=" * 80)
            
            # Save combined report
            report_path = os.path.join(preprocessed_output_dir, f"{pdf_base_name}_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"  ✓ Report saved to: {report_path}")
            
            # ========================================
            # Display Summary
            # ========================================
            logger.info("\n--- EXTRACTION SUMMARY ---")
            logger.info(f"Entities: {len(entities)}")
            logger.info(f"Design Strategies: {len(strategies)}")
            logger.info(f"Output: {json_path}")
            
            # ========================================
            # STEP 6: (FUTURE) Neo4j Integration
            # ========================================
            """
            Uncomment when ready to integrate with Neo4j:
            
            logger.info("\nSTEP 6: Processing with Neo4j pipeline...")
            
            from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
            
            pipeline = PipelineRunner.from_config_file("src/test_config.json")
            
            results = await pipeline.run({
                "file_path": preprocess_result['raw_text_path'],
                "metadata": {
                    "paper_name": os.path.basename(pdf_path),
                    "entities": entities,
                    "design_strategies": strategies
                }
            })
            
            logger.info(f"  ✓ Neo4j results: {results}")
            """
            
            logger.info(f"\n✓ Completed: {os.path.basename(pdf_path)}")
            
        except Exception as e:
            logger.error(f"\n✗ Error processing '{os.path.basename(pdf_path)}': {e}", exc_info=True)
    
    # ========================================
    # Final Summary
    # ========================================
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Total PDFs processed: {len(pdf_files_to_process)}")
    logger.info(f"Output directory: {preprocessed_output_dir}")
    logger.info(f"\nFiles created per paper:")
    logger.info(f"  - *_raw.txt (plain text)")
    logger.info(f"  - *_extraction.json (entities + design strategies)")
    logger.info(f"  - *_report.txt (human-readable)")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
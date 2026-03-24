"""
Modified Neo4j pipeline with building information preprocessing
Now extracts entities, design strategies, AND ecosystem services
"""
 
from requests.models import HTTPError
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from io import BytesIO
import requests
from time import sleep
from neo4j import GraphDatabase
import os
import asyncio
import logging
import json
 
# Import the preprocessor and extractors
from .paper_preprocessor import PaperPreprocessor
from .entity_extractor import EntityInformationExtractor
from .design_strategy_extractor import DesignStrategyExtractor
from .ecosystem_service_extractor import EcosystemServiceExtractor
from .context_resolver import (
    resolve_entity_contexts,
    resolve_design_strategy_contexts,
    resolve_ecosystem_service_contexts,
)
from .entity_resolution import EntityResolutionMatcher
 
# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")

#LlamaIndex model definitions
llm = Ollama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_HOST, context_window=12000, temperature=0.11, request_timeout=180.0)
embed_model = OllamaEmbedding(model_name=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST, request_timeout=180.0) 
 
 
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
 
def _count_verified(items: list, verified_key: str = "anchor_verified") -> tuple[int, int]:
    """Return (verified_count, total_count) for a list of extracted items."""
    total = len(items)
    verified = sum(1 for item in items if item.get(verified_key) is True)
    return verified, total
 
async def main():
    """
    Current extraction pipeline:
    1. Convert PDFs to text
    2. Extract entities (buildings)
    3. Extract design strategies
    4. Extract ecosystem services
    5. Combine results into single JSON
    """
    
    # Initialize the building preprocessor
    logger.info("Initializing building information preprocessor...")
    preprocessor = PaperPreprocessor(llm, embed_model=embed_model)
    
    # Initialize extractors
    logger.info("Initializing entity extractor...")
    entity_extractor = EntityInformationExtractor(llm)
    
    logger.info("Initializing design strategy extractor...")
    design_extractor = DesignStrategyExtractor(llm)
    
    logger.info("Initializing ecosystem service extractor...")
    ecosystem_extractor = EcosystemServiceExtractor(llm)
 
    logger.info("Initializing entity resolution matcher (pre-embedding vocabularies)...")
    resolver = EntityResolutionMatcher(embed_model)
    
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
 
            # Load section-aware text
            # raw_text = full body (no metadata/references) — used as the
            #            universal source for anchor verification.
            # *_text   = targeted subsets passed to each extractor.
            with open(preprocess_result['raw_text_path'], 'r', encoding='utf-8') as f:
                raw_text = f.read()
 
            sections = preprocess_result.get("sections")
 
            def _join(*names) -> str:
                """Concatenate named sections; fall back to raw_text if all empty."""
                if not sections:
                    return raw_text
                parts = [sections.get(n, "") for n in names if sections.get(n)]
                return "\n\n".join(parts) if parts else raw_text
 
            # Text scopes fed to each extractor
            building_text   = _join("abstract", "introduction", "keywords", "methods", "unclassified") 
            ds_text         = _join("methods", "results", "discussion", "unclassified")  
            es_text         = _join("results", "discussion", "conclusion", "unclassified")
            
            # ========================================
            # STEP 2: Extract Entities (Buildings)
            # ========================================
            logger.info("\nSTEP 2: Extracting entities (buildings)...")
            entity_results = entity_extractor.extract_from_text(building_text, verbose=False, file_name=pdf_path)
            
            entities = entity_results.get('entities', [])
 
            # Verify entity context snippets against the source document.
            logger.info("  Verifying entity context snippets against source text...")
            entity_results = resolve_entity_contexts(entity_results, raw_text)
            entities = entity_results.get('entities', [])
            
            # Log entity summary
            if entities:
                logger.info(f"  ✓ Found {len(entities)} entities")
                for ent in entities:
                    name = ent.get('name', {}).get('value') if isinstance(ent.get('name'), dict) else ent.get('name')
                    city = ent.get('city', {}).get('value') if isinstance(ent.get('city'), dict) else ent.get('city')
                    country = ent.get('country', {}).get('value') if isinstance(ent.get('country'), dict) else ent.get('country')
                    
                    # Count verified fields for this entity
                    fields_with_verification = [
                        v for v in ent.values()
                        if isinstance(v, dict) and v.get("context_verified") is not None
                    ]
                    n_verified = sum(1 for f in fields_with_verification if f.get("context_verified"))
                    n_total    = len(fields_with_verification)
                    logger.info(
                        f"    - {name or 'Unnamed'} ({city or 'Unknown'}, {country or 'Unknown'})"
                        f"  [{n_verified}/{n_total} fields verified]"
                    )
            else:
                logger.info("  ✗ No entities identified")
            
            # ========================================
            # STEP 3: Extract Design Strategies
            # ========================================
            logger.info("\nSTEP 3: Extracting design strategies...")
            design_results = design_extractor.extract_from_text(ds_text, verbose=False, file_name=pdf_path)
 
            # Resolve anchors → real context passages from the source document
            logger.info("  Resolving design strategy anchors against source text...")
            design_results = resolve_design_strategy_contexts(design_results, raw_text)
 
            strategies = design_results.get('design_strategies', [])
            verified, total = _count_verified(strategies)
 
            # Semantic vocabulary matching — runs on all items (verified or not)
            logger.info("  Matching design strategies to vocabulary...")
            design_results = resolver.resolve_design_strategy_matches(design_results)
            strategies = design_results.get('design_strategies', [])
 
            if strategies:
                logger.info(f"  ✓ Found {total} design strategies ({verified}/{total} anchors verified)")
                for strategy in strategies[:3]:
                    name = strategy.get('name', 'Unnamed')
                    v = "✓" if strategy.get('anchor_verified') else "✗"
                    logger.info(f"    {v} {name}")
            else:
                logger.info("  ✗ No design strategies identified")
 
            # ========================================
            # STEP 4: Extract Ecosystem Services
            # ========================================
            logger.info("\nSTEP 4: Extracting ecosystem services...")
            ecosystem_results = ecosystem_extractor.extract_from_text(es_text, verbose=False, file_name=pdf_path)
 
            # Resolve anchors → real context passages from the source document
            logger.info("  Resolving ecosystem service anchors against source text...")
            ecosystem_results = resolve_ecosystem_service_contexts(ecosystem_results, raw_text)
 
            eco_services = ecosystem_results.get('ecosystem_services', [])
            verified_eco, total_eco = _count_verified(eco_services)
 
            # Semantic vocabulary matching — runs on all items (verified or not)
            logger.info("  Matching ecosystem services to vocabulary...")
            ecosystem_results = resolver.resolve_ecosystem_service_matches(ecosystem_results)
            eco_services = ecosystem_results.get('ecosystem_services', [])
 
            if eco_services:
                logger.info(f"  ✓ Found {total_eco} ecosystem services ({verified_eco}/{total_eco} anchors verified)")
                for service in eco_services[:3]:
                    name = service.get('name', 'Unnamed')
                    category = service.get('category', 'Unknown')
                    v = "✓" if service.get('anchor_verified') else "✗"
                    logger.info(f"    {v} {name} [{category}]")
            else:
                logger.info("  ✗ No ecosystem services identified")
            
            # ========================================
            # STEP 5: Combine Results into Single JSON
            # ========================================
            logger.info("\nSTEP 5: Combining results...")
            
            # Create combined results dictionary
            combined_results = {
                'entities': entity_results.get('entities', []),
                'design_strategies': design_results.get('design_strategies', []),
                'ecosystem_services': ecosystem_results.get('ecosystem_services', [])
            }
            
            # Save combined results to single JSON file
            pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            json_path = os.path.join(preprocessed_output_dir, f"{pdf_base_name}_extraction.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  ✓ Combined results saved to: {json_path}")
            
            # ========================================
            # STEP 6: Generate Combined Report
            # ========================================
            logger.info("\nSTEP 6: Generating report...")
 
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
                            value   = data.get('value')
                            context = data.get('context')
                            verified = data.get('context_verified')  # None = skipped (null value)
                        else:
                            value    = data
                            context  = None
                            verified = None
 
                        if value is not None:
                            report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
                            # Only show context snippet if it was confirmed in the source
                            if context and context != 'Extracted via QA' and verified is True:
                                report_lines.append(f"    Context: \"{context}\"")
            else:
                report_lines.append("  No entities found.")
 
            report_lines.append("\n")
 
            # Design strategies section
            report_lines.append("### DESIGN STRATEGIES ###")
            if strategies:
                verified_s, total_s = _count_verified(strategies)
                report_lines.append(f"  Anchor verification: {verified_s}/{total_s} verified\n")
                for idx, strategy in enumerate(strategies, 1):
                    if not strategy.get('anchor_verified'):
                        continue
                    name             = strategy.get('name', 'Unnamed')
                    anchor           = strategy.get('anchor_text', 'N/A')
                    context          = strategy.get('context')
                    anchor_score     = strategy.get('anchor_match_score')
                    vocab_top_matches = strategy.get('vocab_top_matches', [])
                    implementation_details = strategy.get('implementation_details', [])
 
                    anchor_score_str = f" (score: {anchor_score:.2f})" if anchor_score is not None else ""
                    report_lines.append(f"\nStrategy {idx}: {name}")
                    report_lines.append(f"  Anchor: {anchor} {anchor_score_str}")
                    
                    if vocab_top_matches:
                        report_lines.append(f"  Vocab top matches:")
                        for match in vocab_top_matches:
                            m_name = match.get('name', 'Unknown')
                            m_score = match.get('score', 0.0)
                            report_lines.append(f"    - {m_name} ({m_score:.4f})")
                    else:
                        report_lines.append(f"  Vocab top matches: N/A")
                    
                    if implementation_details:
                        report_lines.append(f"  Implementation details:")
                        for detail in implementation_details:
                            report_lines.append(f"    - {detail}")
                    else:
                        report_lines.append(f"  Implementation details: N/A")
                        
                    if context:
                        report_lines.append(f'  Context: "{context}"')
 
                if verified_s == 0:
                    report_lines.append("  No strategies with verified anchors.")
            else:
                report_lines.append("  No design strategies found.")
 
            report_lines.append("\n")
 
            # Ecosystem services section
            report_lines.append("### ECOSYSTEM SERVICES ###")
            if eco_services:
                verified_e, total_e = _count_verified(eco_services)
                report_lines.append(f"  Anchor verification: {verified_e}/{total_e} verified\n")
                by_category = {}
                for service in eco_services:
                    if not service.get('anchor_verified'):
                        continue
                    # Group by the category of the top vocab match if available,
                    # else fall back to the LLM-assigned category from the extractor
                    top_matches = service.get('vocab_top_matches', [])
                    vocab_cat = top_matches[0].get('category') if top_matches else None
                    cat = vocab_cat or service.get('category', 'Unknown')
                    by_category.setdefault(cat, []).append(service)
 
                if by_category:
                    for category, cat_services in by_category.items():
                        report_lines.append(f"\n  [{category.upper()}]")
                        for idx, service in enumerate(cat_services, 1):
                            name         = service.get('name', 'Unnamed')
                            anchor       = service.get('anchor_text', 'N/A')
                            context      = service.get('context')
                            anchor_score = service.get('anchor_match_score')
                            vocab_top_matches = service.get('vocab_top_matches', [])
 
                            anchor_score_str = f" (score: {anchor_score:.2f})" if anchor_score is not None else ""
                            report_lines.append(f"\n  Service {idx}: {name}")
                            report_lines.append(f"    Anchor: {anchor} {anchor_score_str}")
                            
                            if vocab_top_matches:
                                report_lines.append(f"    Vocab top matches:")
                                for match in vocab_top_matches:
                                    m_name = match.get('name', 'Unknown')
                                    m_score = match.get('score', 0.0)
                                    m_cat = match.get('category')
                                    cat_str = f" [{m_cat}]" if m_cat else ""
                                    report_lines.append(f"      - {m_name}{cat_str} ({m_score:.4f})")
                            else:
                                report_lines.append(f"    Vocab top matches: N/A")
                            if context:
                                report_lines.append(f'    Context: "{context}"')
                else:
                    report_lines.append("  No services with verified anchors.")
            else:
                report_lines.append("  No ecosystem services found.")
 
            report_lines.append("\n" + "=" * 80)
 
            report_path = os.path.join(preprocessed_output_dir, f"{pdf_base_name}_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
 
            logger.info(f"  ✓ Report saved to: {report_path}")
 
            # ========================================
            # Display Summary
            # ========================================
            d_v, d_t = _count_verified(strategies)
            e_v, e_t = _count_verified(eco_services)
            logger.info("\n--- EXTRACTION SUMMARY ---")
            logger.info(f"Entities:           {len(entities)}")
            logger.info(f"Design Strategies:  {d_t}  ({d_v} anchors verified, {d_t - d_v} unverified)")
            logger.info(f"Ecosystem Services: {e_t}  ({e_v} anchors verified, {e_t - e_v} unverified)")
            logger.info(f"Output: {json_path}")
 
            # ========================================
            # STEP 7: (FUTURE) Neo4j Integration
            # ========================================
            """
            Uncomment when ready to integrate with Neo4j:
 
            logger.info("\nSTEP 7: Processing with Neo4j pipeline...")
 
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
    logger.info(f"  - *_raw.md          (pymupdf4llm markdown)")
    logger.info(f"  - *_sections.json   (canonical section map)")
    logger.info(f"  - *_extraction.json (entities + design strategies + ecosystem services)")
    logger.info(f"  - *_report.txt      (human-readable, with anchor verification status)")
    logger.info(f"{'='*80}")
 
 
if __name__ == "__main__":
    asyncio.run(main())
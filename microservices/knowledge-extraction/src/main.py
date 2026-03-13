"""
Modified Neo4j pipeline with building information preprocessing
Now extracts entities, design strategies, AND ecosystem services
Orchestrated via LangGraph.
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
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

# Import the preprocessor and extractors as LangChain tools
from .building_preprocessor import preprocess_building_document
from .entity_extractor import extract_entities_from_text
from .design_strategy_extractor import extract_design_strategies_from_text
from .ecosystem_service_extractor import extract_ecosystem_services_from_text
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
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

def wait_for_ollama(host: str, max_retries: int = 40, delay: int = 3):
    """Wait for the Ollama server to become available."""
    logger.info(f"Waiting for Ollama at {host}...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("  ✓ Ollama is reachable.")
                return True
        except Exception:
            pass
        if i % 5 == 0 and i > 0:
            logger.info(f"  ...still waiting ({i}/{max_retries})")
        sleep(delay)
    logger.error("  ✗ Ollama not reachable after timeout.")
    return False

# Unused function
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

# ========================================
# LANGGRAPH STATE AND NODES
# ========================================

class PipelineState(TypedDict):
    """LangGraph State representing a single PDF flowing through the pipeline"""
    pdf_path: str
    preprocessed_output_dir: str
    
    # Preprocessor outputs
    raw_text: str
    sections: Dict[str, str] # academic paper sections, from paper_section_extractor.py
    
    # Tool instances
    resolver: EntityResolutionMatcher
    
    # Extractor outputs
    entities: List[Dict[str, Any]]
    design_strategies: List[Dict[str, Any]]
    ecosystem_services: List[Dict[str, Any]]
    
    # Final combined text representation of the report
    report: str

def preprocess_node(state: PipelineState) -> PipelineState:
    logger.info(f"\nSTEP 1: Converting PDF to text and classifying sections: {os.path.basename(state['pdf_path'])}")
    
    try:
        preprocess_result = preprocess_building_document.invoke({
            "pdf_path": state["pdf_path"],
            "output_dir": state["preprocessed_output_dir"]
        })
        
        if "error" in preprocess_result:
            logger.error(f"  ✗ Preprocessing failed: {preprocess_result['error']}")
            # Graph will still try to run but with empty text
            return {**state, "raw_text": "", "sections": {}}
        
        with open(preprocess_result['raw_text_path'], 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        sections = preprocess_result.get("sections")
        sections_dict = sections.to_dict() if sections else {}
        
        # Save the sections dict to a JSON file
        pdf_base_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
        sections_json_path = os.path.join(state["preprocessed_output_dir"], f"{pdf_base_name}_sections.json")
        with open(sections_json_path, 'w', encoding='utf-8') as f:
            json.dump(sections_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ Preprocessing complete. Body text: {len(raw_text)} chars. Sections saved to {os.path.basename(sections_json_path)}")
        return {**state, "raw_text": raw_text, "sections": sections_dict}
        
    except Exception as e:
        logger.error(f"Preprocessing node exception: {e}", exc_info=True)
        return {**state, "raw_text": "", "sections": {}}


def entity_node(state: PipelineState) -> PipelineState:
    logger.info("\nSTEP 2: Extracting entities (buildings)...")
    raw_text = state.get("raw_text", "")
    sections = state.get("sections", {})
    
    if not raw_text:
        return {**state, "entities": []}

    # Selective context: Prioritize Methods, Results, and Unclassified (Case Studies)
    selected_parts = []
    for k in ["abstract", "keywords", "methods", "results"]:
        if sections.get(k):
            selected_parts.append(f"--- {k.upper()} ---\n{sections[k]}")
            
    unclassified = sections.get("unclassified", {})
    if unclassified:
        for k, v in unclassified.items():
            selected_parts.append(f"--- {k.upper()} ---\n{v}")
            
    # Fallback to raw_text if our selective chunks are empty
    context = "\n\n".join(selected_parts) if selected_parts else raw_text
    
    if len(context) < len(raw_text) * 0.9:
         logger.info(f"  → Entity Extraction context selectively reduced from {len(raw_text)} to {len(context)} characters.")
    
    try:
        entity_results = extract_entities_from_text.invoke({"text": context, "file_name": state["pdf_path"]})
        
        logger.info("  Verifying entity context snippets against full source text...")
        entity_results = resolve_entity_contexts.invoke({"results": entity_results, "source_text": raw_text})
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
            
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Entity node exception: {e}", exc_info=True)
        return {"entities": []}


def design_strategy_node(state: PipelineState) -> PipelineState:
    logger.info("\nSTEP 3: Extracting design strategies...")
    raw_text = state.get("raw_text", "")
    sections = state.get("sections", {})
    
    if not raw_text:
        return {"design_strategies": []}
        
    # Selective context: Prioritize Abstract, Intro, Methods, Discussion
    selected_parts = []
    for k in ["abstract", "keywords", "methods", "results", "discussion"]:
        if sections.get(k):
            selected_parts.append(f"--- {k.upper()} ---\n{sections[k]}")

    unclassified = sections.get("unclassified", {})
    if unclassified:
        for k, v in unclassified.items():
            selected_parts.append(f"--- {k.upper()} ---\n{v}")
            
    context = "\n\n".join(selected_parts) if selected_parts else raw_text
    
    if len(context) < len(raw_text) * 0.9:
         logger.info(f"  → Design Strategy context selectively reduced from {len(raw_text)} to {len(context)} characters.")

    try:
        design_results = extract_design_strategies_from_text.invoke({"text": context, "file_name": state["pdf_path"]})

        logger.info("  Resolving design strategy anchors against full source text...")
        design_results = resolve_design_strategy_contexts.invoke({"results": design_results, "source_text": raw_text})

        logger.info("  Matching design strategies to vocabulary...")
        design_results = state["resolver"].resolve_design_strategy_matches(design_results)
        strategies = design_results.get('design_strategies', [])

        verified, total = _count_verified(strategies)
        if strategies:
            logger.info(f"  ✓ Found {total} design strategies ({verified}/{total} anchors verified)")
        else:
            logger.info("  ✗ No design strategies identified")
            
        return {"design_strategies": strategies}
    except Exception as e:
        logger.error(f"Design strategy node exception: {e}", exc_info=True)
        return {"design_strategies": []}


def ecosystem_service_node(state: PipelineState) -> PipelineState:
    logger.info("\nSTEP 4: Extracting ecosystem services...")
    raw_text = state.get("raw_text", "")
    sections = state.get("sections", {})
    
    if not raw_text:
        return {"ecosystem_services": []}
        
    # Selective context: Prioritize Results, Discussion, Conclusion
    selected_parts = []
    for k in ["abstract", "keywords", "results", "discussion", "conclusion"]:
        if sections.get(k):
            selected_parts.append(f"--- {k.upper()} ---\n{sections[k]}")

    unclassified = sections.get("unclassified", {})
    if unclassified:
        for k, v in unclassified.items():
            selected_parts.append(f"--- {k.upper()} ---\n{v}")     
                   
    context = "\n\n".join(selected_parts) if selected_parts else raw_text
    
    if len(context) < len(raw_text) * 0.9:
         logger.info(f"  → Ecosystem Services context selectively reduced from {len(raw_text)} to {len(context)} characters.")

    try:
        ecosystem_results = extract_ecosystem_services_from_text.invoke({"text": context, "file_name": state["pdf_path"]})

        logger.info("  Resolving ecosystem service anchors against source text...")
        ecosystem_results = resolve_ecosystem_service_contexts.invoke({"results": ecosystem_results, "source_text": raw_text})

        logger.info("  Matching ecosystem services to vocabulary...")
        ecosystem_results = state["resolver"].resolve_ecosystem_service_matches(ecosystem_results)
        eco_services = ecosystem_results.get('ecosystem_services', [])

        verified_eco, total_eco = _count_verified(eco_services)
        if eco_services:
            logger.info(f"  ✓ Found {total_eco} ecosystem services ({verified_eco}/{total_eco} anchors verified)")
        else:
            logger.info("  ✗ No ecosystem services identified")
            
        return {"ecosystem_services": eco_services}
    except Exception as e:
        logger.error(f"Ecosystem service node exception: {e}", exc_info=True)
        return {"ecosystem_services": []}


def assemble_node(state: PipelineState) -> PipelineState:
    logger.info("\nSTEP 5: Combining results...")
    
    pdf_path = state["pdf_path"]
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = state["preprocessed_output_dir"]
    
    entities = state.get("entities", [])
    strategies = state.get("design_strategies", [])
    eco_services = state.get("ecosystem_services", [])
    
    combined_results = {
        'entities': entities,
        'design_strategies': strategies,
        'ecosystem_services': eco_services
    }
    
    json_path = os.path.join(output_dir, f"{pdf_base_name}_extraction.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✓ Combined results saved to: {json_path}")
    
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
                    verified = data.get('context_verified')
                else:
                    value    = data
                    context  = None
                    verified = None

                if value is not None:
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
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
    report_str = '\n'.join(report_lines)

    report_path = os.path.join(output_dir, f"{pdf_base_name}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_str)

    logger.info(f"  ✓ Report saved to: {report_path}")
    
    return {**state, "report": report_str}


def build_pipeline_graph() -> StateGraph:
    """Constructs the LangGraph computational graph for the pipeline."""
    logger.info("Building LangGraph Pipeline Architecture...")
    
    workflow = StateGraph(PipelineState)
    
    # Add nodes representing independent tasks
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("extract_entities", entity_node)
    workflow.add_node("extract_strategies", design_strategy_node)
    workflow.add_node("extract_services", ecosystem_service_node)
    workflow.add_node("assemble_results", assemble_node)
    
    # Define edges - Parallel execution
    workflow.set_entry_point("preprocess")
    
    # After preprocessing, we fan-out to the three independent extractors
    workflow.add_edge("preprocess", "extract_entities")
    workflow.add_edge("preprocess", "extract_strategies")
    workflow.add_edge("preprocess", "extract_services")
    
    # We fan-in to assemble results explicitly (LangGraph handles barrier sync on parallel branches natively)
    workflow.add_edge("extract_entities", "assemble_results")
    workflow.add_edge("extract_strategies", "assemble_results")
    workflow.add_edge("extract_services", "assemble_results")
    
    workflow.add_edge("assemble_results", END)
    
    # Compile the graph
    app = workflow.compile()
    return app


async def main():
    # Ensure Ollama is up before proceeding
    if not wait_for_ollama(OLLAMA_HOST):
        logger.error("Ollama is not responding. Exiting pipeline.")
        return
    
    logger.info("Initializing entity resolution matcher (pre-embedding vocabularies)...")
    resolver = EntityResolutionMatcher(
        ollama_host=OLLAMA_HOST,
        embedding_model=OLLAMA_EMBEDDING_MODEL,
    )
    
    # Build LangGraph Application
    app = build_pipeline_graph()
    
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
    
    # Process each PDF using LangGraph
    for i, pdf_path in enumerate(pdf_files_to_process):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing PDF {i+1}/{len(pdf_files_to_process)}: {os.path.basename(pdf_path)}")
        logger.info(f"{'='*80}")
        
        # Initial State
        initial_state = {
            "pdf_path": pdf_path,
            "preprocessed_output_dir": preprocessed_output_dir,
            "raw_text": "",
            "sections": {},
            "resolver": resolver,
            "entities": [],
            "design_strategies": [],
            "ecosystem_services": [],
            "report": ""
        }
        
        try:
            # Execute Graph
            final_state = await app.ainvoke(initial_state)
            
            # Display Summary from Final State
            entities = final_state.get('entities', [])
            strategies = final_state.get('design_strategies', [])
            eco_services = final_state.get('ecosystem_services', [])
            
            d_v, d_t = _count_verified(strategies)
            e_v, e_t = _count_verified(eco_services)
            
            logger.info("\n--- EXTRACTION SUMMARY ---")
            logger.info(f"Entities:           {len(entities)}")
            logger.info(f"Design Strategies:  {d_t}  ({d_v} anchors verified, {d_t - d_v} unverified)")
            logger.info(f"Ecosystem Services: {e_t}  ({e_v} anchors verified, {e_t - e_v} unverified)")
            
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
    logger.info(f"  - *_extraction.json (entities + design strategies + ecosystem services)")
    logger.info(f"  - *_report.txt (human-readable, with anchor verification status)")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
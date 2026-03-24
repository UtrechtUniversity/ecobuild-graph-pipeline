"""
Multi-stage Information Extraction System for Sustainable Case Studies
Uses keyword detection + targeted LLM prompts to minimize unstructured processing
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from llama_index.core.llms import LLM

from .llama_index_interface import LlamaIndexInterface

# ── Few-shot example directory (relative to this file) ──────────────────────
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "test_papers" / "examples"
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

output_dir = Path("/app/test_papers/preprocessed")


@dataclass
class Entity:
    """Entity information structure (Building, Infrastructure, Study, etc.)"""
    
    name: Optional[str] = None
    type: str = "Unknown"  # e.g., Building, Infrastructure, Research Study
    country: Optional[str] = None
    city: Optional[str] = None
    street: Optional[str] = None
    year: Optional[int] = None

class OllamaPromptBuilder:
    """Builds targeted prompts for Ollama based on detected information"""

    # ── Few-shot example helpers ─────────────────────────────────────────

    @staticmethod
    def load_all_building_examples() -> List[dict]:
        """Load all section-skeleton few-shot examples for building extraction.
        Globs for building_extraction_example_*.json in the examples directory.
        Returns an empty list if no files are found so the prompt still works."""
        pattern = "building_extraction_example_*.json"
        example_paths = sorted(_EXAMPLES_DIR.glob(pattern))
        if not example_paths:
            logger.warning(f"No few-shot examples matching {pattern} in {_EXAMPLES_DIR} — skipping.")
            return []
        examples = []
        for path in example_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    examples.append(json.load(f))
                    logger.info(f"Loaded few-shot example: {path.name}")
            except Exception as e:
                logger.warning(f"Failed to load few-shot example {path.name}: {e} — skipping.")
        return examples

    @staticmethod
    def format_example_block(example: dict, index: int = 1, total: int = 1) -> str:
        """Format the loaded example dict into a prompt-ready text block."""
        skeleton = example.get("section_skeleton", "")
        expected = json.dumps(example.get("expected_output", {}), indent=2)
        lesson  = example.get("lesson", "")
        citation = example.get("citation", f"Example {index}")

        return (
            f"\n--- FEW-SHOT EXAMPLE {index} of {total} (DIFFERENT PAPER — DO NOT COPY): {citation} ---\n"
            f"{skeleton}\n\n"
            f"Correct extraction from the example paper above (NOT your task):\n{expected}\n\n"
            f"KEY LESSON: {lesson}\n"
            f"--- END OF EXAMPLE {index} — THE ABOVE IS NOT THE PAPER YOU SHOULD EXTRACT FROM ---\n"
        )

    # ── Main prompt builder ──────────────────────────────────────────────

    @staticmethod
    def build_building_extraction_prompt(text: str, file_name: str = "") -> str:
        """Build prompt for extracting primary entities and their metadata"""
        
        base_name = Path(file_name).stem

        # Optionally inject few-shot examples
        examples = OllamaPromptBuilder.load_all_building_examples()
        if examples:
            example_blocks = "\n".join(
                OllamaPromptBuilder.format_example_block(ex, i + 1, len(examples))
                for i, ex in enumerate(examples)
            )
        else:
            example_blocks = ""

        current_prompt = f"""
        You are an expert in academic information extraction.
        {example_blocks}

        ════════════════════════════════════════════════════
        EXAMPLES COMPLETE — YOUR ACTUAL TASK BEGINS NOW
        ════════════════════════════════════════════════════
        The examples above demonstrate the EXTRACTION PROCESS only. All their content (Singapore,
terraced houses, etc.) is irrelevant — do NOT reproduce any of it.
        Extract only from the paper text below. 
        
        Paper text:
        {text}

        ---
        Based on the paper text ABOVE, identify and extract all primary buildings discussed in the academic study conducted.
        These could be one or more specific sustainable buildings.
        Identify which building(s) are being STUDIED (not just mentioned) and extract ONLY the case study building(s) 
        DO NOT extract company names, or other entities which are not buildings.

        LOOK FOR PHRASES LIKE:
        - "case study"
        - "for this study" 
        - "for this research" 

        IGNORE PHRASES LIKE:
        - "previous studies"
        - Literature review sections
        
        Do not use data about the authors or the journal metadata.
        For each building, provide its name, location details and year of construction. 
        
        Reply with a SINGLE JSON object only. No preamble, no conversational filler.
        The JSON object MUST follow this structure:
        {{
            "entities": [
                {{
                    "building_type": {{"value": "building type", "context": "snippet"}},
                    "country": {{"value": "country name", "context": "snippet"}},
                    "city": {{"value": "city name", "context": "snippet"}},
                    "street": {{"value": "street address", "context": "snippet"}},
                    "year_of_construction": {{"value": 2024, "context": "snippet"}}
                }}
            ]
        }}
        CRITICAL: For EVERY field of every JSON object, provide both the "value" AND a short verbatim "context" snippet (5-7 words) from the text to 
        serve as anchor text for a context resolver. 
        Use null for values if information is not found.

        JSON output (values must come ONLY from the paper text above):"""

        # # Save prompt text
        # prompt_path = output_dir / f"{base_name}_building_extraction_prompt.txt"
        # with open(prompt_path, 'w', encoding='utf-8') as f:
        #     f.write(current_prompt)
        # logger.info(f"  Saved prompt text: {prompt_path}")

        return current_prompt

class EntityInformationExtractor:
    """Main extraction orchestrator"""
    
    def __init__(self, model: LLM):
        self.prompt_builder = OllamaPromptBuilder()
        self.llm_interface = LlamaIndexInterface(model)
    
    def extract_from_text(self, text: str, verbose: bool = True, file_name: str = "") -> Dict:
        """
        Extract all entities from a research paper text
        
        Args:
            text: Full text of the research paper
            verbose: Print progress information
        
        Returns:
            Dictionary containing a list of extracted entities
        """
        if verbose:
            print("Extracting entities from text...")

        prompt = self.prompt_builder.build_building_extraction_prompt(text, file_name)
        response = self.llm_interface.query(prompt)
        
        if verbose and not response:
            print("  Warning: Empty response from Ollama interface.")
        
        raw_results = self.llm_interface.extract_json(response)
        
        # Debug: if entities is empty, log the response for inspection
        if not raw_results.get('entities') and verbose:
            print(f"  Debug: Raw LLM response was: {response[:200]}...")
        
        # Ensure we have a list of entities
        entities = raw_results.get('entities', [])
        if isinstance(raw_results, dict) and not entities and any(key in raw_results for key in ['name', 'type', 'city']):
            # Fallback if it returned a single object instead of a list
            entities = [raw_results]
        
        results = {
            'entities': entities
        }
        
        if verbose:
            print(f"  Found {len(entities)} entities.")
            for ent in entities[:3]:  # Show first 3
                name = ent.get('name', {}).get('value') if isinstance(ent.get('name'), dict) else ent.get('name')
                etype = ent.get('type', {}).get('value') if isinstance(ent.get('type'), dict) else ent.get('type')
                print(f"    - {name or 'Unnamed'} ({etype or 'Unknown type'})")
        
        return results
    
    def extract_from_file(self, filepath: str, verbose: bool = True) -> Dict:
        """
        Extract information from a text file
        
        Args:
            filepath: Path to the text file
            verbose: Print progress information
        
        Returns:
            Dictionary containing all extracted information
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.extract_from_text(text, verbose)
    
    def save_results(self, results: Dict, output_path: str):
        """Save extraction results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable report from results"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ENTITY EXTRACTION REPORT")
        report_lines.append("=" * 80)
        
        entities = results.get('entities', [])
        if not entities:
            report_lines.append("\n  No entities identified.")
        else:
            for i, entity in enumerate(entities, 1):
                report_lines.append(f"\n### Entity {i} ###")
                for key, data in entity.items():
                    # Handle nested structure
                    if isinstance(data, dict) and 'value' in data:
                        value = data.get('value')
                        context = data.get('context')
                    else:
                        value = data
                        context = None
                    
                    if value is not None:
                        report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
                        if context:
                            report_lines.append(f"    Context: \"{context}\"")
                    else:
                        report_lines.append(f"  {key.replace('_', ' ').title()}: Not found")
        
        return "\n".join(report_lines)


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python entity_extractor.py <input_text_file> [output_json_file]")
        print("\nExample:")
        print("  python entity_extractor.py paper.txt results.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extraction_results.json"
    
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print()
    
    # Initialize extractor
    extractor = EntityInformationExtractor(model="llama3.2")
    
    # Extract information
    results = extractor.extract_from_file(input_file, verbose=True)
    
    # Save results
    extractor.save_results(results, output_file)
    print(f"\n✓ Results saved to {output_file}")
    
    # Generate and print report
    report = extractor.generate_report(results)
    print("\n" + report)
    
    # Save report
    report_file = output_file.replace('.json', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Report saved to {report_file}")


if __name__ == "__main__":
    main()

"""
Multi-stage Information Extraction System for Sustainable Case Studies
Uses keyword detection + targeted LLM prompts to minimize unstructured processing
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# ── Few-shot example directory (relative to this file) ──────────────────────
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "test_papers" / "examples"
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


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
    def _load_all_building_examples() -> List[dict]:
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
    def _format_example_block(example: dict, index: int = 1, total: int = 1) -> str:
        """Format the loaded example dict into a prompt-ready text block."""
        skeleton = example.get("section_skeleton", "")
        expected = json.dumps(example.get("expected_output", {}), indent=2)
        lesson  = example.get("lesson", "")
        citation = example.get("citation", f"Example {index}")

        return (
            f"\n--- FEW-SHOT EXAMPLE {index} of {total}: {citation} ---\n"
            f"{skeleton}\n\n"
            f"Correct extraction from the example paper above:\n{expected}\n\n"
            f"KEY LESSON: {lesson}\n"
            f"--- END OF EXAMPLE {index} ---\n"
        )

    # ── Main prompt builder ──────────────────────────────────────────────

    @staticmethod
    def build_building_extraction_prompt(text: str) -> str:
        """Build prompt for extracting primary entities and their metadata"""

        # Optionally inject few-shot examples
        examples = OllamaPromptBuilder._load_all_building_examples()
        if examples:
            example_blocks = "\n".join(
                OllamaPromptBuilder._format_example_block(ex, i + 1, len(examples))
                for i, ex in enumerate(examples)
            )
        else:
            example_blocks = ""

        return f"""
        You are an expert in academic information extraction.
        {example_blocks}
        Now extract from THIS paper:

        Paper text:
        {text}

        ---
        Based on the paper text ABOVE, identify and extract all primary buildings discussed in the academic study conducted.
        These could be one or more specific sustainable buildings.
        Identify which building(s) are being STUDIED (not just mentioned) and extract ONLY the case study building(s) 

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
                    "name": {{"value": "entity name", "context": "snippet"}},
                    "country": {{"value": "country name", "context": "snippet"}},
                    "city": {{"value": "city name", "context": "snippet"}},
                    "street": {{"value": "street address", "context": "snippet"}},
                    "year_of_construction": {{"value": 2024, "context": "snippet"}}
                }}
            ]
        }}
        CRITICAL: For EVERY field of every JSON object, provide both the "value" AND a short verbatim "context" snippet (max 30 words) from the text as evidence. 
        Use null for values if information is not found.

        JSON output:"""


class OllamaInterface:
    """Interface to interact with Ollama via API (Docker-compatible)"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def query(self, prompt: str) -> str:
        """Send query to Ollama API and return response"""
        
        try:
            import requests
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.01,  # Low for factual extraction
                       }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f'{{"error": "API returned status {response.status_code}"}}'
                
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'
    
    def extract_json(self, response: str) -> dict:
        """Extract and parse JSON from LLM response"""
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                # Clean up markdown if present
                content = json_match.group()
                if content.startswith('```json'):
                    content = content[7:-3]
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return {}


class EntityInformationExtractor:
    """Main extraction orchestrator"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.prompt_builder = OllamaPromptBuilder()
        self.ollama = OllamaInterface(model, base_url)
    
    def extract_from_text(self, text: str, verbose: bool = True) -> Dict:
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
        
        if len(text) > 15000:
            if verbose:
                print(f"  Warning: Text is very long ({len(text)} chars). Truncating to 15000 chars for LLM context.")
                print(f"Kept text: {text[:15000]}\n")
            text = text[:15000] + "\n... [TRUNCATED] ..."

        prompt = self.prompt_builder.build_building_extraction_prompt(text)
        response = self.ollama.query(prompt)
        
        if verbose and not response:
            print("  Warning: Empty response from Ollama interface.")
        
        raw_results = self.ollama.extract_json(response)
        
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

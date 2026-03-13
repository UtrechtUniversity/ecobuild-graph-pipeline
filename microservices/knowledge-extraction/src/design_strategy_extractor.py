"""
Design Strategy Extraction System for Sustainable Building Case Studies
Extracts design strategies/systems implemented in the case study building
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# ── Few-shot example directory (relative to this file) ──────────────────────
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "test_papers" / "examples"

logger = logging.getLogger(__name__)

output_dir = Path("/app/test_papers/preprocessed")

class DesignStrategyPromptBuilder:
    """Builds targeted prompts for extracting design strategies"""
    
    # ── Few-shot example helpers ─────────────────────────────────────────

    @staticmethod
    def _load_all_design_strategy_examples() -> List[dict]:
        """Load all section-skeleton few-shot examples for design strategy extraction.
        Globs for design_strategy_extraction_example_*.json in the examples directory.
        Returns an empty list if no files are found so the prompt still works."""
        pattern = "design_strategy_extraction_example_*.json"
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
        lesson = example.get("lesson", "")
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
    def build_design_strategy_extraction_prompt(text: str, file_name: str = "") -> str:
        """Build prompt for extracting design strategies from case study"""

        base_name = Path(file_name).stem

        # Optionally inject few-shot examples
        examples = DesignStrategyPromptBuilder._load_all_design_strategy_examples()
        if examples:
            example_blocks = "\n".join(
                DesignStrategyPromptBuilder._format_example_block(ex, i + 1, len(examples))
                for i, ex in enumerate(examples)
            )
        else:
            example_blocks = ""

        current_prompt = f"""{example_blocks}
        
            The examples above demonstrate the EXTRACTION PROCESS only. The content in them should NOT be used or repeated — extract only from the paper provided below. 

            Paper text:
            {text}

            ---
            You are an expert in sustainable building design and renewable energy systems.

            Your task: Extract ALL design strategies, systems, or technologies implemented in the CASE STUDY BUILDING (not from literature review or other examples).

            A design strategy for a building is a deliberate intervention or set of interventions implemented to achieve specific functional, environmental, social, or economic objectives. 
            It can include both overarching design approaches and concrete measures such as solar panels, green walls, or water filtration systems when they are intentionally applied to meet defined performance goals.
            A sustainable design strategy specifically refers to those interventions that generate ecosystem services, aim to reduce environmental impact, enhance resource efficiency, and contribute positively to ecological 
            and social systems across the building's life cycle. 

            Equivalent terms that also refer to design strategies are:
            - design solution
            - design intervention
            - design type

            Design strategies are often the independent variables in the study. They are manipulated or changed to achieve a specific outcome (ecosystem services).

            LOOK FOR PHRASES LIKE:
            - "for this study" 
            - "for this research" 
            - "Methods"
            - "Results"

            IGNORE PHRASES LIKE:
            - "previous studies"
            - Literature review sections

            For EACH design strategy found, provide:
            1. name: The strategy name or system type
            2. anchor: A SHORT EXACT PHRASE of 5-10 words copied verbatim from the paper that
            best locates where this design strategy is discussed. This phrase will be searched in the
            source document to retrieve the surrounding passage — it must exist exactly as
            written in the paper text above.
            RULES FOR ANCHOR:
            - Copy 5-10 consecutive words exactly as they appear in the paper text above
            - Choose words that are specific and unique to this design strategy mention
            - Do NOT paraphrase, summarise, or change any words
            - If you cannot identify a specific passage in the text above, set anchor to null
            3. implementation_details: A list of the particularities of the design strategy's implementation in the case study building. Each detail should be a short string of 5-10 words, copied EXACTLY as it is written in the paper text above.

            Reply with a SINGLE JSON object only. No preamble, no conversational filler.

            The JSON object MUST follow this structure:
            {{
                "design_strategies": [
                    {{
                        "name": "strategy name",
                        "anchor": "five to ten exact words from the paper",
                        "confidence": 0.8,
                        "implementation_details": ["detail 1", "detail 2", "detail 3"]
                    }},
                    {{
                        "name": "another strategy name",
                        "anchor": null,
                        "confidence": 0.4,
                        "implementation_details": ["detail 1", "detail 2"]
                    }}
                ]
            }}

            CRITICAL RULES:
            - Extract ONLY strategies from the case study building (often in "Case Study" or "Methods" sections)
            - Do NOT extract strategies mentioned in literature review or introduction
            - The anchor must be a verbatim copy of words from the paper text above — do not
            invent or paraphrase
            - If you cannot find a passage in the text that evidences a strategy, set
            anchor to null rather than guessing
            - Include ALL design strategies found, even if multiple models/configurations- Use null for name if no strategies found

            JSON output (values must come ONLY from the paper text above):"""
        # Save prompt text
        # prompt_path = output_dir / f"{base_name}_design_strategy_extraction_prompt.txt"
        # with open(prompt_path, 'w', encoding='utf-8') as f:
        #     f.write(current_prompt)
        # logger.info(f"  ✓ Saved prompt text: {prompt_path}")

        return current_prompt


class DesignStrategyExtractor:
    """Main design strategy extraction orchestrator"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.prompt_builder = DesignStrategyPromptBuilder()
        self.model = model
        self.base_url = base_url
    
    def _query_ollama(self, prompt: str) -> str:
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
                        "temperature": 0.1,  # Low for factual extraction
                        "num_predict": 2000,  # Allow longer responses for multiple strategies
                        "num_ctx": 12000,     # Prevent prompt truncation
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"API returned status {response.status_code}")
                return '{}'
                
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return '{}'
    
    def _extract_json(self, response: str) -> dict:
        """Extract and parse JSON from LLM response"""
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.debug(f"Failed to parse: {json_match.group()[:500]}")
        
        return {'design_strategies': []}
    
    def extract_from_text(self, text: str, verbose: bool = True, file_name: str = "") -> Dict:
        """
        Extract design strategies from a research paper text.

        Returns raw LLM output with 'anchor' fields. Call
        resolve_design_strategy_contexts() from context_resolver.py afterwards
        to replace anchors with verified context passages.

        Args:
            text: Full text of the research paper
            verbose: Print progress information

        Returns:
            Dictionary containing list of design strategies (with anchor fields,
            not yet resolved to context passages)
        """
        if verbose:
            logger.info("Extracting design strategies from text...")
        
        # Build and send prompt
        prompt = self.prompt_builder.build_design_strategy_extraction_prompt(text, file_name)
        response = self._query_ollama(prompt)
        
        if verbose and not response:
            logger.warning("Empty response from Ollama")
        
        # Extract JSON
        raw_results = self._extract_json(response)
        
        # Debug output
        if not raw_results.get('design_strategies') and verbose:
            logger.debug(f"Raw LLM response: {response[:300]}...")
        
        strategies = raw_results.get('design_strategies', [])
        
        if verbose:
            logger.info(f"Found {len(strategies)} design strategies")
            for strategy in strategies[:5]:
                name = strategy.get("name", "Unnamed")
                anchor = strategy.get("anchor")
                anchor_display = f'"{anchor[:40]}..."' if anchor else "null"
                logger.info(f"  - {name} | anchor: {anchor_display}")
        
        return {'design_strategies': strategies}
    
    def extract_from_file(self, filepath: str, verbose: bool = True) -> Dict:
        """
        Extract design strategies from a text file
        
        Note: for anchor resolution you will need the same text string — either
        re-read the file or pass raw_text directly to extract_from_text(). 
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
        report_lines.append("DESIGN STRATEGY EXTRACTION REPORT")
        report_lines.append("=" * 80)
        
        strategies = results.get('design_strategies', [])
        
        if not strategies:
            report_lines.append("\n  No design strategies identified.")
        else:
            report_lines.append(f"\nTotal strategies found: {len(strategies)}\n")
            
            for i, strategy in enumerate(strategies, 1):
                name = strategy.get("name", "Unnamed Strategy")
                confidence = strategy.get("confidence", "N/A")
                context = strategy.get("context")           # resolved passage (str|None)
                anchor_text = strategy.get("anchor_text")  # original LLM anchor
                verified = strategy.get("anchor_verified")
                score = strategy.get("anchor_match_score")

                report_lines.append(f"### Strategy {i}: {name} ###")
                report_lines.append(f"  Confidence: {confidence}")

                # Verification status line
                if verified is not None:
                    status = "✓ verified" if verified else "✗ UNVERIFIED"
                    score_str = f" (match score: {score:.2f})" if score is not None else ""
                    report_lines.append(f"  Anchor: {status}{score_str}")
                    if anchor_text:
                        report_lines.append(f'  Anchor text: "{anchor_text}"')

                if context:
                    report_lines.append(f"\n  Context passage:")
                    report_lines.append(f'    "{context}"')
                else:
                    report_lines.append("\n  Context passage: not retrieved")

                report_lines.append("")  # blank line between strategies
        
        report_lines.append("=" * 80)
        return "\n".join(report_lines)


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python design_strategy_extractor.py <input_text_file> [output_json_file]")
        print("\nExample:")
        print("  python design_strategy_extractor.py paper.txt design_strategies.json")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "design_strategies.json"

    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}\n")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    extractor = DesignStrategyExtractor(model="llama3.2")

    # Step 1: extract (produces anchors)
    results = extractor.extract_from_text(raw_text, verbose=True)

    # Step 2: resolve anchors → real context passages
    from context_resolver import resolve_design_strategy_contexts
    results = resolve_design_strategy_contexts(results, raw_text)
    
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
    print(f"✓ Report saved to {report_file}")


if __name__ == "__main__":
    main()




## Dump for unused prompt text
# LOOK FOR:
# - "Model 1", "Model 2", "Model 3" (different design configurations)
# - "photovoltaic system", "PV panels", "solar panels"
# - "solar heating system", "SHS", "solar collectors"
# - "grid-connected", "zero energy balance"
# - Specific equipment (inverters, boilers, collectors)
# - "implemented", "installed", "adopted", "used"
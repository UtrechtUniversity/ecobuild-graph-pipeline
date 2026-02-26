"""
Design Strategy Extraction System for Sustainable Building Case Studies
Extracts design strategies/systems implemented in the case study building
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DesignStrategyPromptBuilder:
    """Builds targeted prompts for extracting design strategies"""
    
    @staticmethod
    def build_design_strategy_extraction_prompt(text: str) -> str:
        """Build prompt for extracting design strategies from case study"""
        
        return f"""Paper text:
{text}

---
You are an expert in sustainable building design and renewable energy systems.

Your task: Extract ALL design strategies, systems, or technologies implemented in the CASE STUDY BUILDING (not from literature review or other examples).

DESIGN STRATEGIES include (but are not limited to):
- Solar energy systems (photovoltaic panels, solar heating systems, BIPV)
- Green infrastructure (green roofs, green walls, vertical gardens)
- Water management (rainwater harvesting, greywater recycling, stormwater management)
- Energy efficiency measures (insulation, passive design, natural ventilation)
- Renewable energy (wind, geothermal, biomass)
- Smart building systems (BIM, energy management systems)
- Sustainable materials (recycled, bio-based, local materials)

For EACH design strategy found, provide:
1. name: The strategy name or system type
2. context: A list of medium-length quotes (30-50 words) showing where/how it's mentioned

Reply with a SINGLE JSON object only. No preamble, no conversational filler.

The JSON object MUST follow this structure:
{{
    "design_strategies": [
        {{
            "name": "strategy name",
            "context": [
                "first quote showing this strategy (30-50 words from paper)",
                "second quote if mentioned again (30-50 words from paper)"
            ]
        }},
        {{
            "name": "another strategy name",
            "context": [
                "quote about this strategy (30-50 words from paper)"
            ]
        }}
    ]
}}

CRITICAL RULES:
- Extract ONLY strategies from the case study building (often in "Case Study" or "Methods" sections)
- Do NOT extract strategies mentioned in literature review or introduction
- Each quote in "context" must be 30-50 words, verbatim from the text
- Include ALL design strategies found, even if multiple models/configurations
- Use null for name if no strategies found

JSON output:"""


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
                        "temperature": 0.2,  # Low for factual extraction
                        "num_predict": 2000  # Allow longer responses for multiple strategies
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
    
    def extract_from_text(self, text: str, verbose: bool = True) -> Dict:
        """
        Extract design strategies from a research paper text
        
        Args:
            text: Full text of the research paper
            verbose: Print progress information
        
        Returns:
            Dictionary containing list of design strategies
        """
        if verbose:
            logger.info("Extracting design strategies from text...")
        
        # Truncate if too long
        if len(text) > 15000:
            if verbose:
                logger.warning(f"Text truncated from {len(text)} to 15000 chars")
            text = text[:15000]
        
        # Build and send prompt
        prompt = self.prompt_builder.build_design_strategy_extraction_prompt(text)
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
            for strategy in strategies[:5]:  # Show first 5
                name = strategy.get('name', 'Unnamed')
                num_contexts = len(strategy.get('context', []))
                logger.info(f"  - {name} ({num_contexts} mentions)")
        
        return {'design_strategies': strategies}
    
    def extract_from_file(self, filepath: str, verbose: bool = True) -> Dict:
        """
        Extract design strategies from a text file
        
        Args:
            filepath: Path to the text file
            verbose: Print progress information
        
        Returns:
            Dictionary containing extracted design strategies
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
                name = strategy.get('name', 'Unnamed Strategy')
                contexts = strategy.get('context', [])
                
                report_lines.append(f"### Strategy {i}: {name} ###")
                report_lines.append(f"  Mentions: {len(contexts)}")
                
                for j, context in enumerate(contexts, 1):
                    report_lines.append(f"\n  Quote {j}:")
                    report_lines.append(f'    "{context}"')
                
                report_lines.append("")  # Blank line between strategies
        
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
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "design_strategies.json"
    
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}\n")
    
    # Initialize extractor
    extractor = DesignStrategyExtractor(model="llama3.2")
    
    # Extract design strategies
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
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
- Aeroponic systems
- Airdrop irrigation systems
- Aquaponic systems
- Beehives
- Biofilters
- Biomass energy collection
- Biophilic design
- Bioremediation
- Black/greywater treatment
- Blue-green roofs
- Blueroofs
- Botanical garden
- Burning hydrogen
- Cisterns
- Coated glass
- Community gardens
- Compact construction
- Companion planting
- Composting
- Cradle to cradle design
- Cross ventilation
- Delayed drainage systems
- Design for deconstruction
- Downspout strategies
- Drip irrigation systems
- Dynamic/aerodynamic architecture
- Ecological analogue design
- Edible Landscapes
- Elevated constructions
- Extensive green roof
- Façade cladding
- Fit form to function
- Floating contructions
- Fog harvesting
- Food forests
- Green corridors
- Green façade
- Green wall system
- Greenhouse agriculture
- Heat-cold storage installation
- Hedgerows consisting of edible plants
- Herb gardens
- High albedo materials
- Hydro turbines
- Hydrogen burning system
- Hydronic radiant system
- Hydroponic systems
- Increase of native plants
- Increase species diversity
- Indoor (food) gardens
- Infiltration systems for excess rain water
- Insect attracting structures
- Insulation material
- Integration of retaining walls
- Integration of thermal mass
- Intensive green roof
- Intercropping
- Interlocking panels
- Introduction of keystone species in property
- Leakage control
- Light filtering for temperature keeping
- Living Plant Constructions (Baubotanik)
- Location planning with little sun and wind
- Low albedo materials
- Medicinal gardens
- Micro livestock
- Microbial fuel cells
- Moss wall
- Noise reducing technologies
- Organic farming
- Orientation sound barriers
- Permaculture
- Permeable paving system
- Photovoltaic cells integration
- Photovoltaic panels
- Pile foundations
- Placed on columns or stilts
- Planter green wall
- Pocket garden/park
- Private gardens
- Rainwater Cooling System
- Rainwater harvesting systems
- Reduction of light pollution
- Replace equipment or install water saving devices
- Retaining wall system
- Revegetation
- Roof ponds
- Sacrificial ground floors
- Seismic architecture
- Semi-intensive green roof
- Separation of waste streams
- Smart irrigation
- Smart roofs
- Soakwells
- Soil filtration
- Soil quality monitoring
- Solar thermal collectors
- Solar water heaters
- Solid walls
- Source separation of wastewater
- Stack ventilation
- Stepping stone habitats
- Structural bracing strategies
- Sunscreens
- Sustainably sourced materials
- Texture and form based sound barriers
- Thermal desorption
- Thermal energy storage system
- Trellis/fence farms
- Urban mining
- Urban orchards
- Urban (rooftop) farming
- Use of biodegradable materials
- Use of carbon/GHG storing building materials
- Use of daylight
- Use of locally sourced materials
- Use of mulch
- Use of pre-existing vegetation
- Use of readily available materials
- Use of recycled materials
- Use of traditional techniques
- Vegetable gardens
- Vegetated grid pave
- Vegetated pergola
- Vegetation cover increase
- Vegetation for insulation
- Vertical farming
- Vertical mobile garden
- Waste management
- Water cooling systems
- Water efficient systems
- Water less systems
- Water source heat pump
- Water storage
- Wind barriers
- Wind turbines
- Xeriscaping

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
3. confidence: A score from 0.0 to 1.0 indicating how confident you are that the paper's own case study genuinely evidences this strategy (1.0 = explicitly discussed with data/results, 0.5 = implied but not primary focus, 0.1 = only tangentially mentioned)

Reply with a SINGLE JSON object only. No preamble, no conversational filler.

The JSON object MUST follow this structure:
{{
    "design_strategies": [
        {{
            "name": "strategy name",
            "anchor": "five to ten exact words from the paper",
            "confidence": 0.8
        }},
        {{
            "name": "another strategy name",
            "anchor": null,
            "confidence": 0.4
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
                        "temperature": 0.01,  # Low for factual extraction
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
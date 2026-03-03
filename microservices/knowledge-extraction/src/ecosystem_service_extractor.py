"""
Ecosystem Service Extraction System for Sustainable Building Case Studies
Extracts ecosystem services evidenced in the case study research of a paper
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Full taxonomy of ecosystem services ──────────────────────────────────────
# Each entry: (name, category, description)
ECOSYSTEM_SERVICES_TAXONOMY = [
    # --- PROVISIONING SERVICES ---
    ("Food / nutrition for humans",
     "Provisioning",
     "Vast range of food products consumed by humans derived from plants, animals, and microbes; includes cultivated and wild terrestrial and aquatic plants, fungi, algae, as well as animals reared or caught for food or nutritional purposes."),
    ("Food / nutrition for non-humans",
     "Provisioning",
     "Vast range of food products consumed by non-humans derived from plants, animals, and microbes; includes cultivated and wild terrestrial and aquatic plants, fungi, algae, as well as animals reared or caught for food or nutritional purposes."),
    ("Raw materials",
     "Provisioning",
     "Non-living materials, like timber, fiber, stone, minerals, ores, derived from reared or wild animals or cultivated or wild plants, fungi, algae and bacteria, used directly or processed for various purposes including constructions, industrial production and for ornaments."),
    ("Upcyclable materials",
     "Provisioning",
     "Existing materials or products that can be repurposed and transformed into products of higher value, quality, function, or aesthetics."),
    ("Recyclable materials",
     "Provisioning",
     "Existing materials that can be processed or broken down into new raw materials and remanufactured into new products."),
    ("Reusable materials",
     "Provisioning",
     "Existing materials or products that can be used multiple times for the same or a different purpose extending the material or product life."),
    ("Solid fuel",
     "Provisioning",
     "Combustible material in a solid state including biomass, coal, wood, that is usually used to produce energy mainly for heating, cooking, electricity, or other industrial processes."),
    ("Fuel gas",
     "Provisioning",
     "Fuel in a gaseous state at standard temperature (0 °C) and pressure (1 atm) including biogas, natural gas, hydrogen, that is usually used to produce energy mainly for cooking, heating, and electricity."),
    ("Liquid fuel",
     "Provisioning",
     "Fuel in a liquid at standard temperature (0 °C) and pressure (1 atm) including petroleum and water, that is usually used to produce energy mainly for transportation, heating, and electricity."),
    ("Solar energy",
     "Provisioning",
     "Light and heat emitted by the sun, harnessed and converted into various forms of usable energy like electricity and heating."),
    ("Wind",
     "Provisioning",
     "Form of renewable energy harnessed as the kinetic energy from wind and converted into mechanical power or electricity, primarily using wind turbines."),
    ("Geothermal energy",
     "Provisioning",
     "Form of renewable energy harnessed from the heat stored beneath the Earth's surface and used for direct heating or electricity generation."),
    ("Genetic information (biodiversity)",
     "Provisioning",
     "Genes and genetic information used for animal and plant breeding, biotechnology, to provide biodiversity, natural selection, and self-organization in ecosystems, enabling ongoing evolution and the potential for adaptation."),
    ("Biochemicals (medicine, fertilizers, pesticides)",
     "Provisioning",
     "Chemicals found in plants, fungus, etc. that are extracted or used for the provision of chemical products like medicines, nutrient supply and fertility, pest control, preservation, cleaning, fashion, and entertainment."),
    ("Fresh water",
     "Provisioning",
     "Surface or groundwater with less than 3,000 Mg/L TDS which can be purified or directly consumed by humans and non-humans as drinking water, or used for agricultural irrigation, industrial processes, among others."),
    ("Brackish water",
     "Provisioning",
     "Water as a mix of fresh water and saline water with between 3,000-10,000 Mg/L TDS generally found in estuaries."),
    ("Saline water",
     "Provisioning",
     "Saltwater or seawater with more than 10,000 Mg/L TDS."),
    ("Brine water",
     "Provisioning",
     "Saltwater with more than 35,000 Mg/L TDS; used as a preservative in meat-packing, pickling, heat-transfer media and industrial steel among others."),
    ("Reclaimed and recycled water",
     "Provisioning",
     "Treated wastewater originating from various sources like rainwater, industrial processes, or sanitary and kitchen installations, purified by removing contaminants and impurities."),
    ("Fresh air",
     "Provisioning",
     "Mixture of different gases in the Earth's atmosphere with around 78% nitrogen and 21% oxygen, clean air without harmful pollutants and healthy to breathe in."),
    ("Habitat",
     "Provisioning",
     "Shelter and protection of organisms and generally provides access to food/nutrition, place for nursery of young organisms, relevant to both permanent and transient populations."),
    ("Soil",
     "Provisioning",
     "Mixture of minerals, organic matter, water, air, and living organisms on the immediate surface of the Earth that serves as a natural medium for growth of land plants, water flow regulation, pollutant filters and buffers, nutrient cycling."),

    # --- REGULATION AND MAINTENANCE SERVICES ---
    ("Regulation of air quality",
     "Regulation and Maintenance",
     "Process of assimilating and transforming emissions or contributing chemicals to regulate the concentrations of gases in the Earth's atmosphere and reduce or eliminate toxic compounds."),
    ("Protection against harmful radiation",
     "Regulation and Maintenance",
     "Process of filtering out harmful radiation like ultraviolet light, radioactive rays, gamma rays, x-rays, and microwaves by means of shielding, distance, or through atmospheric composition."),
    ("Regulation of temperature",
     "Regulation and Maintenance",
     "Cooling or heating of short-term and long-term ambient atmospheric conditions at the micro- and mesoscale, for instance by the presence of plants, shadowing, ventilation, transpiration, or humidity, and by sequestering or emitting greenhouse gases."),
    ("Regulation of humidity and transpiration",
     "Regulation and Maintenance",
     "Process of increasing or decreasing atmospheric humidity and transpiration, for example through temperature regulation or ventilation."),
    ("Regulation of sound",
     "Regulation and Maintenance",
     "Mitigation of harmful or stressful effects of noise on humans and non-humans, supporting natural sounds that support well-being."),
    ("Water flow regulation",
     "Regulation and Maintenance",
     "Timing and magnitude of runoff, flooding, wave, or aquifer recharge related to land cover to reduce damage on water storage potential and to limit the damage of natural occurrences."),
    ("Attenuation of erosion and mass movement",
     "Regulation and Maintenance",
     "Mediation of solid mass, liquid, and gaseous flows by natural biotic or abiotic structures such as vegetative cover, rocks, root formations for soil retention and prevention of landslides."),
    ("Regulation and attenuation of seismic activity",
     "Regulation and Maintenance",
     "Process of reducing the timing, magnitude, and impact of movements and vibrations of the Earth's crust linked to earthquakes and volcanic eruptions."),
    ("Regulation of water quality",
     "Regulation and Maintenance",
     "Process of assimilation and transformation of excess and toxic compounds through plants, animals, or abiotic filtering to maintain the chemical condition of waters."),
    ("Decomposition",
     "Regulation and Maintenance",
     "Process in ecosystems that enables the breakdown and transformation of nutrients; allows the recovery of nutrients for reuse and ensures wastes are assimilated."),
    ("Pest and disease regulation",
     "Regulation and Maintenance",
     "Process of reducing the abundance of human pathogens or crop and livestock pests and diseases through complex feedback mechanisms in ecosystems."),
    ("Control of invasive species",
     "Regulation and Maintenance",
     "The reduction of biological interactions between native and non-native species that prevents or reduces negative ecological outputs."),
    ("Regulation of smell",
     "Regulation and Maintenance",
     "Reduction in the health impact of toxic or harmful odors for humans and non-humans."),
    ("Pollination and seed dispersal",
     "Regulation and Maintenance",
     "Fertilization of plants through spatial transfer of genetic material by other plants, animals, or dispersal of seeds and spores to facilitate reproduction."),
    ("Regulation of wind (including ventilation)",
     "Regulation and Maintenance",
     "Reduce or increase the speed and pattern of movement of air, for instance by the presence of plants and animals."),
    ("Fire protection and moderation",
     "Regulation and Maintenance",
     "Reduction in the incidence, intensity, or speed of spread of fire by a diversity of strategies including the presence of water, wetland, or fire belts."),
    ("Regulation of flood events",
     "Regulation and Maintenance",
     "Reduction in the incidence and intensity of inundations to prevent damage to humans, non-humans, and the environment."),
    ("Regulation of drought",
     "Regulation and Maintenance",
     "Reduction in the incidence, intensity, and duration of drought periods to ensure water conservation and efficient water usage."),
    ("Soil formation",
     "Regulation and Maintenance",
     "Formation and retention of soil associated with ensuring ongoing soil fertility through nutrient cycling and storage, rock weathering, and microbial activity."),
    ("Regulation of soil quality",
     "Regulation and Maintenance",
     "Process of assimilation and transformation of toxic or excess compounds through decomposition, breakdown, or filtering through abiotic elements to recover mobile nutrients."),
    ("Regulation of weathering processes",
     "Regulation and Maintenance",
     "Breakdown or decomposition of minerals, rocks, and soils through biological, physical, and chemical forces contributing to soil formation, nutrient cycling, and landscape evolution."),
    ("Regulation of biogeochemical cycles",
     "Regulation and Maintenance",
     "Biological processes and mechanisms like nitrogen and carbon fixation that regulate the flow and transformation of chemical elements including carbon sequestration."),
    ("Primary production",
     "Regulation and Maintenance",
     "Fixation of solar energy by plants through photosynthesis; forms the basis of the planet's food chain."),
    ("Upcycling of materials",
     "Regulation and Maintenance",
     "Process of repurposing and transforming existing materials or products into products of higher value, quality, function, or aesthetics."),
    ("Reuse of materials",
     "Regulation and Maintenance",
     "Process of using existing materials or products multiple times for the same or a different purpose extending the material or product's life."),
    ("Recycling of materials",
     "Regulation and Maintenance",
     "Industrial recycling that processes, breaks down, or remanufactures existing materials into new raw materials and new products."),

    # --- CULTURAL SERVICES ---
    ("Cultural identity, heritage and historical values",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that help humans to identify with the history or culture of where they live or come from."),
    ("Spirituality and religion",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that have spiritual or religious importance for humans."),
    ("Knowledge systems and education",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that are subject to research, teaching and skill development."),
    ("Cognitive development, psychological and physical health",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that are viewed, observed, or enjoyed in passive or active ways for mental and physical health."),
    ("Inspiration for human creative thought and work (arts)",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that provide a rich source of inspiration for art, folklore, national symbols, architecture, and advertising."),
    ("Aesthetic experience",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that are appreciated for their inherent beauty which inspires designs."),
    ("Decoration / Adornment",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems recognized for their cultural, historical, or iconic character and used as emblems, signifiers, decoration, or adornment."),
    ("Social relations and cultural diversity",
     "Cultural",
     "Natural, biotic and abiotic characteristics of ecosystems that enable interactions, intellectual activities, have symbolic or spiritual importance, and influence types of social relations."),
    ("Sense of place / Connectedness",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems that humans seek to preserve because of their non-utilitarian qualities."),
    ("Recreation, relaxation and ecotourism",
     "Cultural",
     "Biophysical characteristics or qualities of species or ecosystems or cultivated landscapes that can be used for leisure, amusement, or enjoyment."),
]


def _build_taxonomy_reference() -> str:
    """Format the full taxonomy into a prompt-ready reference block."""
    lines = []
    current_category = None
    for name, category, description in ECOSYSTEM_SERVICES_TAXONOMY:
        if category != current_category:
            current_category = category
            lines.append(f"\n## {category.upper()} SERVICES")
        lines.append(f"- **{name}**: {description}")
    return "\n".join(lines)


class EcosystemServicePromptBuilder:
    """Builds targeted prompts for extracting ecosystem services"""

    @staticmethod
    def build_ecosystem_service_extraction_prompt(text: str) -> str:
        """Build prompt for extracting ecosystem services from case study"""

        taxonomy_ref = _build_taxonomy_reference()

        return f"""Paper text:
{text}

---
You are an expert in ecosystem services analysis and sustainable building research.

Your task: Extract ALL ecosystem services that are EVIDENCED or PROVIDED by the case study / research project described in this paper.

REFERENCE TAXONOMY — use these names and categories when classifying each service:
{taxonomy_ref}

For EACH ecosystem service found, provide:
1. name: The ecosystem service name (use EXACTLY one of the names from the taxonomy above)
2. category: The category it belongs to (Provisioning / Regulation and Maintenance / Cultural)
3. confidence: A score from 0.0 to 1.0 indicating how confident you are that the paper's own case study genuinely evidences this service (1.0 = explicitly discussed with data/results, 0.5 = implied but not primary focus, 0.1 = only tangentially mentioned)
4. anchor: A SHORT EXACT PHRASE of 5-10 words copied verbatim from the paper that
   best locates where this ecosystem service is discussed. This phrase will be searched in the
   source document to retrieve the surrounding passage — it must exist exactly as
   written in the paper text above.
   RULES FOR ANCHOR:
   - Copy 5-10 consecutive words exactly as they appear in the paper text above
   - Choose words that are specific and unique to this ecosystem service mention
   - Do NOT paraphrase, summarise, or change any words
   - If you cannot identify a specific passage in the text above, set anchor to null rather than guessing

Reply with a SINGLE JSON object only. No preamble, no conversational filler.

The JSON object MUST follow this structure:
{{
    "ecosystem_services": [
        {{
            "name": "ecosystem service name from taxonomy",
            "category": "Provisioning",
            "confidence": 0.9,
            "anchor": "five to ten exact words from the paper",
        }},
        {{
            "name": "another ecosystem service name from taxonomy",
            "category": "Regulation and Maintenance",
            "confidence": 0.6,
            "anchor": null,
        }}
    ]
}}

CRITICAL RULES:
- Extract ONLY ecosystem services evidenced in THIS paper's OWN case study, experiment, or research
- Do NOT extract services cited from other literature, reviews, or background references
- Focus on sections like "Case Study", "Methods", "Results", "Discussion" — 
- IGNORE "Literature Review" and references sections
- The anchor must be a verbatim copy of words from the paper text above — do not
  invent or paraphrase
- If you cannot find a passage in the text that evidences a service, set
  anchor to null rather than guessing
- Use the EXACT service name from the taxonomy provided above
- Include the correct category for each service
- If no ecosystem services are found, return {{"ecosystem_services": []}}

JSON output:"""


class EcosystemServiceExtractor:
    """Main ecosystem service extraction orchestrator"""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.prompt_builder = EcosystemServicePromptBuilder()
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
                        "temperature": 0.1,  # Slightly higher to allow for matching services to the taxonomy
                        "num_predict": 3000,
                    }
                },
                timeout=240  # Longer timeout due to large taxonomy in prompt
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

        return {'ecosystem_services': []}

    def extract_from_text(self, text: str, verbose: bool = True) -> Dict:
        """
        Extract ecosystem services from a research paper text.

        Returns raw LLM output with 'anchor' fields. Call
        resolve_ecosystem_service_contexts() from context_resolver.py afterwards
        to replace anchors with verified context passages.

        Args:
            text: Full text of the research paper
            verbose: Print progress information

        Returns:
            Dictionary containing list of ecosystem services (with anchor fields,
            not yet resolved to context passages)
        """
        if verbose:
            logger.info("Extracting ecosystem services from text...")

        # Truncate if too long
        if len(text) > 15000:
            if verbose:
                logger.warning(f"Text truncated from {len(text)} to 15000 chars")
            text = text[:15000]

        # Build and send prompt
        prompt = self.prompt_builder.build_ecosystem_service_extraction_prompt(text)
        response = self._query_ollama(prompt)

        if verbose and not response:
            logger.warning("Empty response from Ollama")

        # Extract JSON
        raw_results = self._extract_json(response)

        # Debug output
        if not raw_results.get('ecosystem_services') and verbose:
            logger.debug(f"Raw LLM response: {response[:300]}...")

        services = raw_results.get('ecosystem_services', [])

        if verbose:
            logger.info(f"Found {len(services)} ecosystem services")
            for service in services[:5]:
                name = service.get("name", "Unnamed")
                category = service.get("category", "Unknown")
                confidence = service.get("confidence", "N/A")
                anchor = service.get("anchor")
                anchor_display = f'"{anchor[:40]}..."' if anchor else "null"
                logger.info(
                    f"  - {name} [{category}] (confidence: {confidence}) | anchor: {anchor_display}"
                )

        return {'ecosystem_services': services}

    def extract_from_file(self, filepath: str, verbose: bool = True) -> Dict:
        """
        Extract ecosystem services from a text file

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
        report_lines.append("ECOSYSTEM SERVICE EXTRACTION REPORT")
        report_lines.append("=" * 80)

        services = results.get('ecosystem_services', [])

        if not services:
            report_lines.append("\n  No ecosystem services identified.")
        else:
            report_lines.append(f"\nTotal ecosystem services found: {len(services)}\n")

            # Group by category for readability
            by_category: Dict[str, list] = {}
            for service in services:
                cat = service.get('category', 'Unknown')
                by_category.setdefault(cat, []).append(service)

            for category, cat_services in by_category.items():
                report_lines.append(f"\n--- {category.upper()} SERVICES ({len(cat_services)}) ---")

                for i, service in enumerate(cat_services, 1):
                    name = service.get("name", "Unnamed Service")
                    confidence = service.get("confidence", "N/A")
                    context = service.get("context")           # resolved passage (str|None)
                    anchor_text = service.get("anchor_text")  # original LLM anchor
                    verified = service.get("anchor_verified")
                    score = service.get("anchor_match_score")

                    report_lines.append(f"\n  Service {i}: {name}")
                    report_lines.append(f"  Confidence: {confidence}")

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

                    report_lines.append("")

        report_lines.append("=" * 80)
        return "\n".join(report_lines)


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ecosystem_service_extractor.py <input_text_file> [output_json_file]")
        print("\nExample:")
        print("  python ecosystem_service_extractor.py paper.txt ecosystem_services.json")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "ecosystem_services.json"

    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}\n")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    extractor = EcosystemServiceExtractor(model="llama3.2")

    # Step 1: extract (produces anchors)
    results = extractor.extract_from_text(raw_text, verbose=True)

    # Step 2: resolve anchors → real context passages
    from context_resolver import resolve_ecosystem_service_contexts
    results = resolve_ecosystem_service_contexts(results, raw_text)

    extractor.save_results(results, output_file)
    print(f"\n✓ Results saved to {output_file}")

    report = extractor.generate_report(results)
    print("\n" + report)

    report_file = output_file.replace(".json", "_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ Report saved to {report_file}")


if __name__ == "__main__":
    main()
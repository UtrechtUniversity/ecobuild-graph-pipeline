# ======================= #
# TODO: Implement pipeline for semantic matching of extracted desin strategies and ecosystem services onto existing vocabulary
# ======================= #

import logging
import math
from typing import Optional

import numpy as np
import requests

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

# ── Full taxonomy of design strategies ──────────────────────────────────────
DESIGN_STRATEGIES_TAXONOMY = [
    "Aeroponic systems",
    "Airdrop irrigation systems",
    "Aquaponic systems",
    "Beehives",
    "Biofilters",
    "Biomass energy collection",
    "Biophilic design",
    "Bioremediation",
    "Black/greywater treatment",
    "Blue-green roofs",
    "Blueroofs",
    "Botanical garden",
    "Burning hydrogen",
    "Cisterns",
    "Coated glass",
    "Community gardens",
    "Compact construction",
    "Companion planting",
    "Composting",
    "Cradle to cradle design",
    "Cross ventilation",
    "Delayed drainage systems",
    "Design for deconstruction",
    "Downspout strategies",
    "Drip irrigation systems",
    "Dynamic/aerodynamic architecture",
    "Ecological analogue design",
    "Edible Landscapes",
    "Elevated constructions",
    "Extensive green roof",
    "Façade cladding",
    "Fit form to function",
    "Floating contructions",
    "Fog harvesting",
    "Food forests",
    "Green corridors",
    "Green façade",
    "Green wall system",
    "Greenhouse agriculture",
    "Heat-cold storage installation",
    "Hedgerows consisting of edible plants",
    "Herb gardens",
    "High albedo materials",
    "Hydro turbines",
    "Hydrogen burning system",
    "Hydronic radiant system",
    "Hydroponic systems",
    "Increase of native plants",
    "Increase species diversity",
    "Indoor (food) gardens",
    "Infiltration systems for excess rain water",
    "Insect attracting structures",
    "Insulation material",
    "Integration of retaining walls",
    "Integration of thermal mass",
    "Intensive green roof",
    "Intercropping",
    "Interlocking panels",
    "Introduction of keystone species in property",
    "Leakage control",
    "Light filtering for temperature keeping",
    "Living Plant Constructions (Baubotanik)",
    "Location planning with little sun and wind",
    "Low albedo materials",
    "Medicinal gardens",
    "Micro livestock",
    "Microbial fuel cells",
    "Moss wall",
    "Noise reducing technologies",
    "Organic farming",
    "Orientation sound barriers",
    "Permaculture",
    "Permeable paving system",
    "Photovoltaic cells integration",
    "Photovoltaic panels",
    "Pile foundations",
    "Placed on columns or stilts",
    "Planter green wall",
    "Pocket garden/park",
    "Private gardens",
    "Rainwater Cooling System",
    "Rainwater harvesting systems",
    "Reduction of light pollution",
    "Replace equipment or install water saving devices",
    "Retaining wall system",
    "Revegetation",
    "Roof ponds",
    "Sacrificial ground floors",
    "Seismic architecture",
    "Semi-intensive green roof",
    "Separation of waste streams",
    "Smart irrigation",
    "Smart roofs",
    "Soakwells",
    "Soil filtration",
    "Soil quality monitoring",
    "Solar thermal collectors",
    "Solar water heaters",
    "Solid walls",
    "Source separation of wastewater",
    "Stack ventilation",
    "Stepping stone habitats",
    "Structural bracing strategies",
    "Sunscreens",
    "Sustainably sourced materials",
    "Texture and form based sound barriers",
    "Thermal desorption",
    "Thermal energy storage system",
    "Trellis/fence farms",
    "Urban mining",
    "Urban orchards",
    "Urban (rooftop) farming",
    "Use of biodegradable materials",
    "Use of carbon/GHG storing building materials",
    "Use of daylight",
    "Use of locally sourced materials",
    "Use of mulch",
    "Use of pre-existing vegetation",
    "Use of readily available materials",
    "Use of recycled materials",
    "Use of traditional techniques",
    "Vegetable gardens",
    "Vegetated grid pave",
    "Vegetated pergola",
    "Vegetation cover increase",
    "Vegetation for insulation",
    "Vertical farming",
    "Vertical mobile garden",
    "Waste management",
    "Water cooling systems",
    "Water efficient systems",
    "Water less systems",
    "Water source heat pump",
    "Water storage",
    "Wind barriers",
    "Wind turbines",
    "Xeriscaping",
]

# ── Embedding + cosine similarity helpers ────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Main matcher class ────────────────────────────────────────────────────────

class EntityResolutionMatcher:
    """
    Embeds vocabulary terms once at initialisation, then matches extracted
    items against them using cosine similarity via Ollama.

    Parameters
    ----------
    ollama_host : str
        Base URL of the Ollama server (e.g. "http://ollama:11434").
    embedding_model : str
        Ollama model name to use for embeddings (e.g. "nomic-embed-text").
    """

    def __init__(self, ollama_host: str, embedding_model: str):
        self.ollama_host = ollama_host.rstrip("/")
        self.embedding_model = embedding_model

        logger.info(
            f"EntityResolutionMatcher: pre-embedding vocabularies via "
            f"{embedding_model} at {ollama_host} ..."
        )

        # ── Ecosystem services ────────────────────────────────────────────
        # Embed "{name}: {description}" — richer signal than the name alone.
        self._eco_names: list[str] = []
        self._eco_categories: list[str] = []
        self._eco_embeddings: list[np.ndarray] = []

        for name, category, description in ECOSYSTEM_SERVICES_TAXONOMY:
            vocab_string = f"{name}: {description}"
            emb = self._embed(vocab_string)
            if emb is not None:
                self._eco_names.append(name)
                self._eco_categories.append(category)
                self._eco_embeddings.append(emb)

        logger.info(
            f"  Embedded {len(self._eco_embeddings)}/{len(ECOSYSTEM_SERVICES_TAXONOMY)} "
            f"ecosystem service vocabulary terms."
        )

        # ── Design strategies ─────────────────────────────────────────────
        # Only names available — embed the name string directly.
        self._ds_names: list[str] = []
        self._ds_embeddings: list[np.ndarray] = []

        for name in DESIGN_STRATEGIES_TAXONOMY:
            emb = self._embed(name)
            if emb is not None:
                self._ds_names.append(name)
                self._ds_embeddings.append(emb)

        logger.info(
            f"  Embedded {len(self._ds_embeddings)}/{len(DESIGN_STRATEGIES_TAXONOMY)} "
            f"design strategy vocabulary terms."
        )

    # ── Ollama embedding call ─────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Return a normalised embedding vector for `text`, or None on failure."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=30,
            )
            if response.status_code == 200:
                vec = np.array(response.json()["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            else:
                logger.warning(
                    f"Embedding request failed (status {response.status_code}) "
                    f"for text: '{text[:60]}'"
                )
                return None
        except Exception as e:
            logger.error(f"Embedding error for '{text[:60]}': {e}")
            return None

    # ── Query construction ────────────────────────────────────────────────────

    @staticmethod
    def _build_query(name: str, context: Optional[str]) -> str:
        """Combine extracted name and context into a single query string."""
        if context:
            return f"{name}. {context}"
        return "SKIP"

    # ── Per-item matching ─────────────────────────────────────────────────────

    def _find_best_match(
        self,
        query: str,
        vocab_embeddings: list[np.ndarray],
        vocab_names: list[str],
    ) -> tuple[str, float]:
        """Embed `query` and return (best_name, cosine_score)."""
        query_emb = self._embed(query)
        if query_emb is None or not vocab_embeddings:
            return ("", 0.0)

        scores = [_cosine_similarity(query_emb, v) for v in vocab_embeddings]
        best_idx = int(np.argmax(scores))
        return vocab_names[best_idx], round(scores[best_idx], 4)

    # ── Public batch-resolution methods ──────────────────────────────────────

    def resolve_design_strategy_matches(self, results: dict) -> dict:
        """
        Add `vocab_match` and `vocab_match_score` to each design strategy
        in `results`. Operates on all items regardless of anchor verification
        status (matching is independent of whether the context was found in
        the source). Returns `results` modified in-place.
        """
        strategies = results.get("design_strategies", [])
        if not self._ds_embeddings:
            logger.warning("Design strategy vocabulary embeddings unavailable; skipping resolution.")
            return results

        for strategy in strategies:
            name    = strategy.get("name") or ""
            context = strategy.get("context")  # may be None if anchor unverified
            query   = self._build_query(name, context)
            if query == "SKIP":
                strategy["vocab_match"]       = "No context available"
                strategy["vocab_match_score"] = 0.0
                continue
            match, score = self._find_best_match(query, self._ds_embeddings, self._ds_names)
            strategy["vocab_match"]       = match
            strategy["vocab_match_score"] = score

        logger.info(
            f"Design strategy vocab resolution complete for {len(strategies)} items."
        )
        return results

    def resolve_ecosystem_service_matches(self, results: dict) -> dict:
        """
        Add `vocab_match`, `vocab_match_score`, and `vocab_category` to each
        ecosystem service in `results`. Returns `results` modified in-place.
        """
        services = results.get("ecosystem_services", [])
        if not self._eco_embeddings:
            logger.warning("Ecosystem service vocabulary embeddings unavailable; skipping resolution.")
            return results

        for service in services:
            name    = service.get("name") or ""
            context = service.get("context")
            query   = self._build_query(name, context)
            if query == "SKIP":
                service["vocab_match"]       = "No context available"
                service["vocab_match_score"] = 0.0
                continue
            match, score = self._find_best_match(query, self._eco_embeddings, self._eco_names)
            service["vocab_match"]       = match
            service["vocab_match_score"] = score

            # Look up the category of the matched term
            try:
                idx = self._eco_names.index(match)
                service["vocab_category"] = self._eco_categories[idx]
            except ValueError:
                service["vocab_category"] = None

        logger.info(
            f"Ecosystem service vocab resolution complete for {len(services)} items."
        )
        return results
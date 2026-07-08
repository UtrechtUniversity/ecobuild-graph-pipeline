"""FastAPI wrapper around the extraction pipeline.

Runs section preprocessing, labeling, entity extraction, design-strategy
extraction, and ecosystem-service extraction for a single uploaded PDF.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from .context_resolver import (
    resolve_design_strategy_contexts,
    resolve_ecosystem_service_contexts,
    resolve_entity_contexts,
)
from .design_strategy_extractor import DesignStrategyExtractor
from .ecosystem_service_extractor import EcosystemServiceExtractor
from .entity_extractor import EntityInformationExtractor
from .entity_resolution import EntityResolutionMatcher
from .paper_labeler import LLMLabeler
from .paper_preprocessor import PaperPreprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
OUTPUT_DIR = "/app/test_papers/preprocessed"

pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = Ollama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_HOST, context_window=14000, temperature=0.11, request_timeout=180.0)
    embed_model = OllamaEmbedding(model_name=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST, request_timeout=180.0)
    pipeline["preprocessor"] = PaperPreprocessor(llm, embed_model=embed_model)
    pipeline["labeler"] = LLMLabeler(llm, fuzzy_threshold=0.8)
    pipeline["entity_extractor"] = EntityInformationExtractor(llm)
    pipeline["design_extractor"] = DesignStrategyExtractor(llm)
    pipeline["ecosystem_extractor"] = EcosystemServiceExtractor(llm)
    pipeline["resolver"] = EntityResolutionMatcher(embed_model)
    logger.info("Extraction pipeline ready (llm=%s, embed=%s)", OLLAMA_LLM_MODEL, OLLAMA_EMBEDDING_MODEL)
    yield
    pipeline.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(paper_id: int, request: Request):
    pdf_bytes = await request.body()
    if pdf_bytes[:5] != b"%PDF-":
        raise HTTPException(status_code=400, detail="Request body is not a PDF")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"paper_{paper_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    try:
        preprocess_result = pipeline["preprocessor"].preprocess_pdf(pdf_path=pdf_path, output_dir=OUTPUT_DIR)
        if "error" in preprocess_result:
            raise HTTPException(status_code=422, detail=preprocess_result["error"])

        labels: dict[str, list[dict]] = {}
        for section, preprocessed in preprocess_result["sections"].items():
            label_result = pipeline["labeler"].label(preprocessed, section_name=section)
            labels[section] = [
                decision.to_dict()
                for decision in label_result.decisions
                if decision.verdict.value in {"YES", "UNVERIFIED"}
            ]

        with open(preprocess_result["raw_text_path"], "r", encoding="utf-8") as f:
            raw_text = f.read()

        # ponytail: whole-document prompt, capped by the LLM's 14000-token context
        # window (set above). Matches labels' whole-document scope rather than the
        # abstract-only text these extractors used before being reconnected. Very
        # long papers can still get truncated by llama_index before the extractor
        # sees the tail of the text — chunk-and-merge per section if that turns
        # out to matter in practice.
        entity_result = pipeline["entity_extractor"].extract_from_text(raw_text, verbose=False, file_name=pdf_path)
        entity_result = resolve_entity_contexts(entity_result, raw_text)

        design_result = pipeline["design_extractor"].extract_from_text(raw_text, verbose=False, file_name=pdf_path)
        design_result = resolve_design_strategy_contexts(design_result, raw_text)
        design_result = pipeline["resolver"].resolve_design_strategy_matches(design_result)

        ecosystem_result = pipeline["ecosystem_extractor"].extract_from_text(raw_text, verbose=False, file_name=pdf_path)
        ecosystem_result = resolve_ecosystem_service_contexts(ecosystem_result, raw_text)
        ecosystem_result = pipeline["resolver"].resolve_ecosystem_service_matches(ecosystem_result)

        return {
            "paper_id": paper_id,
            "labels": labels,
            "entities": entity_result.get("entities", []),
            "design_strategies": design_result.get("design_strategies", []),
            "ecosystem_services": ecosystem_result.get("ecosystem_services", []),
        }
    finally:
        os.remove(pdf_path)

import re
import json
import logging
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

class LlamaIndexInterface:
    """Shared LLM interface for all extractors."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def query(self, prompt: str) -> str:
        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return '{}'

    def extract_json(self, response: str) -> dict:
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
        return {}
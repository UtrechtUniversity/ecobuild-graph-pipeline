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
        if not json_match:
            logger.warning("No JSON object found in LLM response: %r", response[:300])
            return {}
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            # Logged unconditionally (not gated on caller verbosity) since a parse
            # failure here silently degrades to an empty result downstream —
            # this is the only place that still has the raw text to explain why.
            logger.warning("JSON parse error (%s) in LLM response: %r", e, json_match.group()[:300])
            return {}

    def query_json(self, prompt: str, retries: int = 2) -> dict:
        """query() + extract_json(), retrying on empty/unparseable output.

        # ponytail: the remote Ollama proxy sometimes cuts a response stream short
        # (same wall-clock time whether it returns 0 or 900 chars, so it looks like
        # a fixed-duration cutoff rather than the model actually finishing) — a
        # couple of retries recovers most of these. If it keeps happening, look at
        # the proxy's read/idle timeout rather than raising this further.
        """
        for attempt in range(retries + 1):
            result = self.extract_json(self.query(prompt))
            if result:
                return result
            if attempt < retries:
                logger.warning("Empty/unparseable LLM response (attempt %d/%d) — retrying.", attempt + 1, retries + 1)
        return {}


def _demo() -> None:
    """ponytail: smallest check that extract_json degrades safely instead of raising."""
    class _Stub:
        def complete(self, *_):
            raise NotImplementedError

    iface = LlamaIndexInterface(_Stub())
    assert iface.extract_json('{"a": 1}') == {"a": 1}
    assert iface.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert iface.extract_json('{"a": 1,}') == {}          # trailing comma -> invalid JSON
    assert iface.extract_json('no json here') == {}
    print("llama_index_interface._demo: ok")


if __name__ == "__main__":
    _demo()
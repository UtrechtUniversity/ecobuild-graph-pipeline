import requests
import time
import os

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/") + "/api/pull"
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")

def pull_model(name: str):
    print(f"Requesting model '{name}' from Ollama...")
    while True:
        try:
            response = requests.post(OLLAMA_URL, json={"name": name})
            response.raise_for_status()
            print(f"Model '{name}' pulled successfully.")
            break
        except Exception as e:
            print(f"Error pulling model '{name}': {e}")
            time.sleep(2)

if __name__ == "__main__":
    pull_model(LLM_MODEL)
    pull_model(EMBED_MODEL)


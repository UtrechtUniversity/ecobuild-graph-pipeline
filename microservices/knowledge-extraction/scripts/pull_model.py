import requests
import time

import os

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/") + "/api/pull"
MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "llama3")

print(f"Requesting model '{MODEL_NAME}' from Ollama...")

while True:
    try:
        response = requests.post(OLLAMA_URL, json={"name": MODEL_NAME})
        response.raise_for_status()
        print("Model pulled successfully.")
        break
    except Exception as e:
        print(f"Error pulling model: {e}")
        time.sleep(2)


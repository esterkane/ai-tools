import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "zephyr")

def get_answer_from_llm(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for internal KB support."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        return response.json()["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama Error] {e}"

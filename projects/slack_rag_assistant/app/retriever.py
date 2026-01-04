import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "zephyr"

def handle_question(question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()

        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    answer += data.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"[Ollama JSON error] {e}")

        return answer.strip()

    except Exception as e:
        print(f"[Ollama Error] {e}")
        return "Sorry, something went wrong when generating the answer."

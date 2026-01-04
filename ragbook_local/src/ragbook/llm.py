from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class LLM:
    backend: str
    llama: Optional[object] = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    @classmethod
    def from_config(
        cls,
        backend: str,
        *,
        model_path: str,
        n_ctx: int,
        temperature: float,
        max_tokens: int,
        ollama_url: str,
        ollama_model: str,
    ) -> "LLM":
        backend = backend.lower().strip()
        inst = cls(backend=backend, ollama_url=ollama_url, ollama_model=ollama_model)
        inst._max_tokens = int(max_tokens)
        inst._temperature = float(temperature)

        if backend == "llama_cpp":
            try:
                from llama_cpp import Llama
            except Exception as e:
                raise RuntimeError(
                    "llama-cpp-python nicht installiert. Installiere es oder nutze backend=ollama."
                ) from e
            inst.llama = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
            return inst

        if backend == "ollama":
            return inst

        raise ValueError(f"Unbekanntes LLM backend: {backend}")

    def generate(self, prompt: str) -> str:
        if self.backend == "llama_cpp":
            out = self.llama(
                prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stop=["</s>"],
            )
            return out["choices"][0]["text"].strip()

        if self.backend == "ollama":
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            }
            with httpx.Client(timeout=120) as client:
                r = client.post(f"{self.ollama_url}/api/generate", json=payload)
                r.raise_for_status()
                return (r.json().get("response") or "").strip()

        raise RuntimeError("Backend nicht initialisiert.")

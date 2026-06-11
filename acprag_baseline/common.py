"""Shared helpers for the ACP-RAG baseline scripts.

The original ACP-RAG_Pipeline scripts call locally-hosted HF chat models
(AutoModelForCausalLM.generate). We instead reuse this project's existing
DashScope-compatible OpenAI client (see core/llm.py, config.yaml) so the
pipeline runs on Alvis without needing extra local checkpoints. Unlike
core.llm.call_llm, chat_llm() does not force JSON-object output, since the
filter/generation prompts here expect free-text responses.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.llm import client, config  # noqa: E402


def chat_llm(prompt, model=None, temperature=0.0):
    """Call the configured chat LLM with a plain-text prompt and return its reply."""
    if model is None:
        model = config["models"]["llm_model"]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

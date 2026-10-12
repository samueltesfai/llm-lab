"""Model-level evaluation suite for Ollama-hosted models.

Run with:  uv run python -m llm_eval --model qwen3.5:9b
"""

CATEGORIES = (
    "long_context",
    "instruction_following",
    "multi_turn",
    "corrections",
    "grounded_qa",
    "programming",
)

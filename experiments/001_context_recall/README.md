# Experiment 001 — Context Recall

## Question
How reliably can Qwen3.5-9B retrieve a specific fact from
a long synthetic conversation?

## Method
Generate a deterministic conversation with a known fact,
submit the history to Ollama, and evaluate the answer.

## Variables
- Context window size
- Conversation length
- Position of the target fact

## Measurements
- Recall correctness
- Prompt token count
- Wall-clock latency
- Output token count

## Run
python3 experiments/001_context_recall/run.py --ctx 16384 --notes 90

## Findings
Not yet evaluated.

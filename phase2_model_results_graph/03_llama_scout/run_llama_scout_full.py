"""
Domain 2 (Graphs) — FULL Llama 4 Scout query run (300 graphs x 8 properties).

Experiment 1, Phase 2 (model queries) for Llama 4 Scout, a Primary model.

Per Experiment 1 protocol (serialization_experiment_1.pdf):
  - 300 graphs x 8 properties = 2,400 queries (uncertified chromatic_number
    pairs excluded)
  - Llama 4 Scout via OpenRouter (Table 1)
  - No thinking mode exists for this model ("n/a (native)" in the PDF's
    per-model table) -- nothing is sent to disable it.
  - JSON mode ON by default: forces response_format={"type":"json_object"} so
    the model answers directly instead of narrating, which otherwise risks
    overflowing the 512-token cap with no parseable answer.
  - temperature 0, max_tokens 512, no system prompt, zero-shot

Key required: OPENROUTER_API_KEY.

All output stays inside this folder:
  llama_scout_results.jsonl / llama_scout_results.json / logs/llama_scout_<timestamp>.log

NOTHING runs on import. Execute explicitly:
  python run_llama_scout_full.py                 # full run, concurrent
  python run_llama_scout_full.py --fail-fast     # sequential, stop at first failure
  python run_llama_scout_full.py --dry-run       # build prompts, no API calls
  python run_llama_scout_full.py --limit 5       # smoke test: first 5 graphs

Shared logic lives in ../_common/graph_harness.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))

from graph_harness import ModelConfig, main  # noqa: E402

CONFIG = ModelConfig(
    key="llama_scout",
    label="LLAMA-SCOUT",
    default_provider="openrouter",
    default_model="meta-llama/llama-4-scout",
    default_price_in=0.10,
    default_price_out=0.30,
    thinking_mode="none",
    supports_json_mode=True,
    default_json_mode="on",
    default_subset="none",
    model_help="Llama 4 Scout via OpenRouter (PDF Table 1).",
)

if __name__ == "__main__":
    main(CONFIG, HERE)

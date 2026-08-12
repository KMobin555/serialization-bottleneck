"""
Domain 2 (Graphs) — FULL GPT-4.1 Mini query run (300 graphs x 8 properties).

Experiment 1, Phase 2 (model queries) for GPT-4.1 Mini, the closed-source
model. Full 300-graph coverage.

Per Experiment 1 protocol (serialization_experiment_1.pdf):
  - 300 graphs x 8 properties = 2,400 queries (uncertified chromatic_number
    pairs excluded)
  - No thinking mode exists for this model ("n/a (native)") -- nothing is
    sent to disable it.
  - JSON mode ON by default: forces response_format={"type":"json_object"}.
  - temperature 0, max_tokens 512, no system prompt, zero-shot
  - Provider: OpenAI (api.openai.com), key OPENAI_API_KEY

All output stays inside this folder:
  gpt_results.jsonl / gpt_results.json / logs/gpt_<timestamp>.log

NOTHING runs on import. Execute explicitly:
  python run_gpt_full.py                 # full run, concurrent
  python run_gpt_full.py --fail-fast     # sequential, stop at first failure
  python run_gpt_full.py --dry-run       # build prompts, no API calls
  python run_gpt_full.py --limit 5       # smoke test: first 5 graphs

Shared logic lives in ../_common/graph_harness.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))

from graph_harness import ModelConfig, main  # noqa: E402

CONFIG = ModelConfig(
    key="gpt",
    label="GPT",
    default_provider="openai",
    default_model="gpt-4.1-mini",
    default_price_in=0.40,
    default_price_out=1.60,
    thinking_mode="none",
    supports_json_mode=True,
    default_json_mode="on",
    default_subset="none",
    model_help="GPT-4.1 Mini via OpenAI (PDF Table 1).",
)

if __name__ == "__main__":
    main(CONFIG, HERE)

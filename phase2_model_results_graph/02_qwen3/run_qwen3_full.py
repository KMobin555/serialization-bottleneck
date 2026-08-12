"""
Domain 2 (Graphs) — FULL Qwen3-32B query run (300 graphs x 8 properties).

Experiment 1, Phase 2 (model queries) for Qwen3-32B, a Primary model. Same
harness as the validated V4-Flash run; only the provider/model/non-thinking
mechanism differ.

Per Experiment 1 protocol (serialization_experiment_1.pdf):
  - 300 graphs x 8 properties = 2,400 queries (uncertified chromatic_number
    pairs excluded)
  - Qwen3-32B via OpenRouter (Table 1: Qwen3-32B = API/OpenRouter)
  - NON-thinking mode. OpenRouter's unified reasoning param is not reliably
    honored by every provider Qwen routes through, so this also appends
    " /no_think" to the prompt -- see
    ../../phase2_model_results/README.md §3 "The one prompt exception" for
    why the Geometry domain's Qwen3 run does the same thing.
  - temperature 0, max_tokens 512, no system prompt, zero-shot

Key required: OPENROUTER_API_KEY (NOT the DeepSeek key).

All output stays inside this folder:
  qwen3_results.jsonl / qwen3_results.json / logs/qwen3_<timestamp>.log

NOTHING runs on import. Execute explicitly:
  python run_qwen3_full.py                 # full run, concurrent
  python run_qwen3_full.py --fail-fast     # sequential, stop at first failure
  python run_qwen3_full.py --dry-run       # build prompts, no API calls
  python run_qwen3_full.py --limit 5       # smoke test: first 5 graphs

Shared logic lives in ../_common/graph_harness.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))

from graph_harness import ModelConfig, main  # noqa: E402

CONFIG = ModelConfig(
    key="qwen3",
    label="QWEN3",
    default_provider="openrouter",
    default_model="qwen/qwen3-32b",
    default_price_in=0.08,
    default_price_out=0.28,
    thinking_mode="openrouter_qwen",
    supports_json_mode=False,
    default_subset="none",
    model_help="Qwen3-32B via OpenRouter (PDF Table 1: Qwen3-32B = API/OpenRouter).",
)

if __name__ == "__main__":
    main(CONFIG, HERE)

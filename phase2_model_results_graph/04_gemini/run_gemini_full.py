"""
Domain 2 (Graphs) — FULL Gemini 2.5 Flash-Lite query run (300 graphs x 8 properties).

Experiment 1, Phase 2 (model queries) for Gemini 2.5 Flash-Lite, the "small
baseline" (non-thinking) model. Full 300-graph coverage.

Per Experiment 1 protocol (serialization_experiment_1.pdf):
  - 300 graphs x 8 properties = 2,400 queries (uncertified chromatic_number
    pairs excluded)
  - NON-thinking: Gemini's OpenAI-compat endpoint disables it via
    reasoning_effort="none".
  - JSON mode ON by default: reasoning_effort=none alone does not stop
    Gemini writing content-level chain-of-thought, which can overflow the
    512-token cap before an answer appears -- JSON mode suppresses that.
  - temperature 0, max_tokens 512, no system prompt, zero-shot
  - Provider: Google (generativelanguage.googleapis.com OpenAI-compat
    endpoint), key GEMINI_API_KEY

All output stays inside this folder:
  gemini_results.jsonl / gemini_results.json / logs/gemini_<timestamp>.log

NOTHING runs on import. Execute explicitly:
  python run_gemini_full.py                 # full run, concurrent
  python run_gemini_full.py --fail-fast     # sequential, stop at first failure
  python run_gemini_full.py --dry-run       # build prompts, no API calls
  python run_gemini_full.py --limit 5       # smoke test: first 5 graphs

Shared logic lives in ../_common/graph_harness.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))

from graph_harness import ModelConfig, main  # noqa: E402

CONFIG = ModelConfig(
    key="gemini",
    label="GEMINI",
    default_provider="google",
    default_model="gemini-2.5-flash-lite",
    default_price_in=0.10,
    default_price_out=0.40,
    thinking_mode="google",
    supports_json_mode=True,
    default_json_mode="on",
    default_subset="none",
    model_help="Gemini 2.5 Flash-Lite via Google API (PDF Table 1).",
)

if __name__ == "__main__":
    main(CONFIG, HERE)

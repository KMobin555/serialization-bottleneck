"""
Domain 2 (Graphs) — FULL DeepSeek-V4-Flash query run (300 graphs x 8 properties).

This is Experiment 1, Phase 2 (model queries) for the V4-Flash debug-first
model — same role as in the Geometry domain (Table 1: "Primary; debug first").
Run this one first; it is the harness-validation run before the other five.

Per Experiment 1 protocol (serialization_experiment_1.pdf):
  - 300 graphs x 8 properties = 2,400 queries (chromatic_number pairs whose
    ground truth is uncertified are excluded -- see Phase 1 README)
  - DeepSeek-V4-Flash, NON-thinking mode -> extra_body={"thinking": {"type": "disabled"}}
  - temperature 0, max_tokens 512, no system prompt, zero-shot
  - retry: exponential backoff, transient errors only

All output stays inside this folder:
  v4flash_results.jsonl  -- append-only, resume-safe, one record/query
  v4flash_results.json   -- JSON array export (latest record per pair)
  logs/v4flash_<timestamp>.log

Resume: re-running skips already-logged (object_id, property) pairs, so a
crash loses nothing. Fail-fast (--fail-fast) stops at the first error/parse
failure.

NOTHING runs on import. Execute explicitly:
  python run_v4flash_full.py                 # full run, concurrent
  python run_v4flash_full.py --fail-fast     # sequential, stop at first failure
  python run_v4flash_full.py --dry-run       # build prompts, no API calls
  python run_v4flash_full.py --limit 5       # smoke test: first 5 graphs

Shared logic lives in ../_common/graph_harness.py -- see that module's
docstring for why (one harness, six ~40-line config blocks, not six
~700-line near-duplicates).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))

from graph_harness import ModelConfig, main  # noqa: E402

CONFIG = ModelConfig(
    key="v4flash",
    label="V4-FLASH",
    default_provider="deepseek",
    default_model="deepseek-v4-flash",
    default_price_in=0.14,
    default_price_out=0.28,
    thinking_mode="deepseek",
    supports_json_mode=False,
    default_subset="none",
    model_help="DeepSeek-direct: deepseek-v4-flash (default).",
)

if __name__ == "__main__":
    main(CONFIG, HERE)

"""
Domain 2 (Graphs) — DeepSeek-V4-Pro NON-THINKING query run (20% subsample x 8 = 480).

Companion to a would-be thinking-mode V4-Pro run, matching the Geometry
domain's own 06_v4pro_nonthinking precedent
(../../phase2_model_results/06_v4pro_nonthinking/): same model
(deepseek-v4-pro), thinking mode DISABLED, reusing the non-thinking inference
parameters the PDF defines for the other five models.

This is OFF the PDF protocol (the PDF specifies V4-Pro in thinking mode,
Section 2). It is committed anyway, mirroring the Geometry domain's
documented deviation, so the two domains stay comparable model-for-model.

  - thinking DISABLED -> extra_body={"thinking": {"type": "disabled"}}
  - max_tokens 512 (non-thinking cap)
  - temperature 0, no system prompt, zero-shot
  - 20% stratified subsample (tier x family), seed 42 -- see build_subsample.py
  - Provider: DeepSeek direct (api.deepseek.com), key DEEPSEEK_API_KEY

Check the subsample first: `python build_subsample.py --verify`. The
subsample file is generated once and committed; the runner just reads its
object_ids.

All output stays inside this folder:
  v4pro_nonthinking_results.jsonl / v4pro_nonthinking_results.json /
  logs/v4pro_nonthinking_<timestamp>.log

NOTHING runs on import. Execute explicitly:
  python run_v4pro_nonthinking_full.py                 # full run, concurrent
  python run_v4pro_nonthinking_full.py --fail-fast     # sequential, stop at first
  python run_v4pro_nonthinking_full.py --dry-run       # build prompts, no API calls
  python run_v4pro_nonthinking_full.py --limit 5       # smoke test: first 5 graphs

Shared logic lives in ../_common/graph_harness.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))

from graph_harness import ModelConfig, main  # noqa: E402

CONFIG = ModelConfig(
    key="v4pro_nonthinking",
    label="V4-PRO-NONTHINKING",
    default_provider="deepseek",
    default_model="deepseek-v4-pro",
    default_price_in=0.435,
    default_price_out=0.87,
    thinking_mode="deepseek",
    supports_json_mode=False,
    default_subset="subsample_v4pro_nonthinking.json",
    default_max_tokens=512,
    model_help="DeepSeek-direct: deepseek-v4-pro (default).",
)

if __name__ == "__main__":
    main(CONFIG, HERE)

# Phase 2 — Domain 2 (Graphs) Model Query Runs

Documentation for the shared harness and the six model configurations that
use it.

Phase 1 ([`../phase1_dataset_graph/`](../phase1_dataset_graph/README.md))
produced the 300-graph dataset. Phase 2 asks each model 8 questions per
graph, scores the answers against ground truth, and writes one record per
query — same contract as
[`../phase2_model_results/`](../phase2_model_results/README.md) (Geometry),
adapted to this domain's serialization and property set.

**Structural difference from the Geometry domain's Phase 2:** this domain
uses one shared harness module (`_common/graph_harness.py`, ~500 lines) plus
six ~50-line config blocks, instead of six ~700-line near-duplicate scripts.
The Geometry README's own porting section says why:

> "If you port, fix the duplication first. Extract the shared harness into
> one module and reduce each runner to a config block... That is roughly 60
> lines per model instead of 700, and it makes [schema drift across runners]
> structurally impossible."

This domain is a fresh build, not a port of already-duplicated code, so
there was no reason to introduce the duplication in the first place.
Everything documented below about *behavior* (resume, retry, parsing,
metrics, record schema) is intentionally identical to the Geometry domain's
harness — only the domain-specific pieces (prompt, properties, serialization
field name) differ.

**No API calls have been made against this domain yet.** The six folders
below contain runnable scripts and a verified-by-dry-run harness; actually
querying the six models needs your API keys in `.env` (root README §"API
keys") and is a paid operation — nothing here does that automatically.

---

## 1. Folder layout

| Folder | Model | Coverage | Queries |
|--------|-------|----------|---------|
| `01_v4flash/` | DeepSeek-V4-Flash | full 300 | 2,400 |
| `02_qwen3/` | Qwen3-32B | full 300 | 2,400 |
| `03_llama_scout/` | Llama 4 Scout | full 300 | 2,400 |
| `04_gemini/` | Gemini 2.5 Flash-Lite | full 300 | 2,400 |
| `05_gpt/` | GPT-4.1 Mini | full 300 | 2,400 |
| `06_v4pro_nonthinking/` | DeepSeek-V4-Pro, thinking OFF | 20% subsample (60) | 480 |
| `_common/` | *(not a model)* | — | shared harness module |

2,400 = 300 graphs × 8 properties, *before* excluding any graph whose
`chromatic_number` is uncertified (see §6 below) — the committed Phase-1
dataset has 0 such exclusions, so all six models query exactly the counts
above. 480 = 60 × 8, same accounting.

Each model folder holds only its runner script (plus, for `06`, the
subsample file and its builder). Running a model writes into its own folder:

```
run_<model>_full.py            the runner (imports _common/graph_harness.py)
<model>_results.jsonl          append-only, one line per query
<model>_results.json           JSON array, latest record per (object_id, property)
logs/<model>_<timestamp>.log   per-run log
```

None of these output files exist yet in this repository — they are created
on first run.

---

## 2. The shared harness (`_common/graph_harness.py`)

Same eight-stage pipeline as the Geometry domain:

```
load .env  →  load + validate dataset  →  apply subset (if any)
    →  read JSONL, compute already-done pairs  →  build task list
    →  async fan-out, semaphore-capped  →  call model, retry on transient errors
    →  parse answer  →  score against ground truth  →  append JSONL
    →  export JSON  →  print cost summary
```

### Dataset loading and validation

`load_dataset()` reads the Phase-1 dataset, warns if the record count isn't
300, raises on a duplicate `object_id`, and raises if any record is missing
`object_id`, `tier`, `family`, `num_nodes`, `num_edges`, `edge_list`,
`properties`, or `metadata`. The default `--dataset` path
(`../../phase1_dataset_graph/graph_exp1_dataset.json`) resolves correctly
out of the box — unlike the Geometry domain's known-gap #1 (a stale default
pointing at a folder that doesn't exist in this repo), there was no reason
to reproduce that here.

### Excluding uncertified chromatic numbers

Per the PDF (`serialization_experiment_1.pdf`, Section 4, "Chromatic number
computation"): *"If no certificate can be obtained, exclude chromatic number
from evaluation for that graph and record the exclusion. Do not use
approximate values."* `build_tasks()` implements this directly: any
`(object_id, "chromatic_number")` pair where
`metadata.chromatic_number_certified` is `false` is never queried at all,
for any model. On the committed dataset this excludes 0 pairs (Phase 1's
run closed every graph's chromatic number via clique certificate or
time-boxed backtracking) — but the check stays in the harness in case the
dataset is regenerated with a different seed or time budget.

### Resume, subsetting, concurrency, retry, logging, cost accounting

Byte-for-byte the same mechanics as the Geometry domain — see
[`../phase2_model_results/README.md`](../phase2_model_results/README.md) §2
for the full explanation of: JSONL-as-source-of-truth resume,
`--retry-parse-failures`, `--force`, `apply_subset` for `06`'s 20%
subsample, `asyncio.Semaphore`-capped concurrency, the two retry policies
(this domain's harness always uses the more selective one — retry only
`APITimeoutError` / `RateLimitError` / `APIConnectionError` /
`InternalServerError`, not every `Exception` — since there is no
first-runner-predates-the-fix history to preserve here), `--fail-fast`
sequential debugging mode, and the DEBUG-file/INFO-console logging split.

---

## 3. The prompt

One template, all 8 properties, all six models:

```
You are given an undirected graph as an edge list.

Graph:
{edge_list}

Question:
{question}

Respond with only a JSON object in this format:
{"answer": <your answer>}
```

Zero-shot. No system prompt. No few-shot examples. No chain-of-thought
instruction. Single user message. `{edge_list}` is the Phase-1 record's
`edge_list` field verbatim (the `GRAPH (n=.., m=..):` header plus sorted
edge lines).

### The 8 questions

| Property | Question text |
|----------|----------------|
| `degree_of_node_0` | What is the degree of node 0 in this graph? |
| `edge_count` | How many edges does this graph have? |
| `triangle_count` | How many triangles are in this graph? |
| `is_bipartite` | Is this graph bipartite? Answer true or false. |
| `is_planar` | Is this graph planar? Answer true or false. |
| `diameter` | What is the diameter of this graph? |
| `chromatic_number` | What is the chromatic number of this graph? |
| `avg_clustering` | What is the average clustering coefficient of this graph? |

### The one prompt exception (same exception as Geometry, same model)

`02_qwen3` appends `" /no_think"` to the prompt when thinking is disabled.
Qwen3's chat template honors that token regardless of which upstream
provider OpenRouter routes to, whereas the API-level reasoning flag is not
reliably respected — identical reasoning to the Geometry domain's Qwen3
runner. Every other model's stored prompt ends with
`...{"answer": <your answer>}` untouched.

---

## 4. Per-model configuration

All shared settings: **temperature 0, max_tokens 512, concurrency 5,
thinking disabled, no system prompt, zero-shot, all 8 properties** — except
where the table says otherwise. Same models, providers, and prices as the
Geometry domain (Table 1 in the PDF is domain-agnostic).

| | 01 V4-Flash | 02 Qwen3-32B | 03 Llama 4 Scout | 04 Gemini 2.5 Flash-Lite | 05 GPT-4.1 Mini | 06 V4-Pro (no-think) |
|---|---|---|---|---|---|---|
| model id | `deepseek-v4-flash` | `qwen/qwen3-32b` | `meta-llama/llama-4-scout` | `gemini-2.5-flash-lite` | `gpt-4.1-mini` | `deepseek-v4-pro` |
| provider | deepseek | openrouter | openrouter | google | openai | deepseek |
| API key env | `DEEPSEEK_API_KEY` | `OPENROUTER_API_KEY` | `OPENROUTER_API_KEY` | `GEMINI_API_KEY` | `OPENAI_API_KEY` | `DEEPSEEK_API_KEY` |
| thinking | disabled | disabled | n/a (native) | disabled | n/a (native) | disabled |
| disable mechanism | `extra_body={"thinking":{"type":"disabled"}}` | `extra_body={"reasoning":{"enabled":false}}` **+** `/no_think` in prompt | none needed | `reasoning_effort="none"` | none needed | `extra_body={"thinking":{"type":"disabled"}}` |
| JSON mode | off | off | **on** | **on** | **on** | off |
| coverage | 300 | 300 | 300 | 300 | 300 | 60 (20%) |
| price in / out ($/1M) | 0.14 / 0.28 | 0.08 / 0.28 | 0.10 / 0.30 | 0.10 / 0.40 | 0.40 / 1.60 | 0.435 / 0.87 |

Each `ModelConfig` in the six runner files is the literal source of this
table — see any `run_<model>_full.py` for the exact values.

**06 is off-protocol, mirroring the Geometry domain's own documented
deviation.** The PDF specifies DeepSeek-V4-Pro in *thinking* mode at 20%
coverage; the committed Geometry run instead used a non-thinking companion
run at the same coverage (`../phase2_model_results/06_v4pro_nonthinking/`,
whose docstring says so explicitly). This domain's `06` reproduces that same
deviation rather than the literal PDF spec, so the two domains stay
model-for-model comparable. If a true thinking-mode V4-Pro run is wanted
later, it needs `max_tokens` raised well past 512 (thinking traces can be
long — the PDF suggests 2,048) and the provider's default thinking
configuration left enabled.

---

## 5. Running the scripts

### Prerequisites

Same as the root README: `source .venv/bin/activate`, `pip install -r
requirements.txt` from the repo root (already covers this domain —
`networkx` was added for Phase 1), and the API key for whichever provider
you're about to run in `.env`.

### Per-model commands

Same three-step pattern as Geometry: dry run (free), a small `--fail-fast`
smoke test, then the full run. Run each from inside its own folder — every
run is resumable, so re-issuing the same command after an interruption
continues where it stopped.

```bash
cd phase2_model_results_graph/01_v4flash
python run_v4flash_full.py --dry-run                 # inspect prompts, no API calls
python run_v4flash_full.py --limit 5 --fail-fast     # 5 graphs x 8 props = 40 queries
python run_v4flash_full.py                            # full run, resumable
```

Repeat for `02_qwen3`, `03_llama_scout`, `04_gemini`, `05_gpt` — same three
commands, each from its own folder, each needing its own provider's key.

**06 — DeepSeek-V4-Pro, non-thinking** (needs `DEEPSEEK_API_KEY`, 480
queries). Check the subsample first — it's already committed
(`subsample_v4pro_nonthinking.json`), `build_subsample.py` is only needed to
verify it or rebuild it in a fresh checkout:

```bash
cd phase2_model_results_graph/06_v4pro_nonthinking
python build_subsample.py --verify      # confirms the committed 60 ids regenerate
python run_v4pro_nonthinking_full.py --dry-run
python run_v4pro_nonthinking_full.py --limit 5 --fail-fast
python run_v4pro_nonthinking_full.py
```

`--subset` defaults to `subsample_v4pro_nonthinking.json` in that folder
automatically.

### Useful variations (identical flags across all six)

```bash
# Re-run one property only, e.g. after a parser fix
python run_gpt_full.py --properties diameter --force

# Re-query only pairs that failed to parse
python run_gpt_full.py --retry-parse-failures

# Gentler on rate limits
python run_qwen3_full.py --concurrency 2
```

### Flags

Identical across all six runners (the shared `build_arg_parser`), except
`--json-mode` which only exists for models where `supports_json_mode=True`
in their config (03, 04, 05):

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset` | `../../phase1_dataset_graph/graph_exp1_dataset.json` | resolves correctly by default |
| `--subset` | `none` (01–05) / subsample file (06) | restrict to listed object_ids |
| `--jsonl-output` | `<model>_results.jsonl` in this folder | append-only log |
| `--json-output` | `<model>_results.json` in this folder | deduplicated export |
| `--log-dir` | `logs/` | per-run log directory |
| `--provider`, `--model` | per model | see §4 |
| `--temperature` | `0.0` | |
| `--max-tokens` | `512` | completion cap |
| `--concurrency` | `5` | in-flight query cap |
| `--thinking` | `disabled` | thinking mode toggle |
| `--json-mode` | `on` (03, 04, 05 only) | `response_format` JSON object mode |
| `--properties` | all 8 | re-run a subset |
| `--fail-fast` | off | sequential, stop at first failure, dump context |
| `--price-in` / `--price-out` | per model | $/1M tokens for the cost summary |
| `--limit N` | none | first N graphs — smoke test |
| `--force` | off | re-run even already-logged pairs |
| `--retry-parse-failures` | off | re-query pairs logged with `parse_success: false` |
| `--dry-run` | off | build prompts and task list, print 3, no API calls |

Nothing executes on import — every runner is guarded by
`if __name__ == "__main__":`. Verified: all six `--dry-run` correctly (no
API key needed), all six `--limit N` correctly slice the task list, and a
run without an API key fails immediately with `<KEY_NAME> not set. Add it to
.env.` before any network call.

---

## 6. Answer parsing

Identical three-tier strategy to the Geometry domain's `parse_answer()`:

1. Parse the whole response as JSON, read `["answer"]`.
2. Regex-scan for the first `{...}` block, parse each match, read
   `["answer"]`.
3. Fall back to normalizing the raw string directly.

`normalize_answer()` branches, per property:

| Property group | Accepts | Rejects |
|-----------------|---------|---------|
| `degree_of_node_0`, `edge_count`, `triangle_count`, `diameter`, `chromatic_number` | int, integer-valued float, first number in a string | booleans, non-integer floats |
| `avg_clustering` | int/float, first number in a string (commas stripped) | booleans, non-finite values |
| `is_bipartite`, `is_planar` | bool, or `true`/`yes`/`false`/`no` as a word in a string | anything else |

Booleans are rejected everywhere a number is expected, same reasoning as
Geometry: Python treats `True` as `1`, which would otherwise silently score
a `true` answer as a numeric one. There is no bbox/centroid-style
list-valued property or cw/ccw-style categorical property in this domain —
every property here is a single int, float, or bool, so `normalize_answer`
has three branches instead of Geometry's five.

**One domain-specific parser fix, found by testing against hand-crafted
truncated/messy outputs (PDF execution checklist item 4).** The question
text for `degree_of_node_0` names the node by number: *"What is the degree
of node 0 in this graph?"* If steps 1–2 both fail to find valid JSON and the
raw-string fallback (step 3) kicks in, a prose answer like *"the degree of
node 0 is 4"* would otherwise have its first-number scan collide with the
"0" in "node 0" — silently mis-scoring a correct answer of 4 as 0, with no
parse-failure flag to catch it. `normalize_answer` strips a leading
`node ?#?_?0` match before applying the numeric regex, only for this one
property. Verified against a hand-crafted case set (clean JSON, JSON
prefixed by chatter, no JSON, JSON truncated mid-value, JSON truncated
mid-key, and the `degree_of_node_0` collision case specifically) before and
after the fix.

Failure to parse sets `parse_success: false`, `parsed_answer: null`, and
`correct`/`correct_1pct` etc. to `false` — never dropped, per the same
reasoning as Geometry: an unparseable answer is *incorrect*, not missing.

---

## 7. Result record shape

```json
{
  "object_id": "graph_simple_erdos_renyi_001",
  "tier": "simple",
  "family": "erdos_renyi",
  "num_nodes": 13,
  "num_edges": 19,
  "property": "triangle_count",
  "property_locality": "global",
  "ground_truth": 4,
  "prompt": "You are given an undirected graph as an edge list....",
  "raw_model_output": "{\"answer\": 4}",
  "parsed_answer": 4,
  "parse_success": true,
  "finish_reason": "stop",
  "failure_type": null,
  "model": "deepseek-v4-flash",
  "provider": "deepseek",
  "temperature": 0.0,
  "prompt_tokens": 210,
  "completion_tokens": 6,
  "timestamp": "2026-08-12T07:26:00+00:00"
}
```

The first five fields are copied from the Phase-1 record, so results can be
sliced by tier/family/complexity without re-joining against the dataset.

`property_locality` is `"local"` for `degree_of_node_0` and `edge_count` —
both readable from a bounded scan of the edge-list string (count lines
containing `0` as an endpoint; count/read the header) — and `"global"` for
the other six, which require reasoning over the whole graph structure. This
is the local-vs-global cut the experiment is built to test, same axis as
Geometry's `vertex_count`/`bbox` vs. the other seven.

Unlike Geometry, **every record in this domain carries `provider`,
`finish_reason`, and `failure_type`** — there is no first-runner-without-
these-fields history to reproduce, since this harness was built once, after
those fields were already part of the design (Geometry's `01_v4flash`
predates them; this domain's `01_v4flash` does not).

### Metric fields — depend on the property

| Property group | Fields added |
|-----------------|----------------|
| `avg_clustering` (1) | `absolute_error`, `relative_error`, `correct_1pct`, `correct_5pct`, `correct_10pct` |
| `degree_of_node_0`, `edge_count`, `triangle_count`, `diameter`, `chromatic_number` (5) | `absolute_error`, `correct` |
| `is_bipartite`, `is_planar` (2) | `correct` |

For a full 2,400-query run: `relative_error` on 300 records (one property ×
300 graphs), `absolute_error`+`correct` (exact) on 1,500, `correct` (boolean)
on 600 — before any chromatic-number exclusions.

**Relative error** = `|predicted − truth| / |truth|`, falling back to
absolute error whenever `|truth| < 0.01` — the PDF's Eq. 1 threshold
(§6.2: *"If |v| < 0.01, use absolute error instead to avoid division by
near-zero"*), not just when truth is exactly 0. This matters concretely for
`avg_clustering`: it's bounded `[0, 1]` and plenty of sparse/tree-like graphs
in this dataset land at or near 0, so a near-zero-but-nonzero truth (e.g.
`0.005`) would otherwise blow up a small absolute miss into a huge relative
error. (The Geometry domain's own `relative_scalar_error` only special-cases
*exactly* 0 — harmless there since `area`/`perimeter` are bounded away from
0 by the validity constraints, but not a pattern to copy blindly into a
domain where the numeric property can genuinely sit near zero.)
**`correct_Npct`** = relative error ≤ N%, three thresholds stored so the
accuracy bar is chosen at analysis time (PDF §6.2's strict/moderate/lenient
tiers: 1% / 5% / 10%). There is no bbox/centroid-style normalized point
error in this domain — nothing here is a multi-value coordinate, so no scale
normalizer (Geometry's `bbox_diagonal`) is needed either.

---

## 8. Known gaps

Carried over deliberately, since they are true of this harness too, and
listed here so a future porter doesn't have to rediscover them:

1. **06 is off-protocol** (§4) — mirrors the Geometry domain's own
   deviation from the PDF's literal thinking-mode-V4-Pro spec, for
   cross-domain comparability. If a true thinking-mode run is added later,
   it needs a much higher `max_tokens` and the provider's default thinking
   config left on.
2. **No production run logs exist yet** — none of the six models have been
   queried against real APIs as of this commit. `logs/` directories don't
   exist until a run creates them.
3. Schema drift risk is structurally lower than Geometry's (one shared
   `make_record`, not six copies), but analysis code should still use
   `.get()` rather than `[...]` for optional fields as a defensive habit.

---

## 9. Porting to a new domain

Because the harness is already factored out, porting to a third domain (or
back-porting this structure onto the Geometry domain) only touches
`_common/graph_harness.py`'s domain-specific section (top of the file) and
the record fields copied from the dataset in `make_record`:

| What | Where | Notes |
|------|-------|-------|
| Property list | `PROPERTIES`, `LOCAL_PROPERTIES`, `INTEGER_PROPERTIES`, `SCALAR_REL_PROPERTIES`, `BOOLEAN_PROPERTIES` | drives CLI choices, parsing, and metrics |
| Prompt | `TEMPLATE`, `QUESTIONS`, `build_prompt` | keep one template for all properties |
| Answer normalization | `normalize_answer` | one branch per answer *type*, not per property |
| Metrics | `add_metrics`, `relative_scalar_error` | plus a scale normalizer if the domain has multi-value or unbounded-magnitude properties |
| Dataset field names copied into records | `make_record` | e.g. `family`/`num_nodes` here vs. `shape_type`/`num_vertices` in Geometry |
| Per-model differences | `ModelConfig` fields in each `run_<model>_full.py` | model id, provider, thinking mechanism, JSON mode, prices, coverage |

Everything else — resume, retry, concurrency, logging, cost accounting,
`--fail-fast`, `write_json_export`, `classify_failure`, the CLI surface —
carries over completely unchanged, because none of it depends on what a
"property" or "object" means in a given domain.

See [`../phase1_dataset_graph/README.md`](../phase1_dataset_graph/README.md)
§6 for the dataset-generation side of this domain's own porting notes.

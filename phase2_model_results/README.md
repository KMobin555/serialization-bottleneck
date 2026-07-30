# Phase 2 — Model Query Runs

Documentation for the six runner scripts covered here.

They are **one harness, six configurations**. This document describes the shared
harness once, then tabulates what differs per model. It deliberately does not
walk through each script separately — a `diff` between any two runners is ~50
lines out of ~700, and all of it is provider plumbing and naming.

Phase 1 (`../phase1_dataset/`) produced the 300-polygon dataset. Phase 2 asks
each model 9 questions per polygon, scores the answers against ground truth, and
writes one record per query.

---

## 1. Folder layout

| Folder | Model | Coverage | Queries |
|--------|-------|----------|---------|
| `01_v4flash/` | DeepSeek-V4-Flash | full 300 | 2,700 |
| `02_qwen3/` | Qwen3-32B | full 300 | 2,700 |
| `03_llama_scout/` | Llama 4 Scout | full 300 | 2,700 |
| `04_gemini/` | Gemini 2.5 Flash-Lite | full 300 | 2,700 |
| `05_gpt/` | GPT-4.1 Mini | full 300 | 2,700 |
| `06_v4pro_nonthinking/` | DeepSeek-V4-Pro, thinking OFF | 20% subsample (60) | 540 |

Each folder holds:

```
run_<model>_full.py            the runner
<model>_results.jsonl          append-only, one line per query
<model>_results.json           JSON array, latest record per (object_id, property)
logs/<model>_<timestamp>.log   per-run log
subsample_<model>.json         only in 06 — the 60 object_ids
build_subsample.py             only in 06 — generates that subsample file
```

All six are **non-thinking** runs under the same inference settings —
temperature 0, no system prompt, zero-shot, thinking disabled, all 9 properties.
No run is a side experiment; all six feed the final analysis.

**The only structural difference between them is coverage**, and it is a
property of the input set, not of the harness:

- `01`–`05` query the full 300-polygon dataset → 2,700 records each.
- `06` queries a **20% stratified subsample**, 60 polygons → 540 records.

Everything downstream — prompt, parsing, metrics, record schema — is identical.
An analysis that groups by `tier` / `shape_type` therefore works on `06` exactly
as it does on the others, only with ~⅕ the records per cell.

---

## 2. The shared harness

Every runner is the same eight-stage pipeline:

```
load .env  →  load + validate dataset  →  (optional) apply subsample
    →  read JSONL, compute already-done pairs  →  build task list
    →  async fan-out, semaphore-capped  →  call model, retry on transient errors
    →  parse answer  →  score against ground truth  →  append JSONL
    →  export JSON  →  print cost summary
```

### Dataset loading and validation

`load_dataset()` reads the Phase-1 dataset given by `--dataset`, then:

- warns if the record count is not 300,
- raises on a duplicate `object_id`,
- raises if any record is missing `object_id`, `tier`, `shape_type`,
  `num_vertices`, `wkt`, or `properties`.

A malformed dataset fails at startup, not 2,000 queries in.

> **Path note:** `DEFAULT_DATASET` in every script points at
> `../phase_1_domain1_geometry/geometry_exp1_dataset.json`, which is the folder
> name from the original working tree. In this repository the folder is
> `phase1_dataset/`, so the default path does not resolve — pass
> `--dataset ../phase1_dataset/geometry_exp1_dataset.json` explicitly, or update
> the constant.

### Subsampling (06 only)

`apply_subset()` filters the dataset to the `object_ids` listed in a subsample
file. It raises if any requested id is missing from the dataset, so a stale
subsample file cannot silently shrink the run.

`06_v4pro_nonthinking/subsample_v4pro_nonthinking.json` holds 60 ids plus
metadata:

```json
{"purpose": ..., "seed": 42, "rate": 0.2, "n_polygons": 60, "n_queries": 540, "object_ids": [...]}
```

The stratification is exact, verified against the Phase-1 dataset — 20 polygons
per tier, and within each tier 7 convex / 7 concave / 6 irregular:

| Tier | convex | concave | irregular | Total |
|------|--------|---------|-----------|-------|
| simple | 7 | 7 | 6 | 20 |
| medium | 7 | 7 | 6 | 20 |
| hard | 7 | 7 | 6 | 20 |

So the subsample preserves the full dataset's tier and shape balance rather than
being a random 20% that could skew toward one cell. Every `(tier, shape)` group
is represented, which is what keeps `06` comparable to the full-coverage runs
under the same group-by.

#### How the subsample is generated

`06_v4pro_nonthinking/build_subsample.py` produces this file. Run it in that
folder and the 20% subsample is built from the Phase-1 dataset, all constraints
enforced:

```bash
python build_subsample.py --verify     # check the committed file, no write
python build_subsample.py --dry-run    # print the allocation table
python build_subsample.py              # write it (refuses to clobber; --force to overwrite)
```

`--verify` regenerates the ids and compares them against the committed file:
**60/60 match**. The file and the script are two views of the same thing, and
the check is one command.

The procedure, all of it determined by `--seed` (default 42):

1. **Group** the dataset into the 9 `(tier, shape_type)` strata — 34/33/33 per
   tier.
2. **Allocate per tier by largest remainder.** At 20% the exact quotas are
   6.8 / 6.6 / 6.6; the floors give 6/6/6 = 18, and the 2 leftover slots go to
   the largest fractional parts — convex (.8) first, then concave. Result 7/7/6,
   exactly 20 per tier. Allocating *within* each tier rather than globally is
   what guarantees equal tier counts; a global draw would let one tier drift.
3. **Draw** each stratum with `random.Random(seed).sample()`, iterating tiers in
   order `simple → medium → hard` and shapes in order `convex → concave →
   irregular`. **The iteration order is part of the algorithm**: it fixes the
   sequence in which the RNG is consumed, and changing it changes which polygons
   come out for the same seed.
4. **Validate** by re-deriving the composition from the drawn ids against the
   dataset — the same independent-check discipline Phase 1 uses — then write the
   ids sorted.

Every polygon is eligible; there is no secondary filtering. The Phase-1 generator
already rejected anything invalid, so the dataset is the sampling frame as-is.

The script is domain-agnostic apart from the `TIERS` / `SHAPES` constants and the
`N_PROPERTIES` count, so a port only edits those.

`01`–`05` pass `--subset none` by default and run the full 300.

### Resume

The JSONL is **append-only** and is the source of truth for what has been done.
On startup `done_pairs()` reads every line and builds the set of already-logged
`(object_id, property)` pairs; `build_tasks()` skips them.

Consequences:

- a crash or Ctrl-C costs only the in-flight queries,
- re-running the same command resumes rather than duplicating,
- `--force` re-runs everything,
- `--retry-parse-failures` frees pairs whose logged record has
  `parse_success: false`, so they get re-queried.

Because it is append-only, a re-queried pair leaves **both** records in the
JSONL. `write_json_export()` resolves that: it keeps the *last* record per pair
and writes the `.json` array. **The `.json` file is the clean, deduplicated
artifact — analyze that, not the JSONL.**

Live example: `03_llama_scout` has 2,749 JSONL lines but 2,700 unique pairs — 49
pairs were parse failures on the first pass and were re-queried. The `.json`
export has exactly 2,700 records.

> **Resume only works if the script is pointed at the existing JSONL.** The
> scripts default to writing inside a `results/` subdirectory, but the committed
> result files sit at the folder root. Run one unmodified and it will find an
> empty history and re-query all 2,700 pairs. Always pass explicit
> `--jsonl-output` / `--json-output` paths, or fix the constants first.

### Concurrency and retry

- `asyncio.gather` over all tasks, capped by `asyncio.Semaphore(--concurrency)`,
  default 5.
- An `asyncio.Lock` serializes JSONL appends, each flushed immediately.
- `tenacity` retry with exponential backoff on the API call.

Retry policy differs slightly between the first runner and the rest:

| | `01_v4flash` | `02`–`06` |
|---|---|---|
| retried exceptions | `Exception` (everything) | `APITimeoutError`, `RateLimitError`, `APIConnectionError`, `InternalServerError` |
| attempts | `stop_after_attempt(3)` | `stop_after_attempt(4)` = 1 initial + 3 retries |
| backoff | `multiplier=1, min=2, max=30` | `multiplier=2, min=2, max=30` → 2s, 4s, 8s |

The later policy is the better one: retrying every `Exception` also retries
deterministic failures like a bad model name, three times, for nothing.

### Two execution modes

- **Normal (concurrent)** — a failed query is recorded in a `failures` list and
  the batch continues. At the end the failed pairs are printed; re-running
  retries only those.
- **`--fail-fast` (sequential)** — stops at the *first* API error or parse
  failure and dumps the full context: object_id, tier, shape, vertices,
  property, ground truth, raw model output, WKT, log path. Sequential so that
  "first" is meaningful. Built for debugging a new provider, not for production
  runs.

### Logging

Two handlers: a timestamped file at DEBUG (every query — parse status, token
counts, latency, truncated answer) and stderr at INFO. One file per run. Tenacity
logs retry warnings into the same logger.

> **The production run logs are not in this repository.** The `logs/` folders
> exist but are empty, and log files have never been tracked by git. The real runs
> happened 2026-06-26 → 2026-07-17; that timeline survives only in the
> `timestamp` field of each result record. Any `logs/*.log` you find locally is
> from a later re-run, not from the run that produced the committed results.

### Cost accounting

Token usage is summed from the API response (`usage.prompt_tokens`,
`usage.completion_tokens`) — not estimated. Cost = `in/1e6 × --price-in +
out/1e6 × --price-out`, printed at the end and logged. Prices are per-model
defaults (table below).

---

## 3. The prompt

**One template, all 9 properties, all six models.** Verified: the
`TEMPLATE` + `QUESTIONS` block is byte-identical across every script (same MD5).

```
You are given a polygon in WKT format.

Polygon:
{wkt}

Question:
{question}

Respond with only a JSON object in this format:
{"answer": <your answer>}
```

Zero-shot. **No system prompt.** No few-shot examples. No chain-of-thought
instruction. Single user message.

### The 9 questions

| Property | Question text |
|----------|---------------|
| `vertex_count` | How many vertices does this polygon have? Do not count the repeated closing coordinate. |
| `bbox` | What is the axis-aligned bounding box of this polygon? Give [min_x, min_y, max_x, max_y]. |
| `centroid` | What is the centroid of this polygon? Give [x, y]. |
| `area` | What is the area of this polygon? |
| `perimeter` | What is the perimeter of this polygon? |
| `convex` | Is this polygon convex? Answer true or false. |
| `orientation` | Is the exterior ring of this polygon oriented clockwise or counter-clockwise? Answer "cw" or "ccw". |
| `aspect_ratio` | What is the aspect ratio of this polygon's bounding box (width divided by height)? |
| `edge_length_variance` | What is the variance of the edge lengths of this polygon? |

The `vertex_count` question explicitly excludes the closing coordinate, and the
`bbox` / `centroid` questions pin the output ordering — both remove ambiguity
that would otherwise show up as a wrong answer rather than a modelling result.

### The one prompt exception

`02_qwen3` appends `" /no_think"` to the prompt when the provider is OpenRouter
and thinking is disabled. Qwen3's chat template honours that token regardless of
which upstream provider OpenRouter routes to, whereas the API-level reasoning
flag is not reliably respected. The runner sends both.

So Qwen3's stored prompts end with `...{"answer": <your answer>} /no_think`,
and every other model's end with `...{"answer": <your answer>}`. The question
text itself is untouched. Every stored record carries its exact prompt in the
`prompt` field, so this is auditable per record.

---

## 4. Per-model configuration

All shared settings, in one place: **temperature 0, max_tokens 512, concurrency
5, thinking disabled, no system prompt, zero-shot, all 9 properties** — except
where the table says otherwise.

| | 01 V4-Flash | 02 Qwen3-32B | 03 Llama 4 Scout | 04 Gemini 2.5 Flash-Lite | 05 GPT-4.1 Mini | 06 V4-Pro (no-think) |
|---|---|---|---|---|---|---|
| model id | `deepseek-v4-flash` | `qwen/qwen3-32b` | `meta-llama/llama-4-scout` | `gemini-2.5-flash-lite` | `gpt-4.1-mini` | `deepseek-v4-pro` |
| provider | deepseek | openrouter | openrouter | google | openai | deepseek |
| API key env | `DEEPSEEK_API_KEY` | `OPENROUTER_API_KEY` | `OPENROUTER_API_KEY` | `GEMINI_API_KEY` | `OPENAI_API_KEY` | `DEEPSEEK_API_KEY` |
| thinking | disabled | disabled | n/a (native) | disabled | n/a (native) | disabled |
| disable mechanism | `extra_body={"thinking":{"type":"disabled"}}` | `extra_body={"reasoning":{"enabled":false}}` **+** `/no_think` in prompt | none needed | `reasoning_effort="none"` | none needed | `extra_body={"thinking":{"type":"disabled"}}` |
| JSON mode | off | off | **on** | **on** | **on** | off |
| max_tokens | 512 | 512 | 512 | 512 | 512 | 512 |
| coverage | 300 | 300 | 300 | 300 | 300 | 60 (20%) |
| price in / out ($/1M) | 0.14 / 0.28 | 0.08 / 0.28 | 0.10 / 0.30 | 0.10 / 0.40 | 0.40 / 1.60 | 0.435 / 0.87 |

Provider base URLs (`PROVIDERS` dict, present in every script, superset grows
across the later ones):

| Key | Base URL |
|-----|----------|
| `deepseek` | `https://api.deepseek.com` |
| `openrouter` | `https://openrouter.ai/api/v1` |
| `google` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `openai` | `https://api.openai.com/v1` |

Every provider is reached through the **OpenAI-compatible `AsyncOpenAI`
client** — only `base_url` and the key change. That is the single design choice
that makes one harness serve five vendors. OpenRouter attribution headers
(`X-Title`, `HTTP-Referer`) are attached only when the provider is `openrouter`.

### Two mechanisms, easy to confuse

**Disabling thinking** and **JSON mode** solve different problems, and the models
here need both:

- *Thinking* is the provider's hidden reasoning pass. Disabling it is a per-vendor
  API call: DeepSeek `extra_body.thinking.type`, OpenRouter
  `extra_body.reasoning.enabled`, Google `reasoning_effort="none"`. Llama 4 Scout
  and GPT-4.1 Mini have no such mode, so nothing is sent.
- *Content-level chain-of-thought* is the model writing "Let's compute this step
  by step…" in the visible answer. Disabling thinking does not stop it. On hard
  polygons that narration overflows the 512-token cap and the answer never
  arrives.

`response_format={"type":"json_object"}` (the `--json-mode on` default in 03, 04,
05) is the fix for the second problem. It enforces exactly what the prompt
already asks for; the question text is unchanged.

Evidence that this is a real effect, not a precaution: run `06` (V4-Pro,
thinking disabled, **no** JSON mode) still produced answers like

> `Let's compute the perimeter step by step. Vertices: (10, 9), (1, 41), …`

at 536–4,268 completion tokens, versus a 9–17 token average for the JSON-mode
models.

---

## 5. Running the scripts

### Prerequisites

**Setup lives in the [root README](../Readme.md#setup)** — virtual environment,
`pip install -r requirements.txt`, and the `.env` API keys, covering Phase 1 and
Phase 2 at once.

This section assumes all of that is already in place. Activate the environment
and go straight to the run commands:

```bash
source .venv/bin/activate     # from the repo root
```

The only thing to confirm here is that you have the API key for the provider you
intend to run (§4 lists which key each model needs). A missing key fails at
startup with `<KEY_NAME> not set. Add it to .env.`

### Two flags you must pass

Both defaults in the scripts are stale (§9, items 1–2), so every real command
below overrides them:

- `--dataset ../../phase1_dataset/geometry_exp1_dataset.json` — the built-in
  default points at a folder that does not exist here.
- `--jsonl-output <file>.jsonl --json-output <file>.json` — the built-in default
  writes into a `results/` subfolder, while the committed results sit at the
  folder root. Without these, the script sees no history and re-queries all
  2,700 pairs.

The commands below spell both out in full. Do not collapse them into a shell
variable like `$FLAGS` — zsh, the macOS default shell, does not word-split
unquoted variables, so the whole string arrives as one argument and argparse
rejects it.

### Per-model commands

Each block: dry run first (free, no API key needed), then a 45-query smoke test,
then the full run. Run each from inside its own folder. Every run is resumable —
re-issue the same command after any interruption and it continues where it
stopped.

**01 — DeepSeek-V4-Flash** (needs `DEEPSEEK_API_KEY`, 2,700 queries)

```bash
cd phase2_model_results/01_v4flash

python run_v4flash_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4flash_results.jsonl \
  --json-output v4flash_results.json \
  --dry-run

# smoke test: 5 polygons x 9 properties = 45 queries, stops at the first problem
python run_v4flash_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4flash_results.jsonl \
  --json-output v4flash_results.json \
  --limit 5 --fail-fast

# full run
python run_v4flash_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4flash_results.jsonl \
  --json-output v4flash_results.json
```

**02 — Qwen3-32B** (needs `OPENROUTER_API_KEY`, 2,700 queries)

```bash
cd phase2_model_results/02_qwen3

python run_qwen3_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output qwen3_results.jsonl \
  --json-output qwen3_results.json \
  --dry-run

python run_qwen3_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output qwen3_results.jsonl \
  --json-output qwen3_results.json \
  --limit 5 --fail-fast

python run_qwen3_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output qwen3_results.jsonl \
  --json-output qwen3_results.json
```

**03 — Llama 4 Scout** (needs `OPENROUTER_API_KEY`, 2,700 queries)

```bash
cd phase2_model_results/03_llama_scout

python run_llama_scout_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output llama_scout_results.jsonl \
  --json-output llama_scout_results.json \
  --dry-run

python run_llama_scout_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output llama_scout_results.jsonl \
  --json-output llama_scout_results.json \
  --limit 5 --fail-fast

python run_llama_scout_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output llama_scout_results.jsonl \
  --json-output llama_scout_results.json
```

**04 — Gemini 2.5 Flash-Lite** (needs `GEMINI_API_KEY`, 2,700 queries)

```bash
cd phase2_model_results/04_gemini

python run_gemini_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gemini_results.jsonl \
  --json-output gemini_results.json \
  --dry-run

python run_gemini_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gemini_results.jsonl \
  --json-output gemini_results.json \
  --limit 5 --fail-fast

python run_gemini_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gemini_results.jsonl \
  --json-output gemini_results.json
```

**05 — GPT-4.1 Mini** (needs `OPENAI_API_KEY`, 2,700 queries)

```bash
cd phase2_model_results/05_gpt

python run_gpt_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gpt_results.jsonl \
  --json-output gpt_results.json \
  --dry-run

python run_gpt_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gpt_results.jsonl \
  --json-output gpt_results.json \
  --limit 5 --fail-fast

python run_gpt_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gpt_results.jsonl \
  --json-output gpt_results.json
```

**06 — DeepSeek-V4-Pro, non-thinking** (needs `DEEPSEEK_API_KEY`, 540 queries)

Check the subsample first, then query it. The subsample file is already
committed, so `build_subsample.py` is only needed to verify it, or to rebuild it
in a fresh checkout.

```bash
cd phase2_model_results/06_v4pro_nonthinking

python build_subsample.py --verify      # confirms the committed 60 ids regenerate

python run_v4pro_nonthinking_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4pro_nonthinking_results.jsonl \
  --json-output v4pro_nonthinking_results.json \
  --dry-run

python run_v4pro_nonthinking_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4pro_nonthinking_results.jsonl \
  --json-output v4pro_nonthinking_results.json \
  --limit 5 --fail-fast

python run_v4pro_nonthinking_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4pro_nonthinking_results.jsonl \
  --json-output v4pro_nonthinking_results.json
```

Note: `--subset` already defaults to `subsample_v4pro_nonthinking.json` in this
folder, so it does not need to be passed.

### Useful variations

```bash
# Re-run one property only, e.g. after fixing a parser branch (from 05_gpt/)
python run_gpt_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gpt_results.jsonl --json-output gpt_results.json \
  --properties perimeter --force

# Re-query only the pairs that failed to parse
python run_gpt_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output gpt_results.jsonl --json-output gpt_results.json \
  --retry-parse-failures

# Gentler on rate limits (from 02_qwen3/)
python run_qwen3_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output qwen3_results.jsonl --json-output qwen3_results.json \
  --concurrency 2

# Phase-1 dataset regeneration (notebook, not a runner) — from the repo root
jupyter notebook phase1_dataset/Geometry_Experiment1_Phase1.ipynb
```

Re-running a completed model prints `Planned new queries: 0 (skipping 2700
already done)` and exits without spending anything — that is the resume logic
confirming the results are intact.

### Flags

Identical across all six runners except where noted.

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset` | `../phase_1_domain1_geometry/...json` | Phase-1 dataset path (see path note above) |
| `--subset` | `none` (01–05) / subsample file (06) | restrict to listed object_ids |
| `--jsonl-output` | `results/<model>_results.jsonl` | append-only log |
| `--json-output` | `results/<model>_results.json` | deduplicated export |
| `--log-dir` | `logs/` | per-run log directory |
| `--provider` | per model | key into `PROVIDERS` |
| `--model` | per model | model id string |
| `--temperature` | `0.0` | |
| `--max-tokens` | `512` | completion cap |
| `--concurrency` | `5` | in-flight query cap |
| `--thinking` | `disabled` | thinking mode toggle |
| `--json-mode` | `on` (03, 04, 05 only) | `response_format` JSON object mode |
| `--properties` | all 9 | re-run a subset, e.g. `--properties perimeter` |
| `--fail-fast` | off | sequential, stop at first failure, dump context |
| `--price-in` / `--price-out` | per model | $/1M tokens for the cost summary |
| `--limit N` | none | first N polygons — smoke test |
| `--force` | off | re-run even already-logged pairs |
| `--retry-parse-failures` | off | re-query pairs logged with `parse_success: false` |
| `--dry-run` | off | build prompts and task list, print 3, no API calls |

Nothing executes on import — everything is behind `main()` under
`if __name__ == "__main__"`.

Typical sequence for a new model:

```bash
python run_<model>_full.py --dry-run              # inspect the prompts
python run_<model>_full.py --limit 5 --fail-fast  # 45 queries, stop at first problem
python run_<model>_full.py                        # full run, resumable
```

---

## 6. Answer parsing

Model output is text; ground truth is typed. `parse_answer()` bridges them in
three escalating attempts:

1. Parse the whole response as JSON and read `["answer"]`.
2. Regex-scan for any `{...}` block, parse each, read `["answer"]`.
3. Fall back to normalizing the raw string directly.

Whatever survives goes through `normalize_answer()`, which is per-property:

| Property group | Accepts | Rejects |
|----------------|---------|---------|
| `vertex_count` | int, integer-valued float, first number in a string | booleans, non-integer floats |
| `area`, `perimeter`, `aspect_ratio`, `edge_length_variance` | int/float, first number in a string (commas stripped) | booleans, non-finite values |
| `bbox` (4 values), `centroid` (2 values) | list/tuple of numbers, or a string scanned for numbers | wrong length, booleans, non-finite |
| `convex` | bool, or `true`/`yes`/`false`/`no` as a word in a string | anything else |
| `orientation` | `cw` / `ccw` / `clockwise` / `counter-clockwise` in a string | anything else |

Booleans are rejected everywhere numbers are expected, because Python treats
`True` as `1` and would otherwise score `true` as a numeric answer.

Failure to parse sets `parse_success: false`, `parsed_answer: null`, and every
correctness flag to `false` — an unparseable answer is **incorrect**, never
missing. This matters: dropping unparseable rows instead would inflate accuracy
for models that ramble.

---

## 7. Result record shape

One JSON object per query. Fields that **every** record carries:

```json
{
  "object_id": "poly_simple_convex_001",
  "tier": "simple",
  "shape_type": "convex",
  "num_vertices": 3,
  "property": "area",
  "property_locality": "global",
  "ground_truth": 1278.5,
  "prompt": "You are given a polygon in WKT format....",
  "raw_model_output": "{\"answer\": 1278.5}",
  "parsed_answer": 1278.5,
  "parse_success": true,
  "model": "deepseek-v4-flash",
  "temperature": 0.0,
  "prompt_tokens": 185,
  "completion_tokens": 9,
  "timestamp": "2026-06-26T05:33:56.300346+00:00"
}
```

The first four fields are copied from the Phase-1 record so results can be sliced
by tier / shape / complexity without re-joining against the dataset.

`property_locality` is `"local"` for `vertex_count` and `bbox` — properties
readable from a bounded window of the coordinate string — and `"global"` for the
other seven, which require the whole ring. This is the local-vs-global cut the
experiment is built to test.

`prompt` and `raw_model_output` are stored verbatim on every record. Nothing has
to be reconstructed at analysis time, and any scoring dispute can be re-checked
against the exact bytes sent and received.

### Metric fields — depend on the property

Scoring keys are **not** uniform across records. Which ones appear is determined
by the property's evaluation mode:

| Property group | Fields added |
|----------------|--------------|
| `area`, `perimeter`, `aspect_ratio`, `edge_length_variance` (4) | `absolute_error`, `relative_error`, `correct_1pct`, `correct_5pct`, `correct_10pct` |
| `bbox`, `centroid` (2) | same five, but the error is a **distance normalized by `bbox_diagonal`** |
| `vertex_count` (1) | `absolute_error`, `correct`, `off_by_one` |
| `convex`, `orientation` (2) | `correct` |

For a full 2,700-query run that means: `relative_error` on 1,800 records,
`absolute_error` on 2,100, `correct` on 900, `off_by_one` on 300.

Definitions:

- **relative error** (scalars) = `|predicted − truth| / |truth|`, falling back to
  absolute error when truth is 0.
- **normalized error** (`bbox`, `centroid`) = Euclidean distance for the 2-value
  centroid, max coordinate deviation for the 4-value bbox, divided by that
  polygon's `bbox_diagonal` from Phase-1 metadata. Without this normalization a
  hard-tier polygon (coordinates to 1000) and a simple-tier one (to 100) are not
  comparable.
- **`correct_Npct`** = relative/normalized error ≤ N%. Three thresholds stored,
  so the accuracy bar can be chosen at analysis time instead of being baked in.
- **`off_by_one`** = the vertex-count answer missed by exactly 1 — usually the
  model counting the repeated closing coordinate, which the prompt explicitly
  warns against.

### Fields that vary by runner

The record schema grew as the runs progressed. Three optional fields:

| Field | Present in | Meaning |
|-------|------------|---------|
| `provider` | 02–06 (**not 01**) | which vendor served the call |
| `finish_reason`, `failure_type` | 04, 05, 06 | see below |
| `thinking` | 06 only | hard-coded `"disabled"`, recording the mode explicitly |

`failure_type` implements PDF §6.1: when a record fails to parse, it is
`"reasoning_truncated"` if the call hit the token ceiling
(`finish_reason == "length"` or `completion_tokens >= max_tokens`), otherwise
`"parse_failure"`. It is `null` when parsing succeeded. Both kinds count as
incorrect — the distinction is diagnostic, separating "the model was cut off"
from "the model answered in a format we could not read".

Runners 01, 02, 03 predate this field, so their parse failures cannot be
attributed. Concretely, `03_llama_scout`'s 49 first-pass parse failures carry no
`failure_type`. Adding the field to those three and re-exporting is a small,
worthwhile cleanup — the raw output is stored, so it can be back-filled without
re-querying.

---

## 8. What the runs actually produced

Measured from the committed results, not from the scripts:

| Run | JSONL lines | Unique pairs | Parse success | Avg in-tokens | Avg out-tokens | Max out |
|-----|-------------|--------------|---------------|---------------|----------------|---------|
| 01 V4-Flash | 2,700 | 2,700 | 2,700 / 2,700 | 185 | 9.3 | 50 |
| 02 Qwen3-32B | 2,700 | 2,700 | 2,700 / 2,700 | 267 | 17.5 | 47 |
| 03 Llama 4 Scout | 2,749 | 2,700 | 2,700 / 2,749 | 178 | 10.4 | 27 |
| 04 Gemini 2.5 Flash-Lite | 2,700 | 2,700 | 2,700 / 2,700 | 261 | 12.6 | 500 |
| 05 GPT-4.1 Mini | 2,700 | 2,700 | 2,700 / 2,700 | 185 | 9.1 | 23 |
| 06 V4-Pro non-thinking | 540 | 540 | 540 / 540 | 187 | 85.0 | 4,268 |

Notes on reading this table:

- **Every final export is 100% parseable.** The 49-record gap in `03` is
  first-pass failures that were re-queried; the deduplicated `.json` is clean.
- **Input-token averages differ across providers for an identical prompt**
  (178–267). That is tokenizer variation, not prompt variation — the prompt
  strings are stored per record and match.
- **04 Gemini is the only run that hit the token ceiling**: 3 of 2,700 records
  have `finish_reason: "length"`. All three still parsed, so `failure_type` is
  `null` across the whole run and no answer was lost — but it is the one run
  where a slightly longer answer would have been truncated. Every other run is
  `finish_reason: "stop"` on every record.
- **06's average output is 85 tokens against 9–17 for the JSON-mode runs**, and
  its distribution has a long tail. That gap is content-level chain-of-thought,
  not thinking tokens — thinking was disabled for this run too. It is the only
  run without JSON mode among the models that ramble.
- **06's stored records exceed its own `--max-tokens 512` default** — 41 records
  have >512 completion tokens, all with `finish_reason: "stop"`. The production
  run therefore used a higher cap than the value now in the script. The committed
  log is a dry run and does not record the real invocation. Treat the script
  default as *not* reproducing run 06 exactly.
- **Record counts scale exactly with coverage**: 540 = 60 polygons × 9
  properties, against 2,700 = 300 × 9. Per-query behaviour is unchanged; only the
  sample size differs.

---

## 9. Known gaps

Listed so a porter does not rediscover them:

1. `DEFAULT_DATASET` points at `../phase_1_domain1_geometry/`, which does not
   exist in this repo (the folder is `phase1_dataset/`). Every script.
2. **Output paths do not match where the results actually live.** Every script
   defaults to `RESULTS_DIR = HERE / "results"`, but the committed `.jsonl` /
   `.json` files sit at the folder root. Running a script unmodified therefore
   writes to a fresh `results/` subdirectory, finds no prior records there, and
   **re-queries every pair from scratch** — 2,700 paid calls per model, silently.
   Pass `--jsonl-output <folder>/<model>_results.jsonl` (and `--json-output`) to
   resume against the real files. See §2 "Resume".
3. Schema drift: `provider` missing in 01; `finish_reason`/`failure_type` missing
   in 01, 02, 03. Analysis code must use `.get()`, not `[...]`.
4. `01_v4flash` retries on bare `Exception` — retries non-retryable errors three
   times.
5. `05_gpt` line 68 carries a copy-pasted comment saying `DEFAULT_SUBSET = "none"
   # Gemini = 100% coverage`. The value is right for GPT; the comment names the
   wrong model.
6. No run logs are tracked in git, so the production runs (2026-06-26 →
   2026-07-17) have no log record — only the `timestamp` on each result row.
7. Run 06's real `max_tokens` is not recoverable from the repo (see §8).
8. Stale in-file naming in `06_v4pro_nonthinking/`, left over from when the run
   was numbered `07` and treated as supplementary:
   - the runner's module docstring still describes the run as off-protocol and
     points at a sibling thinking-mode folder that no longer exists,
   - `subsample_v4pro_nonthinking.json` has `"purpose": "... (PDF: thinking-mode,
     20% coverage)"`, describing a different run.

   Prose only — no effect on execution or on the stored results, since the file
   is read for its `object_ids` alone. `build_subsample.py --force` rewrites the
   subsample file with corrected wording and byte-identical ids; the runner
   docstring still needs a manual edit.
9. Six near-identical ~700-line scripts. Any harness fix has to be applied six
   times, and the drift in items 3–5 is the direct result.

---

## 10. Porting to a new domain

The harness is domain-agnostic; the domain lives in six places.

**Reuse unchanged:** `PROVIDERS`, `make_client`, the async/semaphore fan-out,
tenacity retry, JSONL resume + `done_pairs`, `write_json_export`, logging,
`--fail-fast`, the CLI surface, cost accounting, `classify_failure`, and the
record envelope.

**Replace for a new domain:**

| What | Where | Notes |
|------|-------|-------|
| Property list | `PROPERTIES` | drives CLI choices and the task grid |
| Locality split | `LOCAL_PROPERTIES` | which properties are readable from a bounded window |
| Prompt | `TEMPLATE`, `QUESTIONS`, `build_prompt` | keep one template for all properties |
| Answer normalization | `normalize_answer` | one branch per answer type |
| Metrics | `relative_scalar_error`, `normalized_point_error`, `add_metrics` | plus the scale normalizer |
| Scale normalizer | `row["metadata"]["bbox_diagonal"]` | whatever makes tiers comparable in the new domain |

Everything else — including `parse_answer`'s three-tier JSON extraction — carries
over as-is, because it operates on the `{"answer": ...}` envelope rather than on
geometry.

**If you port, fix the duplication first.** Extract the shared harness into one
module and reduce each runner to a config block: model id, provider, thinking
mechanism, JSON mode, coverage, prices. That is roughly 60 lines per model
instead of 700, and it makes items 3–5 in §9 structurally impossible.

See `../phase1_dataset/README.md` for the dataset-generation side of the port.

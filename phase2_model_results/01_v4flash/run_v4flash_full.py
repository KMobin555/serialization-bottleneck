"""
Domain 1 — FULL DeepSeek-V4-Flash query run (all 300 polygons x 9 properties).

This is Experiment 1, Phase 2 (model queries) for the V4-Flash debug-first model.
The knock (18-polygon subset) already validated the harness; this runs the whole
geometry dataset.

Per Experiment 1 protocol (serialization_experiment_1.pdf):
  - 300 polygons x 9 properties = 2,700 queries
  - DeepSeek-V4-Flash, NON-thinking mode (PDF: all models except V4-Pro are
    non-thinking) -> extra_body={"thinking": {"type": "disabled"}}
  - temperature 0, max_tokens 512, no system prompt, zero-shot
  - retry: exponential backoff, 3 attempts

All output stays inside this folder:
  results/v4flash_results.jsonl  -- append-only, resume-safe, one record/query
  results/v4flash_results.json   -- JSON array export (latest record per pair)
  logs/v4flash_<timestamp>.log   -- detailed per-query log (one file per run)

Resume: re-running skips already-logged (object_id, property) pairs, so a crash
loses nothing. Fail-fast (--fail-fast) stops at the first error/parse failure.

NOTHING runs on import. Execute explicitly:
  python run_v4flash_full.py                 # full run, concurrent
  python run_v4flash_full.py --fail-fast     # sequential, stop at first failure
  python run_v4flash_full.py --dry-run       # build prompts, no API calls
  python run_v4flash_full.py --limit 5       # smoke test: first 5 polygons
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
# Dataset lives in the Phase-1 folder; this script only writes inside its own folder.
DEFAULT_DATASET = HERE.parent / "phase_1_domain1_geometry" / "geometry_exp1_dataset.json"
RESULTS_DIR = HERE / "results"
LOG_DIR = HERE / "logs"
DEFAULT_JSONL = RESULTS_DIR / "v4flash_results.jsonl"
DEFAULT_JSON = RESULTS_DIR / "v4flash_results.json"

# Module logger; handlers attached in setup_logging(). tenacity logs retries here.
LOG = logging.getLogger("v4flash")


def setup_logging(log_dir: Path) -> Path:
    """Log to a timestamped file (DEBUG, every query) AND console (INFO).
    Each run keeps its own file. Returns the file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"v4flash_{stamp}.log"

    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    LOG.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    LOG.addHandler(ch)
    return log_path


# Provider presets: (base_url, env var for api key)
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# All 9 Experiment-1 geometry properties.
PROPERTIES = (
    "vertex_count", "bbox", "centroid", "area", "perimeter",
    "convex", "orientation", "aspect_ratio", "edge_length_variance",
)
LOCAL_PROPERTIES = {"vertex_count", "bbox"}
SCALAR_REL_PROPERTIES = {"area", "perimeter", "aspect_ratio", "edge_length_variance"}

TEMPLATE = """You are given a polygon in WKT format.

Polygon:
{wkt}

Question:
{question}

Respond with only a JSON object in this format:
{{"answer": <your answer>}}"""

QUESTIONS = {
    "vertex_count": (
        "How many vertices does this polygon have? "
        "Do not count the repeated closing coordinate."
    ),
    "bbox": (
        "What is the axis-aligned bounding box of this polygon? "
        "Give [min_x, min_y, max_x, max_y]."
    ),
    "centroid": "What is the centroid of this polygon? Give [x, y].",
    "area": "What is the area of this polygon?",
    "perimeter": "What is the perimeter of this polygon?",
    "convex": "Is this polygon convex? Answer true or false.",
    "orientation": (
        "Is the exterior ring of this polygon oriented clockwise or counter-clockwise? "
        'Answer "cw" or "ccw".'
    ),
    "aspect_ratio": (
        "What is the aspect ratio of this polygon's bounding box "
        "(width divided by height)?"
    ),
    "edge_length_variance": "What is the variance of the edge lengths of this polygon?",
}

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def build_prompt(wkt: str, property_name: str) -> str:
    return TEMPLATE.format(wkt=wkt, question=QUESTIONS[property_name])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full V4-Flash query run over all 300 polygons.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    p.add_argument("--model", default="deepseek-v4-flash",
                   help="DeepSeek-direct: deepseek-v4-flash (default). "
                        "OpenRouter: deepseek/deepseek-v4-flash.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)   # PDF setting
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled",
                   help="V4-Flash = non-thinking per PDF -> disabled (default).")
    p.add_argument("--properties", nargs="+", choices=PROPERTIES, default=list(PROPERTIES),
                   help="Subset of properties to query (default: all 9). "
                        "Use to re-run a single property, e.g. --properties perimeter.")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop at the FIRST failure (API error OR parse failure), print "
                        "full details, exit. Sequential so 'first' is exact. Re-run resumes.")
    # Real DeepSeek-direct V4-Flash prices ($ per 1M tokens).
    p.add_argument("--price-in", type=float, default=0.14)
    p.add_argument("--price-out", type=float, default=0.28)
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N polygons (smoke test). Resume still applies.")
    p.add_argument("--force", action="store_true",
                   help="Re-run every pair, even ones already logged.")
    p.add_argument("--retry-parse-failures", action="store_true",
                   help="Also re-query pairs logged with parse_success=false.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts/task list without calling the API.")
    return p.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    if len(data) != 300:
        LOG.warning("expected 300 polygons, found %d in %s", len(data), path)
    seen = set()
    for row in data:
        oid = row.get("object_id")
        if oid in seen:
            raise ValueError(f"duplicate object_id: {oid}")
        seen.add(oid)
        missing = {"object_id", "tier", "shape_type", "num_vertices", "wkt", "properties"} - set(row)
        if missing:
            raise ValueError(f"{oid} missing keys: {sorted(missing)}")
    return data


def read_latest_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open() as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[(rec["object_id"], rec["property"])] = rec
    return latest


def done_pairs(path: Path, retry_parse_failures: bool) -> set[tuple[str, str]]:
    """Pairs to SKIP. A pair is done if logged AND (parsed ok, or not retrying
    parse failures). Crash leaves un-logged pairs; --retry-parse-failures frees
    parse-failed pairs too."""
    done: set[tuple[str, str]] = set()
    for key, rec in read_latest_records(path).items():
        if retry_parse_failures and not rec.get("parse_success", False):
            continue
        done.add(key)
    return done


def normalize_answer(answer: Any, property_name: str) -> tuple[Any, bool]:
    if property_name == "vertex_count":
        if isinstance(answer, bool):
            return None, False
        if isinstance(answer, int):
            return answer, True
        if isinstance(answer, float) and answer.is_integer():
            return int(answer), True
        if isinstance(answer, str):
            m = NUMBER_RE.search(answer)
            if m:
                v = float(m.group())
                if v.is_integer():
                    return int(v), True
        return None, False

    if property_name in SCALAR_REL_PROPERTIES:
        if isinstance(answer, bool):
            return None, False
        if isinstance(answer, (int, float)):
            return float(answer), math.isfinite(float(answer))
        if isinstance(answer, str):
            m = NUMBER_RE.search(answer.replace(",", ""))
            if m:
                v = float(m.group())
                return v, math.isfinite(v)
        return None, False

    if property_name in {"bbox", "centroid"}:
        expected_len = 4 if property_name == "bbox" else 2
        if isinstance(answer, (list, tuple)):
            raw = answer
        elif isinstance(answer, str):
            raw = NUMBER_RE.findall(answer.replace(",", " "))
        else:
            return None, False
        values: list[float] = []
        for item in raw:
            if isinstance(item, bool):
                return None, False
            try:
                v = float(item)
            except (TypeError, ValueError):
                return None, False
            if not math.isfinite(v):
                return None, False
            values.append(v)
        if len(values) != expected_len:
            return None, False
        return values, True

    if property_name == "convex":
        if isinstance(answer, bool):
            return answer, True
        if isinstance(answer, str):
            low = answer.strip().lower()
            if re.search(r"\b(true|yes)\b", low):
                return True, True
            if re.search(r"\b(false|no)\b", low):
                return False, True
        return None, False

    if property_name == "orientation":
        if isinstance(answer, str):
            low = answer.strip().lower()
            if re.search(r"\bccw\b|counter-?clockwise", low):
                return "ccw", True
            if re.search(r"\bcw\b|clockwise", low):
                return "cw", True
        return None, False

    raise ValueError(f"unknown property: {property_name}")


def parse_answer(raw: str, property_name: str) -> tuple[Any, bool]:
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "answer" in parsed:
            ans, ok = normalize_answer(parsed["answer"], property_name)
            if ok:
                return ans, True
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r"\{.*?\}", stripped, re.DOTALL):
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            ans, ok = normalize_answer(parsed["answer"], property_name)
            if ok:
                return ans, True
    ans, ok = normalize_answer(stripped, property_name)
    return (ans, True) if ok else (None, False)


def relative_scalar_error(predicted: Any, truth: Any) -> tuple[float, float]:
    abs_err = abs(float(predicted) - float(truth))
    denom = abs(float(truth))
    rel = abs_err if denom == 0 else abs_err / denom
    return abs_err, rel


def normalized_point_error(predicted: Any, truth: Any, diag: float) -> tuple[float, float]:
    if not isinstance(predicted, list) or len(predicted) != len(truth):
        return math.nan, math.nan
    deltas = [float(p) - float(t) for p, t in zip(predicted, truth)]
    abs_err = math.hypot(deltas[0], deltas[1]) if len(deltas) == 2 else max(abs(d) for d in deltas)
    norm = abs_err / diag if diag else abs_err
    return abs_err, norm


def add_metrics(rec: dict[str, Any], row: dict[str, Any], parsed: Any, ok: bool) -> None:
    name = rec["property"]
    truth = rec["ground_truth"]

    if not ok:
        if name in {"convex", "orientation"}:
            rec["correct"] = False
        elif name == "vertex_count":
            rec["absolute_error"] = None
            rec["correct"] = False
            rec["off_by_one"] = False
        else:
            rec["absolute_error"] = None
            rec["relative_error"] = None
            rec["correct_1pct"] = rec["correct_5pct"] = rec["correct_10pct"] = False
        return

    if name in SCALAR_REL_PROPERTIES:
        abs_err, rel = relative_scalar_error(parsed, truth)
        rec["absolute_error"] = abs_err
        rec["relative_error"] = rel
        rec["correct_1pct"] = rel <= 0.01
        rec["correct_5pct"] = rel <= 0.05
        rec["correct_10pct"] = rel <= 0.10
        return

    if name in {"bbox", "centroid"}:
        abs_err, rel = normalized_point_error(parsed, truth, float(row["metadata"]["bbox_diagonal"]))
        rec["absolute_error"] = abs_err
        rec["relative_error"] = rel
        rec["correct_1pct"] = rel <= 0.01
        rec["correct_5pct"] = rel <= 0.05
        rec["correct_10pct"] = rel <= 0.10
        return

    if name == "vertex_count":
        abs_err = abs(int(parsed) - int(truth))
        rec["absolute_error"] = abs_err
        rec["correct"] = abs_err == 0
        rec["off_by_one"] = abs_err == 1
        return

    if name in {"convex", "orientation"}:
        rec["correct"] = parsed == truth
        return

    raise ValueError(f"unknown property: {name}")


def make_record(row, property_name, raw_output, parsed, ok, model, temperature,
                prompt, usage) -> dict[str, Any]:
    rec = {
        "object_id": row["object_id"],
        "tier": row["tier"],
        "shape_type": row["shape_type"],
        "num_vertices": row["num_vertices"],
        "property": property_name,
        "property_locality": "local" if property_name in LOCAL_PROPERTIES else "global",
        "ground_truth": row["properties"][property_name],
        "prompt": prompt,
        "raw_model_output": raw_output,
        "parsed_answer": parsed,
        "parse_success": ok,
        "model": model,
        "temperature": temperature,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    add_metrics(rec, row, parsed, ok)
    return rec


def make_client(base_url: str, api_key: str, provider: str) -> AsyncOpenAI:
    # OpenRouter attribution headers apply ONLY to the openrouter provider.
    # The deepseek run sends nothing OpenRouter-related.
    default_headers = {}
    if provider == "openrouter":
        default_headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "pilot-geo")
        if os.getenv("OPENROUTER_SITE_URL"):
            default_headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
    return AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=default_headers)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=before_sleep_log(LOG, logging.WARNING),
)
async def call_model(client, model, prompt, temperature, max_tokens, extra_body=None):
    start = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or {},
    )
    latency = time.perf_counter() - start
    content = resp.choices[0].message.content or ""
    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    return content, usage, latency


def build_tasks(dataset, properties, completed, limit, force):
    rows = dataset[:limit] if limit is not None else dataset
    tasks = []
    for row in rows:
        for prop in properties:
            if not force and (row["object_id"], prop) in completed:
                continue
            tasks.append((row, prop))
    return tasks


async def run(args) -> None:
    load_dotenv()
    log_path = setup_logging(args.log_dir)
    LOG.info("=" * 56)
    LOG.info("V4-FLASH FULL RUN start")
    LOG.info("provider=%s model=%s temp=%s max_tokens=%s concurrency=%s",
             args.provider, args.model, args.temperature, args.max_tokens, args.concurrency)
    LOG.info("dataset=%s", args.dataset)
    LOG.info("log file=%s", log_path)

    dataset = load_dataset(args.dataset)
    completed = set() if args.force else done_pairs(args.jsonl_output, args.retry_parse_failures)
    tasks = build_tasks(dataset, args.properties, completed, args.limit, args.force)

    LOG.info("polygons=%d props=%d | planned new queries=%d | skipping=%d",
             len(dataset), len(args.properties), len(tasks), len(completed))
    print(f"Provider: {args.provider}  |  Model: {args.model}  |  thinking: {args.thinking}")
    print(f"Polygons: {len(dataset)} x {len(PROPERTIES)} props = {len(dataset)*len(PROPERTIES)} total")
    print(f"Planned new queries: {len(tasks)}"
          + (f"  (skipping {len(completed)} already done)" if completed else ""))
    print(f"Log file: {log_path}")

    if args.dry_run:
        for row, prop in tasks[:3]:
            print(f"\n--- {row['object_id']} / {prop} ---")
            print(build_prompt(row["wkt"], prop))
        LOG.info("[dry-run] %d queries would be sent. No API calls.", len(tasks))
        print(f"\n[dry-run] {len(tasks)} queries would be sent. No API calls.")
        return

    base_url, key_env = PROVIDERS[args.provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"{key_env} not set. Add it to .env.")

    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    client = make_client(base_url, api_key, args.provider)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()

    extra_body = None
    if args.provider == "deepseek":
        extra_body = {"thinking": {"type": args.thinking}}
    LOG.info("thinking=%s extra_body=%s fail_fast=%s", args.thinking, extra_body, args.fail_fast)

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "parsed_ok": 0, "n": 0}
    failures: list[tuple[str, str, str]] = []

    def write_record(rec, usage, ok):
        totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
        totals["completion_tokens"] += usage.get("completion_tokens") or 0
        totals["parsed_ok"] += int(ok)
        totals["n"] += 1
        with args.jsonl_output.open("a") as h:
            h.write(json.dumps(rec, ensure_ascii=True) + "\n")
            h.flush()

    async def run_one(row, prop):
        prompt = build_prompt(row["wkt"], prop)
        raw, usage, latency = await call_model(
            client, args.model, prompt, args.temperature, args.max_tokens, extra_body
        )
        parsed, ok = parse_answer(raw, prop)
        rec = make_record(row, prop, raw, parsed, ok, args.model,
                          args.temperature, prompt, usage)
        LOG.debug("OK  %-28s %-22s parse=%s in=%s out=%s %.2fs ans=%s",
                  row["object_id"], prop, ok,
                  usage.get("prompt_tokens"), usage.get("completion_tokens"),
                  latency, repr(parsed)[:60])
        return rec, ok, raw, usage

    # ---- FAIL-FAST: sequential; stop at first error OR parse failure ----
    if args.fail_fast:
        LOG.info("FAIL-FAST mode: sequential, stop at first failure.")
        progress = tqdm(total=len(tasks), unit="query")
        for row, prop in tasks:
            try:
                rec, ok, raw, usage = await run_one(row, prop)
            except Exception as exc:
                progress.close()
                stop_on_failure(row, prop, "API ERROR",
                                f"{type(exc).__name__}: {exc}", None, log_path, totals)
                return
            if not ok:
                write_record(rec, usage, ok)
                write_json_export(args.jsonl_output, args.json_output)
                progress.close()
                stop_on_failure(row, prop, "PARSE FAILURE",
                                f"could not parse {prop}", raw, log_path, totals)
                return
            write_record(rec, usage, ok)
            progress.update(1)
        progress.close()
        write_json_export(args.jsonl_output, args.json_output)
        print_summary(totals, args)
        print("\nAll queries completed with NO failures. (fail-fast)")
        LOG.info("FAIL-FAST end | completed=%d tokens in=%d out=%d",
                 totals["n"], totals["prompt_tokens"], totals["completion_tokens"])
        return

    # ---- NORMAL: concurrent; one failure does not abort the batch ----
    progress = tqdm(total=len(tasks), unit="query")

    async def tracked(row, prop):
        try:
            async with sem:
                rec, ok, raw, usage = await run_one(row, prop)
            async with write_lock:
                write_record(rec, usage, ok)
            if not ok:
                LOG.warning("PARSE-FAIL %-28s %-22s raw=%s",
                            row["object_id"], prop, repr(raw)[:120])
        except Exception as exc:  # noqa: BLE001
            async with write_lock:
                failures.append((row["object_id"], prop, type(exc).__name__))
            LOG.error("FAIL %-28s %-22s %s: %s",
                      row["object_id"], prop, type(exc).__name__, exc)
        finally:
            progress.update(1)

    await asyncio.gather(*(tracked(row, prop) for row, prop in tasks))
    progress.close()

    write_json_export(args.jsonl_output, args.json_output)
    print_summary(totals, args)
    report_failures(failures)
    LOG.info("RUN end | completed=%d failed=%d tokens in=%d out=%d",
             totals["n"], len(failures), totals["prompt_tokens"], totals["completion_tokens"])
    LOG.info("full log saved -> %s", log_path)


def stop_on_failure(row, prop, kind, detail, raw, log_path, totals) -> None:
    LOG.error("STOP at first failure | %s | %s / %s | %s",
              kind, row["object_id"], prop, detail)
    print("\n" + "X" * 60)
    print(f"STOPPED at first failure  ({kind})")
    print("X" * 60)
    print(f"  object_id : {row['object_id']}")
    print(f"  tier      : {row['tier']}   shape: {row['shape_type']}   "
          f"vertices: {row['num_vertices']}")
    print(f"  property  : {prop}")
    print(f"  ground_truth: {row['properties'][prop]}")
    print(f"  detail    : {detail}")
    if raw is not None:
        print(f"  raw_model_output: {raw!r}")
    print(f"  wkt       : {row['wkt']}")
    print("-" * 60)
    print(f"  queries OK before stop: {totals['n']}  "
          f"(in={totals['prompt_tokens']} out={totals['completion_tokens']} tokens)")
    print(f"  log file  : {log_path}")
    print("X" * 60)
    print("Fix the issue, then re-run the SAME command:")
    print("  successful queries skip, the rest resume from here.")
    print("  (parse failures: add --retry-parse-failures to re-query them too.)")


def report_failures(failures) -> None:
    if not failures:
        print("\nAll queries completed. No failures.")
        return
    LOG.error("%d queries failed (not logged): %s",
              len(failures), [f"{o}/{p}" for o, p, _ in failures])
    print("\n" + "!" * 56)
    print(f"{len(failures)} queries FAILED (not logged) - re-run to retry ONLY these:")
    for oid, prop, err in failures[:30]:
        print(f"  {oid} / {prop}  ({err})")
    if len(failures) > 30:
        print(f"  ... and {len(failures) - 30} more")
    print("!" * 56)
    print("Re-run the same command - logged queries skip, failed ones retry.")


def write_json_export(jsonl_path: Path, json_path: Path) -> None:
    """Export JSONL as a JSON array, keeping the LATEST record per (object_id, property)."""
    if not jsonl_path.exists():
        return
    latest = read_latest_records(jsonl_path)
    records = sorted(latest.values(), key=lambda r: (r["object_id"], r["property"]))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n")


def print_summary(totals, args) -> None:
    n = totals["n"]
    if n == 0:
        print("\nNo new queries ran (all already logged). Use --force to re-run.")
        return
    pin, pout = totals["prompt_tokens"], totals["completion_tokens"]
    cost = pin / 1e6 * args.price_in + pout / 1e6 * args.price_out
    print("\n" + "=" * 56)
    print("V4-FLASH FULL RUN SUMMARY (real usage from API)")
    print("=" * 56)
    print(f"queries run            : {n}")
    print(f"parse success          : {totals['parsed_ok']}/{n}"
          f"  ({100*totals['parsed_ok']/n:.0f}%)")
    print(f"input tokens (total)   : {pin:,}  (avg {pin/n:.0f}/query)")
    print(f"output tokens (total)  : {pout:,}  (avg {pout/n:.0f}/query)")
    print(f"price                  : ${args.price_in}/1M in, ${args.price_out}/1M out")
    print(f"REAL cost              : ${cost:.4f}")
    print("=" * 56)
    print(f"jsonl: {args.jsonl_output}")
    print(f"json : {args.json_output}")
    LOG.info("COST n=%d parse_ok=%d in_tok=%d out_tok=%d real_cost=$%.4f",
             n, totals["parsed_ok"], pin, pout, cost)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

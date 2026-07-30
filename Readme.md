# Experiment 1 — Domain 1 (Geometry)

Does serialization format cost a model accuracy? This repository holds the
Geometry domain of Experiment 1: a 300-polygon benchmark, six model runs over
it, and the evaluation of those runs.

Reference spec: `serialization_experiment_1.pdf`.

## Pipeline

| Phase | Folder | What it does | Docs |
|-------|--------|--------------|------|
| 1 | `phase1_dataset/` | Generates the 300-polygon dataset with 9 ground-truth properties | [README](phase1_dataset/README.md) |
| 2 | `phase2_model_results/` | Queries 6 models — 9 properties per polygon — and scores the answers | [README](phase2_model_results/README.md) |
| 3 | `phase3_evaluation/` | Aggregates the Phase-2 records into metrics and figures | [README](phase3_evaluation/README.md) |

Each phase consumes the previous one's output files, so they run in order. Phase
1 is fully offline; only Phase 2 calls APIs.

---

## Setup

Covers **all three phases** — one virtual environment, one install.

Python 3.10 or newer is required (the Phase-2 runners use `list[dict[str, Any]]`
/ `X | Y` type syntax).

```bash
cd /path/to/EXPERIMENT1_PHASE1-2

python3 -m venv .venv                 # create the virtual environment
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers every phase:

| Package | Used by |
|---------|---------|
| `jupyter>=1.0` | Phase 1, Phase 3 — running the notebooks |
| `shapely>=2.0` | Phase 1 — geometry construction and ground truth |
| `matplotlib>=3.7` | Phase 1 — the spot-check figure; Phase 3 — the figures |
| `openai>=1.30` | Phase 2 — the API client (all providers) |
| `python-dotenv>=1.0` | Phase 2 — loads `.env` |
| `tenacity>=8.2` | Phase 2 — retry with exponential backoff |
| `tqdm>=4.66` | Phase 2 — progress bars |
| `numpy>=1.24` | Phase 3 — metric aggregation |
| `scipy>=1.10` | Phase 3 — statistics |

### API keys

Only Phase 2 calls APIs, but the credentials are set up here so both phases are
ready in one pass:

```bash
cp .env.example .env       # then edit .env and fill in the keys you need
```

| Key | Provider | Used by |
|-----|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek direct | `01_v4flash`, `06_v4pro_nonthinking` |
| `OPENROUTER_API_KEY` | OpenRouter | `02_qwen3`, `03_llama_scout` |
| `GEMINI_API_KEY` | Google | `04_gemini` |
| `OPENAI_API_KEY` | OpenAI | `05_gpt` |

You only need the key for a provider you actually intend to run. A missing key
fails at startup with `<KEY_NAME> not set. Add it to .env.`

`.env` is gitignored; `.env.example` is not. Never commit real keys.

The Phase-2 runners call `load_dotenv()` with no argument, which searches upward
from the script's own directory — so `.env` at this root is found from inside any
runner folder.

### Verify the setup

Two commands, both free and neither needing an API key:

```bash
# Phase 1 — open the notebook and run the cells in order
jupyter notebook phase1_dataset/Geometry_Experiment1_Phase1.ipynb

# Phase 2 — build prompts without calling anything
cd phase2_model_results/01_v4flash
python run_v4flash_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4flash_results.jsonl \
  --json-output v4flash_results.json \
  --dry-run
```

The Phase-2 dry run prints three example prompts and the planned query count.
With the committed results in place it reports `skipping 2700 already done` —
which also confirms the resume logic is reading the existing result files.

The two output flags are required for that: without them the script looks in a
`results/` subfolder that does not exist, finds no history, and reports 2,700
queries planned. See `phase2_model_results/README.md` §5 and §9 for why.

### Phase 3 dependencies

Phase 3 also ships its own `phase3_evaluation/requirements.txt` (numpy, scipy,
matplotlib) so that folder stays self-contained. Those packages are included in
the root `requirements.txt` above, so installing once at the root is enough — the
per-folder file is only needed if Phase 3 is run in isolation.

---

## Where to go next

- Regenerating or porting the dataset → [`phase1_dataset/README.md`](phase1_dataset/README.md)
- Running the models, or porting the query harness → [`phase2_model_results/README.md`](phase2_model_results/README.md)

Both documents end with a porting section describing what is domain-agnostic and
what a new domain has to replace.

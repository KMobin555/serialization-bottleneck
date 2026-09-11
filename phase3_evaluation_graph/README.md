# Phase 3 — Evaluation (Experiment 1, Domain 2: Graphs)

Evaluation layer for Experiment 1's graph domain. Consumes the Phase 2 model
result logs and produces the accuracy tables, statistical tests, failure
analysis, figures, and the Section 9 decision readout.

**Pipeline position**

```
phase1_dataset_graph/    ->  phase2_model_results_graph/  ->  phase3_evaluation_graph/
300 edge-list graphs         6 models x 8 properties           tables, tests, figures,
+ ground truth                query logs (.json/.jsonl)        decision readout
```

Phase 3 reads Phase 2's `*_results.json` logs **in place** from
`../phase2_model_results_graph/`. Nothing is copied or duplicated into this
folder.

**No production data exists yet.** As of this commit, none of the six models
have been queried against a real API (see
`../phase2_model_results_graph/README.md`) — `phase2_model_results_graph/`
has no `*_results.json` files. This notebook's logic was built and verified
against synthetic mock data (schema-correct, realistic error rates, not
committed anywhere in this repo) so that it is ready to run the moment real
Phase 2 output exists; running it for real is just re-executing the cells
against that output.

---

## What is evaluated

| | |
|---|---|
| Objects | 300 graphs (100 simple / 100 medium / 100 hard) |
| Properties | 8 — 2 local (`degree_of_node_0`, `edge_count`) + 6 global |
| Models | 5 capped direct-answer models @ 2,400 queries each |
| | DeepSeek-V4-Pro (non-thinking) @ 480 queries (20% stratified subsample) |
| Grading | `avg_clustering` @ 1% / 5% / 10% relative-error tolerance; every other property is exact match (integer or boolean) |
| Statistics | 10,000-resample bootstrap CIs and two-sided tests, seed 0 |

**Local vs global** is the central contrast, same as the geometry domain:
local properties can be read straight off the edge-list text (scan for lines
containing node `0`; count/read the header), global properties must be
computed from the whole graph structure.

**A real domain difference from Geometry, not a simplification:** only 1 of
this domain's 8 properties (`avg_clustering`) is numeric-tolerance-graded.
Geometry has 6 (`bbox`, `centroid`, `area`, `perimeter`, `aspect_ratio`,
`edge_length_variance`). The other 7 graph properties are int or bool with no
meaningful "close but not exact" — `is_bipartite` is either right or wrong,
`chromatic_number` is either right or wrong. This shows up structurally in
the notebook: Cell 5 (tolerance-band sweep) and Fig 5 (error-distribution
KDE) are single-property here instead of Geometry's six-property tables/grids.

---

## How to run

The notebook `Graph_Experiment1_Phase3.ipynb` contains the same code split
into cells, adapted from
[`../phase3_evaluation/Geometry_Experiment1_Phase3.ipynb`](../phase3_evaluation/Geometry_Experiment1_Phase3.ipynb)
— same statistical machinery (bootstrap CIs, gap analysis, Spearman
consistency, significance tests), this domain's property set and figures.
In Colab, if `../phase2_model_results_graph/` is not present it falls back
to a manual file-upload prompt for the six result logs.

```bash
source .venv/bin/activate   # from the repo root
jupyter notebook phase3_evaluation_graph/Graph_Experiment1_Phase3.ipynb
```

Cell 2 will report "no results found" until at least one model's
`*_results.json` exists under `../phase2_model_results_graph/<NN_model>/`.

---

## Outputs (`results/`, `figures/`)

Created on first run — neither folder is committed yet, since no run has
produced real output.

| File | Contents |
|---|---|
| `results/evaluation_summary.txt` | Headline findings and the Section 9 decision |
| `results/v4flash_failure_analysis.json` | Per-failure heuristic labels + blank `manual_label` field |
| `figures/fig1_accuracy_by_property.png` | Accuracy by property and tier, one panel per model |
| `figures/fig2_accuracy_vs_complexity.png` | Accuracy vs tier for the 4 computed global properties |
| `figures/fig3_local_global_gap_heatmap.png` | Local−global gap, models × tier |
| `figures/fig4_accuracy_vs_nodecount.png` | Accuracy vs node count for `triangle_count` / `diameter` / `chromatic_number` |
| `figures/fig5_error_histogram.png` | Relative-error KDE distribution for `avg_clustering` (the domain's one numeric-tolerance property) |

---

## What differs from the Geometry domain's Phase 3

Everything *structural* (bootstrap methodology, CI level, significance
tests, the local/global gap definition, the Section 9 decision thresholds)
is unchanged — Section 6 of the PDF states these apply uniformly across all
three domains. What's genuinely different:

| | Geometry | Graphs |
|---|---|---|
| Properties | 9 (2 local, 7 global) | 8 (2 local, 6 global) |
| Numeric-tolerance properties | 6 | 1 (`avg_clustering`) |
| Boolean properties | 1 (`convex`) | 2 (`is_bipartite`, `is_planar`) |
| Categorical properties | 1 (`orientation`, cw/ccw) | 0 |
| Complexity axis | vertex count (3–40) | node count (6–80) |
| Category field | `shape_type` (3 values) | `family` (5 values) |
| Confusion-matrix cell (7) | 1 boolean + 1 categorical | 2 boolean, same treatment |
| Fig 5 layout | 2×3 grid (6 numeric properties) | single panel (1 numeric property) |

**Intentionally *not* carried over:** Geometry's `vertex_count`
off-by-one-from-the-closing-coordinate adjustment (Fig 3's footnote
mechanism, `offbyplus1_rate`/`adjusted_gap` in Geometry's Cell 13). That is a
specific artifact of WKT's repeated closing coordinate, with no equivalent in
an edge list, so there's nothing analogous to adjust the *gap figure* for.

**A different off-by-one *does* apply to this domain's failure taxonomy,
though:** `diameter` has two common definitions — edges on the longest
shortest path (this dataset's ground truth, `nx.diameter`) vs. nodes on that
path (= edges + 1). Cell 14's `classify_failure()` labels a `diameter` miss
of exactly `absolute_error == 1` as `off_by_one_definition` rather than
`arithmetic_error`, the direct graph-domain instance of Table 14's general
category (whose only PDF-given example is geometry's closing-coordinate
case).

---

## Known limitations (same caveats as Geometry, where applicable)

- **Not yet run against real data.** Every table/figure/number this notebook
  would produce is unverified against an actual model until Phase 2 runs
  complete. The mock-data test run (not committed) only confirms the *code*
  is correct — it says nothing about what the real models will actually do.
- **Failure labels are heuristic**, same discipline as Geometry:
  `classify_failure()` in Cell 14 is a first-pass labeller to speed up the
  manual pass the protocol requires. `manual_label` is deliberately left
  `null` in the exported JSON.
- **This is the second of three domains.** The Section 9 rule needs at least
  2 of 3 domains, so the final go/no-go is deferred until this notebook has
  real results *and* the tabular domain is evaluated.
- **`is_planar`'s accuracy-rises-with-tier artifact is now fully fixed.**
  An earlier build closed this for `is_bipartite` only (Phase 1's
  `random_bipartite`/`random_planar` families each have an exact internal
  50/50 split on their own named boolean property, see
  `../phase1_dataset_graph/README.md` §1) but left `is_planar`'s *overall*
  true-rate swinging by tier (42% simple / 13% medium / 10% hard), because
  small sparse graphs from the *other* 4 families were incidentally planar
  far more often than large ones — an effect living entirely outside
  `random_planar`, so balancing only that family couldn't correct it.
  Phase 1 now corrects every such incidental hit after generation
  (`try_force_nonplanar` / `try_force_nonbipartite_preserve_planar`, §1/§2
  there), so `is_bipartite=True` and `is_planar=True` come *only* from
  `random_bipartite`/`random_planar` respectively, in every tier, with
  certainty. Both properties' overall true-rate is now exactly flat (11.0%,
  10.0%) across all 3 tiers, and the artifact is gone in the data that
  mattered most: Qwen3-32B and Llama4-Scout's `is_planar` accuracy is now
  **90.0% / 90.0% / 90.0%** across simple/medium/hard — flat, matching the
  majority-class baseline exactly, instead of the previous 68%→87%→90%
  rise. (Gemini and GPT-4.1-mini still show some tier-to-tier accuracy
  variation on `is_planar`, but that's now genuine model behavior — they
  actually vary their True/False predictions by tier rather than defaulting
  to one answer — not a base-rate artifact.)
- **V4-Pro (non-thinking)'s subsample had the same artifact, one layer
  down.** The full dataset being exactly 50/50 on `is_bipartite`/`is_planar`
  within their named families doesn't guarantee a random *20% draw* from it
  preserves that balance — `build_subsample.py`'s original stratification
  was only by `(tier, family)`, so a 4-of-22 or 4-of-20 draw from
  `random_bipartite`/`random_planar` could land on (and did: observed 4/4
  True, 0 False for `medium/random_bipartite` and `hard/random_planar`) a
  badly skewed true/false mix purely by chance, independent of how balanced
  the source family was. Fixed by adding a second stratification level:
  each of those two families' quota is now split in half between its own
  named boolean property before drawing (`SPLIT_FAMILIES` in
  `../phase2_model_results_graph/06_v4pro_nonthinking/build_subsample.py`),
  so the subsample is exactly 2 True + 2 False per property per tier — same
  overall family-level counts and total (60/300) as before, just internally
  balanced now. V4-Pro's remaining per-tier accuracy variation (e.g. its own
  True-prediction rate drops from 70% in simple to 0–5% in medium/hard) is
  therefore now attributable to the model's actual behavior, not to which
  20 graphs happened to get drawn.

---

## Source notebook

Adapted from [`../phase3_evaluation/Geometry_Experiment1_Phase3.ipynb`](../phase3_evaluation/Geometry_Experiment1_Phase3.ipynb).
See that notebook's own README for the original Colab source link.

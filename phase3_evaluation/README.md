# Phase 3 — Evaluation (Experiment 1, Domain 1: Geometry)

Evaluation layer for Experiment 1. Consumes the Phase 2 model result logs and
produces the accuracy tables, statistical tests, failure analysis, figures, and
the Section 9 decision readout.

**Pipeline position**

```
phase1_dataset/          ->  phase2_model_results/      ->  phase3_evaluation/
300 WKT polygons             6 models x 9 properties        tables, tests, figures,
+ ground truth               query logs (.json/.jsonl)      decision readout
```

Phase 3 reads Phase 2's committed `*_results.json` logs **in place** from
`../phase2_model_results/`. Nothing is copied or duplicated into this folder.

---

## What is evaluated

| | |
|---|---|
| Objects | 300 polygons (100 simple / 100 medium / 100 hard) |
| Properties | 9 — 2 local (`vertex_count`, `bbox`) + 7 global |
| Models | 5 capped direct-answer models @ 2700 queries each |
| | DeepSeek-V4-Pro @ 540 queries (20% stratified subsample, thinking disabled) |
| Grading | numeric @1% / 5% / 10% relative-error tolerance; boolean & categorical exact match |
| Statistics | 10,000-resample bootstrap CIs and two-sided tests, seed 0 |

**Local vs global** is the central contrast: local properties can be read
straight off the serialized text, global properties must be computed from it.

---

## How to run

The notebook `Geometry_Experiment1_Phase3.ipynb` contains the same code split
into cells. In Colab, if `../phase2_model_results/` is not present it falls back
to a manual file-upload prompt for the six result logs.

---

## Outputs (`results/`)

| File | Contents |
|---|---|
| `evaluation_summary.txt` | Headline findings and the Section 9 decision |
| `phase3_console_output.txt` | Every table printed by the run (accuracy matrices, CIs, tolerance sweep, error quantiles, confusion matrices, parse audit, gap, gradient, Spearman, significance tests) |
| `v4flash_failure_analysis.json` | Per-failure heuristic labels + blank `manual_label` field |
| `fig1_accuracy_by_property.png` | Accuracy by property and tier, one panel per model |
| `fig2_accuracy_vs_complexity.png` | Accuracy vs tier for the computed global properties |
| `fig3_local_global_gap_heatmap.png` | Local−global gap, models × tier |
| `fig4_accuracy_vs_vertexcount.png` | Accuracy vs vertex count for area / perimeter / centroid |
| `fig5_error_histograms.png` | Relative-error KDE distributions per numeric property |

---

## Findings

Exact figures are in `results/evaluation_summary.txt` — regenerated on every run,
never transcribed by hand.

1. **Large, significant local–global accuracy gap in every capped model.** All
   gaps far exceed the 15 pp threshold, with non-overlapping bootstrap CIs.
2. **Compute failure, not read failure.** `bbox` (directly readable) stays high;
   `area` and `edge_length_variance` collapse toward zero. Loosening the
   tolerance from 1% to 10% does not rescue the computed globals.
3. **One shared bottleneck.** Cross-model Spearman correlation of the per-property
   difficulty ordering is high and uniform among the capped models.
4. **`convex` and `orientation` sit at or below their majority-class baselines**
   for most capped models — no real skill, just base-rate tracking.
5. **Free reasoning closes the gap.** DeepSeek-V4-Pro with an unconstrained token
   budget recovers the global properties, which places the bottleneck at
   single-pass / no-chain-of-thought inference rather than at fundamental
   inability.

---

## Known limitations

- **Failure labels are heuristic.** `classify_failure()` in the failure-mode cell
  is a first-pass labeller to speed up the manual pass the protocol requires. The
  `arithmetic_error` vs `hallucination` split is the least reliable, and
  `instruction_failure` is not auto-detected. `manual_label` is deliberately left
  `null` in the exported JSON.
- **The "local stays above 85%" clause is only partly satisfied.** `bbox` stays
  high, but `vertex_count` degrades with polygon size. That is a
  reading/counting-at-scale effect and is reported separately rather than folded
  into the global-computation story.
- **V4-Pro's token cap was not enforced.** Many answers exceeded the nominal 2048
  budget, all with `finish_reason='stop'` — free-thinking traces, not truncation.
  Flagged for the write-up and cost accounting.
- **V4-Pro's raw gap is understated** by its `vertex_count` off-by-+1 behaviour
  (it counts the repeated closing coordinate). Both the off-by-one rate and the
  adjusted gap are computed in the run and printed to the console log.
- **Geometry is one domain.** The Section 9 rule needs at least 2 of 3 domains,
  so the final go/no-go is deferred until graphs and tabular are evaluated.

---

## Source notebook

https://colab.research.google.com/drive/1BMTVLitcxoCpkNv0vC3I9hK6KUC14OKn

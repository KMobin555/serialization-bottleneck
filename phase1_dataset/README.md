# Phase 1 — Domain 1 (Geometry) Dataset Generation

Documentation for `Geometry_Experiment1_Phase1.ipynb`: what it builds, how it
builds it, what it guarantees, and what a person porting this to a new domain
has to replace.

Reference spec: `serialization_experiment_1.pdf`, Sections 3, 4.4, 7 (Table 4).

**Setup first.** Virtual environment, dependencies, and API keys are covered
once in the [root README](../Readme.md#setup) — it serves both phases. This
document assumes that is done.

---

## 1. What Phase 1 produces

Phase 1 is the **data generation and ground-truth stage**. It emits no model
calls — it only builds the benchmark that Phase 2 queries.

| File | Contents |
|------|----------|
| `geometry_exp1_dataset.json` | 300 polygon records: WKT string + 9 ground-truth properties + metadata |
| `geometry_exp1_summary.json` | Section 7 summary statistics (counts, per-tier distributions, orientation balance) |
| `spotcheck_exp1.png` | 3×3 grid, one polygon per (tier × shape), for visual sanity |

This extends the earlier serialization **pilot** (90 polygons, 7 properties).
Generator logic, validity constraints, and serialization are unchanged from the
pilot — only **scale** (90 → 300) and the **property set** (7 → 9) differ.

### Design matrix

| Tier | Count | Vertices | Coordinate range | Coordinate type |
|------|-------|----------|------------------|-----------------|
| simple | 100 | 3–8   | [0, 100]  | integer |
| medium | 100 | 10–20 | [0, 100]  | integer |
| hard   | 100 | 20–40 | [0, 1000] | float, 2 decimal places |

Each tier splits into 3 shape categories: **convex 34 / concave 33 / irregular 33**.
Total 3 × 100 = 300. The PDF allows approximate balance; 34/33/33 is that.

### The 9 properties (Table 4)

Seven carried over from the pilot, two new in Experiment 1:

| Property | Type | Evaluation mode | New in Exp 1 |
|----------|------|-----------------|--------------|
| `vertex_count` | int | exact match | |
| `bbox` | `[minx, miny, maxx, maxy]` | error normalized by bbox diagonal | |
| `centroid` | `[x, y]` | error normalized by bbox diagonal | |
| `area` | float | relative error | |
| `perimeter` | float | relative error | |
| `convex` | bool | exact match | |
| `orientation` | `"cw"` / `"ccw"` | exact match | |
| `aspect_ratio` | float | relative error | ✅ |
| `edge_length_variance` | float | relative error | ✅ |

`aspect_ratio` = bbox width / bbox height. Not max/min — so values below 1.0 are
normal and expected (observed min 0.15).

`edge_length_variance` = population variance (divide by *n*, not *n−1*) of all
edge lengths in the closed ring.

Both new properties are **orientation-independent**, which is why they are
computed in step 1 of the record builder, before the winding reversal.

---

## 2. Notebook walkthrough

The notebook has 17 cells in 8 sections. Order matters — later cells depend on
names defined earlier, and cell 10 is the one that actually runs everything.

### Cell 1 — Phase 0: environment
Installs `shapely` + `matplotlib`, imports, prints the Shapely version. Written
for Colab (`!pip install`), works locally too.

### Cell 3 — Section 1: core helpers

- **`round_coords(coords, coord_type)`** — integer rounding for simple/medium,
  2-decimal rounding for hard. Applied *before* validity checking, so a polygon
  is only accepted if it is still valid **after** rounding. This is important:
  rounding can collapse vertices and create degeneracies, and the pipeline
  refuses to store any polygon damaged that way.

- **`is_convex(poly)`** — a polygon equals its own convex hull iff it has no
  dents. `.normalize()` on both sides makes the comparison ignore vertex order
  and starting point.

- **`check_validity(poly, vmin, vmax, bound_lo, bound_hi)`** — the Section 3.2 /
  4.4 gate. Returns `(True, None)` or `(False, reason)`. Eight rules:

  | Rule | Rejection reason | Check |
  |------|------------------|-------|
  | 1 | `not_valid` | `poly.is_valid` |
  | 2 | `not_simple` | `poly.is_simple` (no self-intersection) |
  | 3 | `area_too_small` | area > 10 |
  | 4 | `duplicate_adjacent` | no two consecutive identical vertices |
  | 5 | `collinear` | every consecutive triple has cross product ≥ 1e-9 |
  | 5b | `sliver_low_fill` | area / bbox-area ≥ 0.05 (kills near-degenerate slivers) |
  | 6 | `vertex_count_out_of_range` | vmin ≤ n ≤ vmax |
  | 7 | `out_of_bounds` | every coordinate inside [lo, hi] |

  Rule 5b is the non-obvious one. Rules 1–5 pass for a polygon that is a
  visually useless near-line; the fill-ratio floor removes those.

### Cell 5 — Section 2: tiers and the three shape generators

`TIERS` holds `(vmin, vmax, lo, hi, coord_type, m_range)` per tier. `m_range` is
the count of random points fed to the hull generator — it is *not* the vertex
count, it is the input pool size the hull is taken from.

All three generators are **rejection samplers**: build a candidate, run
`check_validity`, retry up to `max_tries=3000`, return `None` on exhaustion.

- **`gen_convex`** — convex hull of *m* uniform random points. Vertex count is
  emergent, not controlled: a random hull over a uniform square yields roughly
  O(log m) vertices, which lands naturally in the simple/medium ranges but
  cannot reliably reach 20–40. Hence the Valtr generator below for the hard tier.

- **`gen_concave`** — radial/star method. Pick a center, pick `r_max ∈ [0.10,
  0.22] × span`, pick a concavity ratio `∈ [0.4, 0.8]` giving `r_min = ratio ×
  r_max`, sort *n* random angles, sample a radius per angle. The `[0.4, 0.8]`
  band produces *mild* concavity — deep spikes are deliberately avoided.
  Accepted only if `is_convex` is **false**, so concave records are guaranteed
  genuinely concave.

- **`gen_irregular`** — a concave base, then an affine pipeline: stretch one
  random axis by `factor ∈ [2.0, 5.0]`, scale down if the longest side exceeds
  0.9 × span, rotate by a random angle, translate back into bounds.

  Note: the base is generated with `coord_type="float_2dp"` regardless of tier,
  because rounding to integers before the affine transforms would compound
  rounding error. Final coordinates are rounded to the tier's type after the
  transforms.

  Scaling and rotation both preserve convexity, and the base is already
  non-convex, so irregular polygons are always non-convex too.

### Cell 6 — Valtr algorithm, dispatcher, exact-vertex helper

- **`_valtr_unit` / `gen_convex_valtr`** — Valtr's algorithm generates a convex
  polygon with **exactly** *n* vertices: build *n* edge vectors whose components
  sum to zero, sort them by angle, and walk them. Used only for the hard tier,
  where the 20–40 vertex requirement is unreachable by random hulls. Output is
  then scaled to `[0.5, 0.85] × span` and placed at a random offset.

- **`generate_one(rng, shape_type, tier)`** — the dispatcher. Only special case:
  `convex + hard` → Valtr; everything else → the plain generator.

- **`gen_convex_exact(rng, target_n, ...)`** — hull sampling that retries until
  the hull has exactly `target_n` vertices. Used only to force low-vertex
  coverage (see cell 10).

### Cell 8 — Section 3: ground truth and the record builder

**The step order here is a correctness requirement, not a style choice**
(Section 3.2):

1. **`compute_orientation_independent(poly)`** — computes the 8 properties that
   do not depend on winding direction, plus `bbox_diagonal`.
2. **`maybe_reverse(poly, rng)`** — with probability 0.5, reverses the
   coordinate ring. This is what makes `orientation` a non-trivial question for
   the model rather than a constant.
3. **`to_wkt(poly, coord_type)`** — serializes to `POLYGON((x y, x y, ...))`,
   formatting integers as integers and floats as `%.2f`, matching the tier.
4. **`get_orientation(poly)`** — reads `cw`/`ccw` from the **possibly reversed**
   ring, so the label matches the string the model will actually see.

Computing orientation-dependent truth before step 2, or serializing before step
2, would silently desynchronize the label from the WKT.

`bbox_diagonal` is stored in metadata because Phase 2 uses it as the normalizer
for `bbox` and `centroid` errors — those are absolute-distance quantities, and
without normalization a hard-tier polygon (coords up to 1000) would dominate a
simple-tier one (coords up to 100) purely by scale.

All stored floats are rounded to 4 decimal places.

**Record schema:**

```json
{
  "object_id": "poly_simple_convex_001",
  "tier": "simple",
  "shape_type": "convex",
  "num_vertices": 3,
  "wkt": "POLYGON((15 13, 62 59, 90 32, 15 13))",
  "properties": {
    "vertex_count": 3,
    "bbox": [15.0, 13.0, 90.0, 59.0],
    "centroid": [55.6667, 34.6667],
    "area": 1278.5,
    "perimeter": 182.0313,
    "convex": true,
    "aspect_ratio": 1.6304,
    "edge_length_variance": 259.6238,
    "orientation": "cw"
  },
  "metadata": {
    "coordinate_type": "integer",
    "is_valid": true,
    "is_simple": true,
    "orientation_was_reversed": false,
    "random_seed": 42,
    "bbox_diagonal": 87.983
  }
}
```

`object_id` format: `poly_{tier}_{shape_type}_{index:03d}`. Index restarts at
001 within each (tier, shape) group, so the id is unique only as a triple —
which it is, since tier and shape are both in the string.

### Cell 10 — Section 4: build the dataset and summary

`build_dataset(seed=42)` iterates tier → shape → index and calls the dispatcher.

One special case: in the **simple + convex** group, indices 1 and 2 are forced
to exactly 3 vertices and indices 3 and 4 to exactly 4 vertices via
`gen_convex_exact`. Without this, random hulls almost never produce triangles or
quadrilaterals, and the lowest-complexity end of the benchmark would be empty.

Generation failure is **fatal**: if a generator exhausts 3000 tries and returns
`None`, `build_dataset` raises `RuntimeError` rather than emitting a short
dataset. Silent under-generation would be worse than a crash.

`summarize(records)` produces `geometry_exp1_summary.json`:

- `total` — 300
- `counts_by_tier_shape` — the 34/33/33 grid per tier
- `convex_overall` — count where `properties.convex` is true
- `distribution_by_tier` — min/max/mean/median/std for `vertex_count`, `area`,
  `perimeter`, `bbox_diagonal`, `aspect_ratio`, `edge_length_variance`, and
  `wkt_length`
- `orientation_balance` — cw/ccw counts per tier

`stats_for` computes population std (divide by *n*).

`wkt_length` is tracked because it is the serialization-length proxy the
experiment cares about — it is the input token pressure per record.

### Cell 12 — Section 5: independent verification

This cell is the reason the dataset can be trusted, and it should be kept in any
port.

It re-parses each record **from the stored WKT string** with `parse_wkt` (a
regex number scan, no Shapely) and recomputes properties with hand-written
implementations — shoelace area, pairwise-distance perimeter, bbox aspect, edge
variance — then compares against the stored ground truth. A bug shared between
the generator and the ground-truth function would survive a self-check; it does
not survive an independent reimplementation reading only the serialized output.

Tolerances (absolute, unless noted):

| Property | Tolerance | Why not exact |
|----------|-----------|---------------|
| `vertex_count` | 0 | integer, must match exactly |
| `area` | 1.0 | coordinates were rounded after the polygon was built |
| `perimeter` | 0.5 | same |
| `aspect_ratio` | 0.02 | same |
| `edge_length_variance` | `max(1.0, 2% of truth)` | variance scales quadratically, so a relative floor is needed on the hard tier |

The cell also asserts no record has `is_valid == false` or `is_simple == false`,
and prints the property key list.

**Known gap:** verification covers 5 of the 9 properties. `bbox`, `centroid`,
`convex`, and `orientation` are not independently re-derived. `bbox` and
`centroid` are cheap to add (min/max scan and the shoelace centroid formula);
`convex` and `orientation` would need an independent cross-product sign test.
Worth closing in a port.

### Cell 14 — Section 6: visual spot-check

Plots the **first** record of each (tier × shape) pair into a 3×3 grid and saves
`spotcheck_exp1.png` at 110 dpi. Purely an eyeball check: convex shapes should
look convex, concave ones should have visible dents, irregular ones should look
stretched and rotated. Not an automated assertion.

### Cell 16 — Section 7: Colab download

Wrapped in `try/except` — on Colab it triggers browser downloads, locally it
prints a message and does nothing. Files are already written to the working
directory either way.

---

## 3. Reproducibility

- Single seed, `SEED = 42`, passed to one `random.Random(seed)` instance that
  drives every generator, the winding reversal, and the exact-vertex helper.
- The seed is stored in **every record** (`metadata.random_seed`), not only in
  the summary.
- Same seed + same Shapely version ⇒ byte-identical dataset. Shapely version
  matters because hull tie-breaking and `normalize()` behaviour are library
  internals; the notebook prints `shapely.__version__` in cell 1 for exactly
  this reason. Record it when reporting results.
- Cells must be run **in order**. Cell 10 depends on every definition above it,
  and cells 12 and 14 depend on the in-memory `records` list from cell 10.
- Notebook outputs are stripped in the committed version — the JSON artifacts in
  this folder are the record of what was produced.

## 4. Observed dataset characteristics (seed 42)

From `geometry_exp1_summary.json`:

- **convex_overall = 102 / 300** (34%). Exactly the 34 convex polygons per tier
  × 3 tiers — confirming that concave and irregular records are never
  accidentally convex, as the generators enforce.
- **Orientation balance** — simple 55/45, medium 53/47, hard 46/54 (cw/ccw).
  Near 50/50 as expected from a p=0.5 coin, with normal sampling noise.
- **Scale separation across tiers** is large and intentional:

  | Tier | mean vertices | mean area | mean WKT length |
  |------|---------------|-----------|-----------------|
  | simple | 6.66 | 2,341 | 61 chars |
  | medium | 13.84 | 3,267 | 110 chars |
  | hard | 29.86 | 169,388 | 470 chars |

  The area jump at the hard tier comes from the coordinate range change
  ([0, 1000] vs [0, 100]) — a ~100× area effect — not from the vertex count.
  This is why every scale-dependent metric in Phase 2 is normalized.

- `aspect_ratio` means sit near 1.0–1.2 in all tiers with min ≈ 0.15 and max
  ≈ 4.7, i.e. the irregular stretch is visible in the tails, not the center.

## 5. How Phase 2 consumes this

Phase 2 runners (`../phase2_model_results/*/run_*_full.py`) read
`geometry_exp1_dataset.json` and use exactly four things per record:

| Field | Used for |
|-------|----------|
| `object_id` | resume key — `(object_id, property)` pairs already in the JSONL are skipped |
| `wkt` | the serialized string embedded in the prompt |
| `properties[prop]` | ground truth for scoring the model's answer |
| `metadata.bbox_diagonal` | normalizer for `bbox` and `centroid` errors |

`tier` and `shape_type` ride along into the result records for slicing at
analysis time. Everything else in `metadata` is provenance.

The practical consequence: **the schema is the interface**. A new domain that
emits records with these key names plugs into the existing Phase 2 harness
without touching the runner's I/O, resume, retry, or logging code.

---

## 6. Porting this to a new domain

What is domain-specific here and what is not:

**Reusable structure — keep the shape, change the content:**

- The tier system (3 difficulty levels defined by a complexity parameter and a
  magnitude range).
- The category split within each tier (3 balanced structural classes).
- Rejection sampling against an explicit, enumerated validity rule set, where
  every rejection has a named reason.
- Rounding/serialization applied *before* the validity check, so the stored
  artifact is what was validated.
- Compute-invariant-properties → randomize-presentation → serialize →
  read-presentation-dependent-property. Any domain with a property that depends
  on how the object is written down needs this ordering.
- Independent verification from the serialized string with an implementation
  that does not share code with the generator.
- The record schema and the summary statistics block.

**Domain-specific — must be rewritten:**

| Component | Geometry version | What a port replaces it with |
|-----------|------------------|------------------------------|
| Object type | Shapely `Polygon` | the new domain's object |
| Serialization | WKT `POLYGON((...))` | SMILES, adjacency list, CSV row, … |
| `TIERS` | vertex ranges + coordinate bounds | the domain's complexity axis |
| Generators | hull / radial / affine-stretch | domain-appropriate samplers |
| `check_validity` | the 8 geometry rules | the domain's validity rules |
| Property functions | area, perimeter, centroid, … | the domain's Table-4 properties |
| Presentation randomizer | `maybe_reverse` (winding) | whatever presentation choice is arbitrary in that domain |
| Normalizer | `bbox_diagonal` | the domain's scale normalizer, if scale-dependent properties exist |
| Independent verifier | shoelace / regex WKT parse | an independent reimplementation for the new format |

**Order of work for a port:**

1. Fix the record schema first — keep the top-level key names.
2. Define tiers and the validity rule set before writing any generator.
3. Write generators as rejection samplers against that rule set.
4. Write ground-truth functions, splitting presentation-independent from
   presentation-dependent, and enforce the 4-step build order.
5. Write the independent verifier *before* trusting any output. Cover more
   properties than this notebook does.
6. Reuse the summary and spot-check cells with the new property names.

Phase 2 needs no changes beyond its own domain-specific hooks — see the runner
scripts for those.

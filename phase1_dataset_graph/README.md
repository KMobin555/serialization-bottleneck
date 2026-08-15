# Phase 1 — Domain 2 (Graphs) Dataset Generation

Documentation for `Graph_Experiment1_Phase1.ipynb`: what it builds, how it
builds it, what it guarantees, and what a person porting this to a new domain
has to replace.

Reference spec: `serialization_experiment_1.pdf`, Section 4.

**Setup first.** Virtual environment, dependencies (including `networkx`,
added to the root `requirements.txt` for this domain), and API keys are
covered once in the [root README](../Readme.md#setup). This document assumes
that is done.

---

## 1. What Phase 1 produces

Phase 1 is the **data generation and ground-truth stage**. It emits no model
calls — it only builds the benchmark that Phase 2 queries.

| File | Contents |
|------|----------|
| `graph_exp1_dataset.json` | 300 graph records: edge-list string + 8 ground-truth properties + metadata |
| `graph_exp1_summary.json` | Summary statistics (counts, per-tier distributions, bipartite/planar balance) |
| `spotcheck_exp1_graph.png` | 3×5 grid, one graph per (tier × family), for visual sanity |

This is the second domain of Experiment 1, sibling to
[`phase1_dataset/`](../phase1_dataset/README.md) (Geometry). It follows the
same methodology — tiered rejection sampling against a named validity rule
set, a presentation-independent/presentation-dependent property split,
independent verification from the serialized string — adapted to graphs.

### Design matrix

| Tier | Count | Nodes | Purpose |
|------|-------|-------|---------|
| simple | 100 | 6–15  | Baseline; small enough for manual verification |
| medium | 100 | 16–40 | Core measurement |
| hard   | 100 | 41–80 | Stress test |

Each tier splits into 5 graph families. The PDF's target is 20/family/tier
(range 18–22, "exact balance is not required"); the committed split is
**erdős_renyi 20 / barabasi_albert 19 / watts_strogatz 19 / random_bipartite
22 / random_planar 20** — 100 per tier, 300 total. See §3 for why the split
is not perfectly even.

### The 8 properties (Table 8)

| Property | Type | Locality | Ground Truth | Eval |
|----------|------|----------|---------------|------|
| `degree_of_node_0` | int | local | `G.degree(0)` | exact |
| `edge_count` | int | local | `G.number_of_edges()` | exact |
| `triangle_count` | int | global | `sum(nx.triangles(G).values()) // 3` | exact |
| `is_bipartite` | bool | global | `nx.is_bipartite(G)` | exact |
| `is_planar` | bool | global | `nx.check_planarity(G)[0]` | exact |
| `diameter` | int | global | `nx.diameter(G)` | exact |
| `chromatic_number` | int | global | exact computation (§4) | exact |
| `avg_clustering` | float | global | `nx.average_clustering(G)` | rel. error |

`degree_of_node_0` and `edge_count` are "local" because both are readable
from a bounded window of the edge-list string: scan for lines containing `0`
as an endpoint, or read/count lines. The other six require reasoning over
the whole structure — this local/global split is the axis the experiment is
built to measure (implicit-structure bottleneck).

---

## 2. Notebook walkthrough

The notebook has 18 cells in 8 sections. Cells must run in order — later
cells depend on names defined earlier, and cell 11 is the one that actually
runs generation.

### Cell 1 — Phase 0: environment
Installs `networkx` + `scipy` + `matplotlib`, imports, prints the NetworkX
version. Written for Colab (`!pip install`), works locally too.

### Cell 3 — Section 1: tiers, families, validity

`TIERS` holds `(vmin, vmax)` node-count bounds per tier — no coordinate axis
here, unlike geometry; a graph's complexity axis is node count alone.

`PER_FAMILY` fixes the per-tier count for each of the 5 families.
`check_validity(G, vmin, vmax)` is the Section 4.2 gate, returning
`(True, None)` or `(False, reason)`:

| Rule | Rejection reason | Check |
|------|-------------------|-------|
| 1 | `node_count_out_of_range` | `vmin <= n <= vmax` |
| 2 | `self_loop` | no self-loops |
| 3 | *(structural)* | no multi-edges — guaranteed by using `nx.Graph`, never `nx.MultiGraph` |
| 4 | `not_connected` | single connected component |
| 5 | `too_few_edges` | `m >= n - 1` |
| 6 | `too_many_edges` | `m <= C(n,2)/2` |

### Cell 5 — Section 2: the five family generators

All five are **rejection samplers**, same pattern as geometry: draw a fresh
`n` from the tier range on every attempt (not fixed by the caller), build a
candidate, run `check_validity`, retry up to `max_tries=3000`.

- **`gen_erdos_renyi`** — `nx.gnp_random_graph(n, p)` with `p` derived from a
  target expected degree drawn uniformly from `[3, 6]`.
- **`gen_barabasi_albert`** — `nx.barabasi_albert_graph(n, m)`,
  `m ∈ {2, 3, 4}`, skipped if `m >= n`.
- **`gen_watts_strogatz`** — `nx.watts_strogatz_graph(n, k, p)`,
  `k ∈ {4, 6}`, `p ∈ {0.1, 0.3, 0.5}`, skipped if `k >= n`.
- **`gen_random_bipartite`** — `nx.bipartite.random_graph(n1, n2, p)` with
  `n1 + n2 = n` and the ratio **between the two partitions**, `n1/n2`,
  constrained to `[0.3, 0.7]` (PDF Table 7 — this is a ratio between
  partition sizes, not either partition's share of `n`). Found by exact
  integer search over `n`'s valid splits rather than rounding a continuous
  target, since rounding can overshoot the bound for small `n` — e.g. `n=7`
  admits only the single split `(2, 5)` (ratio `0.4`); naive rounding can
  land on `(3, 4)` (ratio `0.75`, outside the bound).
- **`gen_random_planar`** — Delaunay triangulation of `n` random points
  (always planar — any subgraph of a planar graph is planar too), then edges
  are randomly thinned toward a target within the `3n-6` planar bound,
  reverting any removal that would disconnect the graph.

All generators pass the shared `rng` (a single `random.Random(seed)`
instance) directly as NetworkX's `seed` parameter — NetworkX accepts a
`random.Random` instance natively, so one seed stream drives every family,
every parameter draw, and the later relabeling step, exactly as geometry
drives everything from one `rng`.

### Cell 7 — Section 3: chromatic number

NetworkX has no exact chromatic-number function. The implementation follows
the PDF's specified fallback chain:

1. **Clique certificate, fast path** — if the greedy `DSATUR` upper bound
   equals the size of a maximum clique (`nx.find_cliques`), the chromatic
   number is certified immediately: a clique of size *k* forces at least *k*
   colors, and greedy already achieves *k*.
2. **Exact backtracking, bounded** — otherwise, search increasing *k* with
   DSATUR vertex ordering and color-symmetry breaking (never open a color
   number more than 1 past the highest used so far), time-boxed at 15s per
   graph.
3. **Uncertified fallback** — if the time box is hit before the gap closes,
   the graph's chromatic number is excluded from evaluation
   (`chromatic_number_certified: false` in metadata) rather than reporting
   an approximate value, per the PDF: *"Do not use approximate values."*

Verified against 7 known graphs (K5, Petersen, C5, C6, K3,3, K4, Star_10)
before trusting it on the dataset — execution checklist item 2. On the
committed run: **1 of 300 graphs is uncertified**
(`graph_hard_watts_strogatz_008`, closing the gap wasn't possible within the
15s time box) — excluded from querying and evaluation per the PDF's
instruction, everything else closed via the clique certificate or
backtracking.

### Cell 9 — Section 3 (continued): ground truth and the record builder

**The step order here is a correctness requirement**, the same principle as
geometry's winding-reversal ordering:

1. **`compute_presentation_independent(G)`** — the 6 properties that do not
   depend on node labeling: `triangle_count`, `is_bipartite`, `is_planar`,
   `diameter`, `chromatic_number`, `avg_clustering`.
2. **`randomize_labeling(G, rng)`** — relabel nodes with a random permutation
   of `0..n-1`.
3. **`to_edge_list_string(G)`** — serialize the *relabeled* graph, edges
   sorted `(min(u,v), max(u,v))` then lexicographically (Section 4.3).
4. **`degree_of_node_0`, `edge_count`** — read from the relabeled graph, so
   the label matches the string the model will actually see.

**Why the relabeling step exists.** The PDF does not explicitly ask for it,
but it is required by the same principle the geometry README states
explicitly: *"Any domain with a property that depends on how the object is
written down needs [compute-invariant → randomize-presentation → serialize →
read-presentation-dependent-property]."* Without it, node `0` would be
whatever label the generator's internals happened to assign — most visibly,
Barabási–Albert's earliest nodes are structurally the hubs — so
`degree_of_node_0` would measure a generator artifact instead of genuine
serialization-reading ability. This is the graph-domain analogue of
geometry's `maybe_reverse` (winding direction).

**Record schema:**

```json
{
  "object_id": "graph_simple_erdos_renyi_001",
  "tier": "simple",
  "family": "erdos_renyi",
  "num_nodes": 13,
  "num_edges": 19,
  "edge_list": "GRAPH (n=13, m=19):\n0 8\n0 9\n1 2\n...",
  "properties": {
    "triangle_count": 4,
    "is_bipartite": false,
    "is_planar": true,
    "diameter": 5,
    "chromatic_number": 3,
    "avg_clustering": 0.2821,
    "degree_of_node_0": 2,
    "edge_count": 19
  },
  "metadata": {
    "generation_params": {"p": 0.2857, "target_degree": 3.43},
    "random_seed": 42,
    "clique_number": 3,
    "chromatic_number_certified": true,
    "is_connected": true
  }
}
```

`object_id` format: `graph_{tier}_{family}_{index:03d}`. Index restarts at
001 within each (tier, family) group.

### Cell 11 — Section 4: build the dataset and summary

`build_dataset(seed=42)` iterates tier → family → index and calls the
dispatcher. Generation failure is **fatal**: if a generator exhausts 3000
tries and returns `None`, `build_dataset` raises rather than emitting a short
dataset.

`summarize(records)` produces `graph_exp1_summary.json`: `total`,
`counts_by_tier_family`, `bipartite_overall`, `planar_overall`,
`distribution_by_tier` (min/max/mean/median/std for `num_nodes`,
`num_edges`, `triangle_count`, `diameter`, `chromatic_number`,
`avg_clustering`, `edge_list_length`), and the chromatic-number
uncertified count/ids. `stats_for` computes population std (divide by *n*),
same as geometry.

### Cell 13 — Section 5: independent verification

Re-parses each record **from the stored edge-list string** with
`parse_edge_list` (plain-text/regex, no NetworkX) and recomputes properties
with hand-written implementations — BFS-based bipartiteness and diameter,
adjacency-set triangle counting, hand-rolled clustering coefficient — then
compares against stored ground truth. On the committed run: **0 mismatches
across 300 graphs, 6 properties each** (1,800 checks, plus header
cross-checks).

**Known gap**, same discipline as geometry's own documented gap: this covers
6 of 8 properties. `is_planar` and `chromatic_number` are not independently
re-derived — an independent planarity test (Boyer–Myrvold) and an
independent exact-coloring implementation are both substantial undertakings
on their own. Worth closing in a port.

### Cell 15 — Section 6: visual spot-check

Plots one graph per (tier × family) — 15 graphs — into a 3×5 grid using
`nx.draw` with a spring layout, saves `spotcheck_exp1_graph.png` at 110 dpi.
Eyeball check: Barabási–Albert should show visible hubs, Watts–Strogatz
should look ring-like with a few long-range rewires, random-planar should
look like a non-crossing mesh. Not an automated assertion.

### Cell 17 — Section 8: Colab download

Same `try/except` pattern as geometry — downloads on Colab, no-ops locally
(files already on disk).

---

## 3. Reproducibility

- Single seed, `SEED = 42`, passed to one `random.Random(seed)` instance that
  drives every generator's parameter draws, the node relabeling, and is
  passed directly to NetworkX as its `seed` argument.
- The seed is stored in **every record** (`metadata.random_seed`).
- Same seed + same NetworkX/SciPy version ⇒ byte-identical dataset (hull tie
  breaking in `scipy.spatial.Delaunay` and NetworkX's RNG consumption order
  are library internals). The notebook prints `networkx.__version__` in cell
  1 for this reason — record it when reporting results.
- Cells must run **in order**. Cell 11 depends on every definition above it;
  cells 13 and 15 depend on the in-memory `records` list from cell 11.
- Notebook outputs are stripped in the committed version — the JSON
  artifacts in this folder are the record of what was produced.

## 4. Observed dataset characteristics (seed 42)

From `graph_exp1_summary.json`:

- **bipartite_overall = 71 / 300 (23.7%)**, **planar_overall = 93 / 300
  (31.0%)** — against the PDF's "aim for approximately 25–35%" for both.
  Planar lands inside the target band; bipartite falls just short. This is
  structural, not a tuning miss: `random_bipartite` is the only family that
  *guarantees* a bipartite instance (66/66), and it is already set to the top
  of the PDF's allowed 18–22 per-tier range (22). The other four families are
  all triangle-rich by construction — Erdős–Rényi at expected degree 3–6,
  Barabási–Albert's preferential attachment, Watts–Strogatz's ring-lattice
  rewiring, and Delaunay-derived planar graphs all produce odd cycles almost
  every time — so they contribute only 5 incidental bipartite hits combined
  across 234 graphs (all 5 from random-planar; 0 from ER, BA, WS). Closing the
  remaining gap without breaking per-family balance would require either
  loosening the PDF's family-balance tolerance or adding a sixth,
  tree-biased family — neither attempted here, since the PDF states the
  target as "aim for," not a hard gate, and 71 positive / 229 negative
  instances is already ample for the boolean-property precision/recall/
  confusion-matrix reporting in Section 6.
- **chromatic_number_uncertified_count = 1**
  (`graph_hard_watts_strogatz_008`) — every other graph's chromatic number
  closed via the clique certificate or the time-boxed backtracking search;
  this one graph didn't close within the 15s budget and is excluded from
  querying/evaluation per the PDF's instruction, not approximated.
- **Scale separation across tiers** is large and intentional:

  | Tier | mean nodes | mean edges | mean triangles | mean edge-list length |
  |------|-----------|-----------|-----------------|------------------------|
  | simple | 12.14 | 22.71 | 9.26 | 120 chars |
  | medium | 28.18 | 69.22 | 22.32 | 385 chars |
  | hard   | 61.26 | 176.96 | 39.29 | 1,025 chars |

  `chromatic_number` stays compact across tiers (mean 3.18 → 3.42 → 3.39,
  max 5 throughout) since every family targets sparse-to-moderate density by
  design (Table 7's degree/`k`/`m` ranges) — this is expected, not a bug:
  chromatic number is bounded by max degree + 1 and these graphs are far
  from dense.
- `avg_clustering` decreases with tier (0.31 → 0.23 → 0.18 mean) — larger,
  sparser graphs have proportionally fewer closed triangles per node.

## 5. How Phase 2 consumes this

Mirrors the geometry contract exactly (see
[`phase2_model_results/README.md`](../phase2_model_results/README.md) §1 for
the pattern). Phase 2 runners read `graph_exp1_dataset.json` and use exactly
four things per record:

| Field | Used for |
|-------|----------|
| `object_id` | resume key — `(object_id, property)` pairs already in the JSONL are skipped |
| `edge_list` | the serialized string embedded in the prompt |
| `properties[prop]` | ground truth for scoring the model's answer |
| `metadata.chromatic_number_certified` | when `false`, the `(object_id, "chromatic_number")` pair is excluded from querying/evaluation entirely, per the PDF |

`tier` and `family` ride along into result records for slicing at analysis
time. Everything else in `metadata` is provenance.

---

## 6. Porting this to a new domain

What is domain-specific here and what is not — same split as geometry's own
porting section:

**Reusable structure — keep the shape, change the content:**

- The tier system (3 difficulty levels defined by a complexity parameter).
- The category split within each tier (5 balanced structural families here;
  3 for geometry — the count is domain-specific, the *pattern* of named,
  balanced generation categories is not).
- Rejection sampling against an explicit, enumerated validity rule set,
  where every rejection has a named reason.
- Compute-invariant-properties → randomize-presentation → serialize →
  read-presentation-dependent-property. Any domain with a property that
  depends on how the object is written down needs this ordering — here,
  node-label identity; for geometry, winding direction.
- Independent verification from the serialized string with an
  implementation that does not share code with the generator.
- The record schema and the summary statistics block.
- Time-boxing an expensive-but-not-always-tractable ground-truth
  computation, with an explicit "exclude and record why" fallback rather
  than reporting an approximate value.

**Domain-specific — must be rewritten:**

| Component | Graph version | What a port replaces it with |
|-----------|----------------|-------------------------------|
| Object type | NetworkX `Graph` | the new domain's object |
| Serialization | edge list `GRAPH (n=.., m=..): ...` | WKT, SMILES, CSV row, … |
| `TIERS` | node-count ranges | the domain's complexity axis |
| Generators | 5 NetworkX/SciPy families | domain-appropriate samplers |
| `check_validity` | the 6 graph rules | the domain's validity rules |
| Property functions | triangles, bipartiteness, planarity, diameter, chromatic number, clustering | the domain's Table-8-equivalent properties |
| Presentation randomizer | `randomize_labeling` (node relabeling) | whatever presentation choice is arbitrary in that domain |
| Independent verifier | BFS/adjacency-set hand implementations | an independent reimplementation for the new format |

See `../phase1_dataset/README.md` for the Geometry-domain side of the port,
and the root [Readme.md](../Readme.md) for the full three-domain pipeline.

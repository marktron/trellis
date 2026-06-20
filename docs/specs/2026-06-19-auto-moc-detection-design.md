# Phase 3 — Auto-MOC detection (design)

Date: 2026-06-19
Status: approved, pre-implementation

## Goal

Surface dense thematic clusters in the Zettelkasten that have **no covering MOC
yet**, and hand them off as candidates the user can build with the `/moc` skill.
trellis detects and reports; it never writes MOCs itself.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Output shape | **Detection + handoff report.** No auto-drafting of MOC files. |
| Clustering deps | **HDBSCAN in a venv.** Accept the venv cost for battle-tested clustering. |
| Reduction step | **UMAP → HDBSCAN** (canonical topic pipeline; better than PCA on 1024-dim text embeddings). |
| Coverage test | **Hybrid:** semantic gate (cluster centroid vs. each MOC embedding) decides candidacy; link coverage (% of members already MOC-linked) is reported as context only. |
| Cluster scope | **`z/` only** — the atomic notes MOCs organize. MOC files are used only for the coverage comparison, never clustered as members. |
| Cluster labeling | **LLM-named themes** — one gen-model call per candidate produces theme title + suggested tag + rationale. |
| Command name | `trellis cluster`. |

## Dependencies & execution

- New venv at `~/Developer/trellis/.venv` (gitignored).
- New `requirements.txt`: `numpy`, `scikit-learn`, `umap-learn`, `hdbscan`.
- The venv becomes the canonical interpreter. `run-garden.sh` and
  `com.trellis.garden.plist` switch to `.venv/bin/python3`.
- `umap` and `hdbscan` are imported **lazily inside the cluster command only**, so
  `index` / `search` / `neighbors` / `garden` / `apply` keep working on bare
  system-python + numpy and pay no numba import cost.

## Pipeline — `trellis cluster`

Flags: `--dry-run`, `--force`, `--scope <prefixes>`, `--limit <n>`.

1. **Load + scope:** read `(path, title, dim, embedding)` from `index.db`; keep
   notes whose path starts with a `cluster_scope` prefix (default `z/`).
2. **Reduce:** UMAP to `umap_components` dims (default 5), with
   `n_neighbors=umap_neighbors` (15), `min_dist=umap_min_dist` (0.0),
   `metric="cosine"`, `random_state=random_state` (42) for reproducibility.
3. **Cluster:** HDBSCAN with `min_cluster_size=hdbscan_min_cluster_size` (8),
   `cluster_selection_method="eom"`. Label `-1` (noise) is ignored.
4. **Per cluster:** compute the **centroid in the original 1024-dim space** (mean
   of the L2-normalized member vectors, renormalized) and collect members. The
   **anchor** is the single member nearest the centroid (stable cluster identity).

## Coverage test (hybrid)

- **Semantic gate — decides candidacy.** Cosine similarity between the cluster
  centroid and each **MOC's own embedding** (MOC files are already in the index
  under `MOCs/`). Record the nearest MOC and its score. If
  `score >= cover_sim_threshold` (default **0.60**, provisional — tune after the
  first real run, as we did with `max_chars`), the cluster is **covered** and is
  not surfaced.
- **Link coverage — context only.** Fraction of the cluster's member notes that
  are already linked from any MOC file, computed by reusing `_scan_vault` +
  `build_link_graph` (inbound edges originating in `MOCs/`). Reported, never a gate.

## LLM naming (candidates only)

For each **uncovered** cluster, one `generate_json` call to the gen model
(`qwen3.6:35b-a3b`) with the cluster's most common tags and the centroid-nearest
note titles. Returns `{"theme": "...", "suggested_tag": "...", "rationale": "..."}`.
Reuses existing gen plumbing: `gen_timeout`, `gen_num_predict` cap, and the
JSON-null guard. A failed/empty generation falls back to top-tags-as-theme so the
candidate still appears.

## Seen-ledger & report

- **Ledger table** `moc_candidates(anchor_path TEXT PRIMARY KEY, theme TEXT,
  tag TEXT, member_count INTEGER, nearest_moc TEXT, score REAL, first_seen REAL,
  status TEXT DEFAULT 'new')`. A candidate whose anchor is already in the ledger
  is not re-surfaced. Once the user actually builds a MOC for a theme, the semantic
  gate excludes that cluster on the next run automatically — no manual status
  update required. `--force` re-surfaces everything and ignores the ledger.
- **Report** `_claude-output/clusters/YYYY-MM-DD.md` (same dated path + same-day
  collision guard as the gardener). Header summary: `N clusters found · M new
  candidates · K covered`. Per candidate:
  - theme title and `suggested_tag`
  - rationale (one line)
  - coverage line: nearest MOC + score; % of members already MOC-linked
  - representative notes (centroid-nearest, up to `cluster_repr_notes`, default 8)
  - full member list
  - a ready-to-run `/moc <theme>` line
- `--dry-run` prints the report and writes nothing (no ledger, no file), matching
  `garden`.

## Configuration (new `trellis.toml` keys)

```
cluster_scope            = ["z/"]
umap_components          = 5
umap_neighbors           = 15
umap_min_dist            = 0.0
hdbscan_min_cluster_size = 8
cover_sim_threshold      = 0.60   # provisional; tune after first run
cluster_repr_notes       = 8
random_state             = 42
```

## Testing

- **Pure helpers (unit-tested, no deps, no network):** centroid + coverage
  scoring, link-coverage fraction, anchor selection, anchor-based ledger dedup,
  naming-prompt builder, report renderer. These join the existing fast suite.
- **Thin wrapper** `reduce_and_cluster(matrix, params) -> labels` isolates the
  UMAP/HDBSCAN call. One **integration test**: 3 synthetic gaussian blobs, seeded,
  assert ≥2 clusters found. Skips cleanly (`unittest.skip`) if `umap`/`hdbscan`
  are not importable, so the fast suite stays dependency-free.

## Scope / non-goals (YAGNI)

- `cluster` is **manual** for now — themes don't shift nightly, so it is not added
  to the 3am launchd job. A weekly schedule can be added later.
- **No auto-drafting** of MOCs. Detection + handoff only.
- No re-clustering optimization or incremental clustering — a full re-cluster each
  run is fine at this corpus size (~1,250 `z/` notes).

## Open items (resolve empirically, not blockers)

- `cover_sim_threshold` (0.60) and `hdbscan_min_cluster_size` (8) are starting
  defaults to tune against the first real run.
- If HDBSCAN labels too much as noise, revisit `min_cluster_size` /
  `umap_neighbors` before adding `min_samples`.

# Phase 3 — Auto-MOC Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `trellis cluster` command that finds dense thematic clusters in the `z/` Zettelkasten with no covering MOC and writes a dated handoff report of LLM-named candidates.

**Architecture:** UMAP → HDBSCAN over the existing embedding index (z/ only), reduced in a new venv. Each cluster gets a centroid in original 1024-dim space; a hybrid coverage test (semantic similarity to existing MOC embeddings + % of members already MOC-linked) decides candidacy; uncovered clusters are named by the local gen model; an anchor-keyed seen-ledger prevents re-surfacing. Pure logic is extracted into unit-tested helpers; the UMAP/HDBSCAN call is a thin wrapper with one skippable integration test.

**Tech Stack:** Python 3.12, numpy, scikit-learn, umap-learn, hdbscan, SQLite, Ollama (qwen3.6 for naming). Spec: `docs/specs/2026-06-19-auto-moc-detection-design.md`.

**Commit convention:** every commit message must end with:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

**Test runner (venv):** `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py`
The pure-helper tests need only numpy; `umap`/`hdbscan` are imported lazily inside the cluster wrapper, so the fast suite never imports them.

---

## File Structure

- **Create** `requirements.txt` — pinned-ish dep list for the venv.
- **Modify** `.gitignore` — ignore `.venv/`.
- **Modify** `trellis.py` — add cluster config defaults, ~10 pure helpers, the `reduce_and_cluster` wrapper, `_ensure_cluster_tables`, `cmd_cluster`, and the `cluster` CLI subparser + dispatch.
- **Modify** `tests/test_trellis.py` — pure-helper unit tests + one skippable integration test.
- **Modify** `trellis.toml` — add the cluster config block.
- **Modify** `README.md` — Phase 3 usage + venv setup.
- **Modify** `run-garden.sh`, `com.trellis.garden.plist` — point at `.venv/bin/python3`.

---

## Task 1: Stand up the venv and confirm a green baseline

**Files:**
- Create: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Write `requirements.txt`**

```
numpy
scikit-learn
umap-learn
hdbscan
```

- [ ] **Step 2: Add `.venv/` to `.gitignore`**

Append this line to `.gitignore`:

```
.venv/
```

- [ ] **Step 3: Create the venv and install**

Run:
```bash
python3 -m venv ~/Developer/trellis/.venv
~/Developer/trellis/.venv/bin/python3 -m pip install -U pip
~/Developer/trellis/.venv/bin/python3 -m pip install -r ~/Developer/trellis/requirements.txt
```
Expected: installs complete; `~/Developer/trellis/.venv/bin/python3 -c "import umap, hdbscan, sklearn, numpy"` exits 0.

- [ ] **Step 4: Confirm the existing suite is green in the venv**

Run: `~/Developer/trellis/.venv/bin/python3 ~/Developer/trellis/tests/test_trellis.py`
Expected: `Ran 34 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add venv requirements for Phase 3 clustering"
```

---

## Task 2: Add cluster config defaults

**Files:**
- Modify: `trellis.py` (the `DEFAULTS` dict)
- Modify: `trellis.toml`

- [ ] **Step 1: Add keys to `DEFAULTS`**

In `trellis.py`, inside the `DEFAULTS` dict, after the gardener block (after `"tag_candidate_neighbors": 15,`) add:

```python
    # --- auto-MOC clustering (phase 3) ---
    "cluster_scope": ["z/"],          # path prefixes clustered for MOC candidates
    "umap_components": 5,             # UMAP target dimensionality
    "umap_neighbors": 15,            # UMAP n_neighbors
    "umap_min_dist": 0.0,            # UMAP min_dist (0 = tightest clusters)
    "hdbscan_min_cluster_size": 8,   # smallest group worth a MOC
    "cover_sim_threshold": 0.60,     # centroid≥this to an MOC embedding ⇒ covered
    "cluster_repr_notes": 8,         # representative notes shown per candidate
    "random_state": 42,              # seed UMAP for run-to-run stability
```

- [ ] **Step 2: Add the matching block to `trellis.toml`**

Append to `trellis.toml`:

```toml
# --- auto-MOC clustering (phase 3) ---
cluster_scope            = ["z/"]
umap_components          = 5
umap_neighbors           = 15
umap_min_dist            = 0.0
hdbscan_min_cluster_size = 8
cover_sim_threshold      = 0.60   # provisional; tune after first run
cluster_repr_notes       = 8
random_state             = 42
```

- [ ] **Step 3: Confirm nothing broke**

Run: `~/Developer/trellis/.venv/bin/python3 ~/Developer/trellis/tests/test_trellis.py`
Expected: `Ran 34 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add trellis.py trellis.toml
git commit -m "feat: add Phase 3 cluster config defaults"
```

---

## Task 3: `cluster_members` helper

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trellis.py` before `if __name__ == "__main__":`:

```python
class TestClusterHelpers(unittest.TestCase):
    def test_cluster_members_groups_and_drops_noise(self):
        labels = [0, 1, 0, -1, 1, 1]
        out = t.cluster_members(labels)
        self.assertEqual(out[0], [0, 2])
        self.assertEqual(out[1], [1, 4, 5])
        self.assertNotIn(-1, out)   # noise dropped
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k test_cluster_members_groups_and_drops_noise`
Expected: FAIL — `module 'trellis' has no attribute 'cluster_members'`

- [ ] **Step 3: Implement**

In `trellis.py`, add to the gardener/cluster helper area (after `candidate_tags`):

```python
def cluster_members(labels):
    """Map each non-noise cluster label to its member row indices (-1 = noise)."""
    out = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab < 0:
            continue
        out.setdefault(lab, []).append(i)
    return out
```

- [ ] **Step 4: Run it, verify it passes**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k test_cluster_members_groups_and_drops_noise`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: cluster_members helper"
```

---

## Task 4: `centroid` and `rank_by_centrality` helpers

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestClusterHelpers`:

```python
    def test_centroid_is_normalized_mean(self):
        mat = t.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        c = t.centroid(mat, [0, 1])
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
        self.assertAlmostEqual(float(c[0]), float(c[1]), places=5)  # symmetric

    def test_rank_by_centrality_orders_by_cosine(self):
        mat = t.l2_normalize(np.array([
            [1.0, 0.0],   # 0 — on axis
            [0.0, 1.0],   # 1 — orthogonal
            [0.9, 0.1],   # 2 — near axis
        ], dtype=np.float32))
        cen = np.array([1.0, 0.0], dtype=np.float32)
        self.assertEqual(t.rank_by_centrality(mat, [0, 1, 2], cen), [0, 2, 1])
```

- [ ] **Step 2: Run them, verify they fail**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k "centroid or rank_by_centrality"`
Expected: FAIL — attributes not defined

- [ ] **Step 3: Implement**

In `trellis.py`, after `cluster_members`:

```python
def centroid(matrix, indices):
    """L2-normalized mean of the given rows (rows assumed already normalized)."""
    c = matrix[indices].mean(axis=0)
    n = np.linalg.norm(c)
    return (c / n if n else c).astype(np.float32)


def rank_by_centrality(matrix, indices, centroid_vec):
    """Return `indices` sorted by descending cosine similarity to centroid_vec."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = matrix[indices] @ centroid_vec
    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return [indices[i] for i in np.argsort(-sims)]
```

- [ ] **Step 4: Run them, verify they pass**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k "centroid or rank_by_centrality"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: centroid + rank_by_centrality helpers"
```

---

## Task 5: `coverage_score` helper

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestClusterHelpers`:

```python
    def test_coverage_score_picks_nearest_moc(self):
        moc = t.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        cen = np.array([1.0, 0.0], dtype=np.float32)
        j, score = t.coverage_score(cen, moc)
        self.assertEqual(j, 0)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_coverage_score_no_mocs(self):
        cen = np.array([1.0, 0.0], dtype=np.float32)
        self.assertEqual(t.coverage_score(cen, np.zeros((0, 2), np.float32)), (-1, 0.0))
```

- [ ] **Step 2: Run them, verify they fail**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k coverage_score`
Expected: FAIL — attribute not defined

- [ ] **Step 3: Implement**

In `trellis.py`, after `rank_by_centrality`:

```python
def coverage_score(centroid_vec, moc_matrix):
    """Best (row_index, cosine) of the centroid vs MOC embeddings.

    Returns (-1, 0.0) when there are no MOC vectors. moc_matrix rows assumed
    L2-normalized.
    """
    if moc_matrix.shape[0] == 0:
        return -1, 0.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = moc_matrix @ centroid_vec
    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
    j = int(np.argmax(sims))
    return j, float(sims[j])
```

- [ ] **Step 4: Run them, verify they pass**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k coverage_score`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: coverage_score helper"
```

---

## Task 6: `moc_linked_targets` and `link_coverage` helpers

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestClusterHelpers`:

```python
    def test_moc_linked_targets_only_from_mocs(self):
        notes = {
            "MOCs/Strategy MOC.md": {"title": "Strategy MOC", "out": {"moats"}},
            "z/moats.md": {"title": "moats", "out": set()},
            "z/other.md": {"title": "other", "out": {"moats"}},  # not a MOC source
        }
        t2r, _ = t.build_link_graph(notes)
        linked = t.moc_linked_targets(notes, t2r)
        self.assertIn("z/moats.md", linked)          # linked from the MOC
        self.assertEqual(len(linked), 1)             # the z/ source doesn't count

    def test_link_coverage_fraction(self):
        self.assertEqual(t.link_coverage(["a", "b", "c", "d"], {"a", "c"}), 0.5)
        self.assertEqual(t.link_coverage([], {"a"}), 0.0)
```

- [ ] **Step 2: Run them, verify they fail**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k "moc_linked or link_coverage"`
Expected: FAIL — attributes not defined

- [ ] **Step 3: Implement**

In `trellis.py`, after `coverage_score`:

```python
def moc_linked_targets(notes, title_to_rel):
    """Set of rel paths that are wikilink targets from any note under MOCs/."""
    linked = set()
    for rel, n in notes.items():
        if not rel.startswith("MOCs"):
            continue
        for tgt in n["out"]:
            dest = title_to_rel.get(tgt)
            if dest:
                linked.add(dest)
    return linked


def link_coverage(member_paths, moc_linked):
    """Fraction of member_paths already linked from some MOC (0.0 if empty)."""
    if not member_paths:
        return 0.0
    return sum(1 for p in member_paths if p in moc_linked) / len(member_paths)
```

- [ ] **Step 4: Run them, verify they pass**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k "moc_linked or link_coverage"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: moc_linked_targets + link_coverage helpers"
```

---

## Task 7: `filter_unseen` ledger-dedup helper

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing test**

Add to `TestClusterHelpers`:

```python
    def test_filter_unseen_drops_known_anchors(self):
        cands = [{"anchor": "z/a.md"}, {"anchor": "z/b.md"}, {"anchor": "z/c.md"}]
        out = t.filter_unseen(cands, {"z/b.md"})
        self.assertEqual([c["anchor"] for c in out], ["z/a.md", "z/c.md"])
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k filter_unseen`
Expected: FAIL — attribute not defined

- [ ] **Step 3: Implement**

In `trellis.py`, after `link_coverage`:

```python
def filter_unseen(candidates, seen_anchors):
    """Drop candidates whose anchor path is already in seen_anchors."""
    return [c for c in candidates if c["anchor"] not in seen_anchors]
```

- [ ] **Step 4: Run it, verify it passes**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k filter_unseen`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: filter_unseen ledger-dedup helper"
```

---

## Task 8: `build_cluster_naming_prompt` helper + prompt template

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing test**

Add to `TestClusterHelpers`:

```python
    def test_naming_prompt_includes_tags_and_titles(self):
        p = t.build_cluster_naming_prompt(["health", "aging"], ["Sleep and aging", "VO2max"])
        self.assertIn("health, aging", p)
        self.assertIn("Sleep and aging", p)
        self.assertIn("VO2max", p)
        self.assertIn("theme", p)          # asks for the JSON theme field
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k naming_prompt`
Expected: FAIL — attribute not defined

- [ ] **Step 3: Implement**

In `trellis.py`, near the other prompt templates (after `TAG_PROMPT`), add:

```python
CLUSTER_NAME_PROMPT = """You organize a Zettelkasten into topic maps (MOCs). \
Below is a cluster of related notes found by semantic similarity. Name the single \
coherent theme they share, in a few words suitable as a MOC title.

COMMON TAGS: {tags}

REPRESENTATIVE NOTES:
{titles}

Return JSON only:
{{"theme": "<short title>", "suggested_tag": "<one lowercase tag, nested ok>", "rationale": "<8 words max>"}}"""
```

And, after `filter_unseen`:

```python
def build_cluster_naming_prompt(top_tags, repr_titles):
    tags = ", ".join(top_tags) if top_tags else "(none)"
    titles = "\n".join(f"- {x}" for x in repr_titles)
    return CLUSTER_NAME_PROMPT.format(tags=tags, titles=titles)
```

- [ ] **Step 4: Run it, verify it passes**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k naming_prompt`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: cluster naming prompt builder"
```

---

## Task 9: `render_cluster_report` helper

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

A candidate dict has these keys (defined here, produced in Task 12):
`anchor, theme, tag, rationale, member_count, link_coverage, nearest_moc` (a `(title, score)` tuple or `None`), `repr_titles` (list), `member_titles` (list).

- [ ] **Step 1: Write the failing tests**

Add to `TestClusterHelpers`:

```python
    def _sample_candidate(self):
        return {"anchor": "z/sleep.md", "theme": "Sleep & Aging", "tag": "aging/sleep",
                "rationale": "sleep quality declines with age", "member_count": 9,
                "link_coverage": 0.0, "nearest_moc": ("Active Aging & Longevity MOC", 0.41),
                "repr_titles": ["Sleep and aging", "Deep sleep"],
                "member_titles": ["Sleep and aging", "Deep sleep", "Naps"]}

    def test_render_report_includes_candidate_and_moc_line(self):
        md = t.render_cluster_report(
            "2026-06-19", {"clusters": 5, "candidates": 1, "covered": 4},
            [self._sample_candidate()])
        self.assertIn("# MOC candidates — 2026-06-19", md)
        self.assertIn("## Sleep & Aging", md)
        self.assertIn("`aging/sleep`", md)
        self.assertIn("/moc Sleep & Aging", md)
        self.assertIn("[[Sleep and aging]]", md)
        self.assertIn("Active Aging & Longevity MOC", md)

    def test_render_report_empty(self):
        md = t.render_cluster_report(
            "2026-06-19", {"clusters": 0, "candidates": 0, "covered": 0}, [])
        self.assertIn("No new MOC candidates", md)
```

- [ ] **Step 2: Run them, verify they fail**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k render_report`
Expected: FAIL — attribute not defined

- [ ] **Step 3: Implement**

In `trellis.py`, after `build_cluster_naming_prompt`:

```python
def render_cluster_report(date_str, summary, candidates):
    """Render the MOC-candidate review as markdown. Pure (no I/O) for testing."""
    L = [f"# MOC candidates — {date_str}", ""]
    L.append(f"_{summary['clusters']} cluster(s) found · "
             f"{summary['candidates']} new candidate(s) · "
             f"{summary['covered']} already covered._")
    L.append("")
    if not candidates:
        L.append("_No new MOC candidates this run._")
        return "\n".join(L)
    L.append("> Each candidate is a dense theme with no covering MOC. "
             "Run the suggested `/moc` line for any worth building.")
    L.append("")
    for c in candidates:
        L.append(f"## {c['theme']}")
        L.append(f"- suggested tag: `{c['tag']}`")
        L.append(f"- {c['rationale']}")
        if c.get("nearest_moc"):
            title, score = c["nearest_moc"]
            L.append(f"- nearest existing MOC: [[{title}]] (sim {score:.2f})")
        L.append(f"- {c['member_count']} notes · "
                 f"{c['link_coverage'] * 100:.0f}% already linked from a MOC")
        L.append("")
        L.append("Representative notes:")
        for title in c["repr_titles"]:
            L.append(f"- [[{title}]]")
        L.append("")
        L.append(f"`/moc {c['theme']}`")
        L.append("")
        L.append("<details><summary>All members</summary>")
        L.append("")
        for title in c["member_titles"]:
            L.append(f"- [[{title}]]")
        L.append("")
        L.append("</details>")
        L.append("")
    return "\n".join(L)
```

- [ ] **Step 4: Run them, verify they pass**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k render_report`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: render_cluster_report helper"
```

---

## Task 10: `reduce_and_cluster` wrapper + integration test

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing (skippable) integration test**

Add a new class to `tests/test_trellis.py` before `if __name__ == "__main__":`:

```python
class TestReduceAndCluster(unittest.TestCase):
    def setUp(self):
        try:
            import umap  # noqa: F401
            import hdbscan  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("umap/hdbscan not installed")

    def test_finds_separated_blobs(self):
        rng = np.random.RandomState(0)
        dim = 50
        blobs = []
        for center in (0, 5, 10):
            base = np.zeros(dim, dtype=np.float32)
            base[center] = 1.0
            blobs.append(base + rng.normal(0, 0.05, size=(20, dim)).astype(np.float32))
        mat = t.l2_normalize(np.vstack(blobs))
        params = {"umap_components": 2, "umap_neighbors": 10, "umap_min_dist": 0.0,
                  "hdbscan_min_cluster_size": 5, "random_state": 42}
        labels = t.reduce_and_cluster(mat, params)
        self.assertEqual(len(labels), 60)
        self.assertGreaterEqual(len({int(x) for x in labels if x >= 0}), 2)
```

- [ ] **Step 2: Run it, verify it fails (not skips)**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k finds_separated_blobs`
Expected: FAIL — `module 'trellis' has no attribute 'reduce_and_cluster'` (the venv has the libs, so it must fail, not skip)

- [ ] **Step 3: Implement**

In `trellis.py`, after `render_cluster_report`:

```python
def reduce_and_cluster(matrix, params):
    """UMAP → HDBSCAN. Returns an int label array (-1 = noise).

    umap/hdbscan are imported here (not at module top) so the other commands
    never pay the heavy import cost and keep working on bare numpy.
    """
    import umap
    import hdbscan
    reducer = umap.UMAP(
        n_components=params["umap_components"],
        n_neighbors=params["umap_neighbors"],
        min_dist=params["umap_min_dist"],
        metric="cosine",
        random_state=params["random_state"],
    )
    reduced = reducer.fit_transform(matrix)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params["hdbscan_min_cluster_size"],
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced)
```

- [ ] **Step 4: Run it, verify it passes**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k finds_separated_blobs`
Expected: PASS (may print UMAP/numba warnings; the assertion passes)

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: reduce_and_cluster wrapper + integration test"
```

---

## Task 11: `_ensure_cluster_tables` ledger DDL

**Files:**
- Modify: `trellis.py`
- Test: `tests/test_trellis.py`

- [ ] **Step 1: Write the failing test**

Add a new class to `tests/test_trellis.py`:

```python
class TestClusterTables(unittest.TestCase):
    def test_creates_moc_candidates_table(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        t._ensure_cluster_tables(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(moc_candidates)")}
        self.assertEqual(
            cols,
            {"anchor_path", "theme", "tag", "member_count",
             "nearest_moc", "score", "first_seen", "status"})
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k creates_moc_candidates_table`
Expected: FAIL — attribute not defined

- [ ] **Step 3: Implement**

In `trellis.py`, near `_ensure_garden_tables`:

```python
def _ensure_cluster_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS moc_candidates (
                        anchor_path TEXT PRIMARY KEY, theme TEXT, tag TEXT,
                        member_count INTEGER, nearest_moc TEXT, score REAL,
                        first_seen REAL, status TEXT DEFAULT 'new')""")
    conn.commit()
```

- [ ] **Step 4: Run it, verify it passes**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py -k creates_moc_candidates_table`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "feat: moc_candidates ledger table"
```

---

## Task 12: `cmd_cluster` orchestration + CLI wiring

**Files:**
- Modify: `trellis.py` (add `cmd_cluster`; add subparser + dispatch in `main`)

This task wires the helpers together. It is integration-level; verification is a real `--dry-run` against the index (Step 4), since it depends on Ollama + the real DB.

- [ ] **Step 1: Implement `cmd_cluster`**

In `trellis.py`, after `cmd_garden` (and before the Apply section), add:

```python
def cmd_cluster(cfg, args):
    vault = cfg["vault"]
    scope = tuple(args.scope.split(",")) if args.scope else tuple(cfg["cluster_scope"])
    gen_model = cfg["gen_model"]
    limit = args.limit if args.limit is not None else 0

    conn = connect(cfg["db_path"])
    _ensure_cluster_tables(conn)
    paths, titles, mat = _load_matrix(conn)
    if not paths:
        print("index is empty — run:  python3 trellis.py index", file=sys.stderr)
        return 1

    # Notes to cluster (scope) and MOC vectors (coverage reference).
    cl_idx = [i for i, p in enumerate(paths) if p.startswith(scope)]
    moc_idx = [i for i, p in enumerate(paths) if p.startswith("MOCs")]
    if len(cl_idx) < cfg["hdbscan_min_cluster_size"]:
        print(f"too few notes in scope {scope} to cluster", file=sys.stderr)
        return 1
    sub = mat[cl_idx]
    moc_mat = mat[moc_idx] if moc_idx else np.zeros((0, mat.shape[1]), np.float32)

    print(f"clustering {len(cl_idx)} notes in scope {scope}…", flush=True)
    labels = reduce_and_cluster(sub, {
        "umap_components": cfg["umap_components"],
        "umap_neighbors": cfg["umap_neighbors"],
        "umap_min_dist": cfg["umap_min_dist"],
        "hdbscan_min_cluster_size": cfg["hdbscan_min_cluster_size"],
        "random_state": cfg["random_state"],
    })
    groups = cluster_members(labels)   # sub-index -> members (sub indices)

    # Vault scan for tags + MOC link coverage.
    print("scanning vault for tags + MOC links…", flush=True)
    notes = _scan_vault(vault, set(cfg["exclude_dirs"]), cfg["max_chars"])
    title_to_rel, _ = build_link_graph(notes)
    moc_linked = moc_linked_targets(notes, title_to_rel)

    seen_anchors = {row[0] for row in
                    conn.execute("SELECT anchor_path FROM moc_candidates").fetchall()}
    now = time.time()
    candidates, covered = [], 0

    for lab, sub_members in sorted(groups.items()):
        members = [cl_idx[m] for m in sub_members]            # back to global indices
        cen = centroid(mat, members)
        ranked = rank_by_centrality(mat, members, cen)        # global indices
        anchor_rel = paths[ranked[0]]

        j, score = coverage_score(cen, moc_mat)
        if score >= cfg["cover_sim_threshold"]:
            covered += 1
            continue                                          # an MOC already covers it
        nearest = (titles[moc_idx[j]], score) if j >= 0 else None

        member_paths = [paths[i] for i in members]
        member_titles = [titles[i] for i in ranked]
        repr_titles = member_titles[:cfg["cluster_repr_notes"]]
        top_tags = candidate_tags([notes.get(p, {}).get("tags", []) for p in member_paths],
                                  set(), 6)

        candidates.append({
            "anchor": anchor_rel, "theme": "", "tag": "", "rationale": "",
            "member_count": len(members), "link_coverage": link_coverage(member_paths, moc_linked),
            "nearest_moc": nearest, "repr_titles": repr_titles, "member_titles": member_titles,
            "top_tags": top_tags,
        })

    # Only-new (unless --force), then optional cap.
    if not args.force:
        candidates = filter_unseen(candidates, seen_anchors)
    if limit:
        candidates = candidates[:limit]

    # Name each candidate with the gen model (fallback: top tag).
    for c in candidates:
        prompt = build_cluster_naming_prompt(c["top_tags"], c["repr_titles"])
        try:
            out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                timeout=cfg["gen_timeout"], num_predict=cfg["gen_num_predict"])
        except OllamaError as e:
            print(f"  naming failed for {c['anchor']}: {str(e)[:120]}", file=sys.stderr)
            out = {}
        c["theme"] = str(out.get("theme") or (c["top_tags"][0] if c["top_tags"] else "Untitled theme")).strip()
        c["tag"] = str(out.get("suggested_tag") or (c["top_tags"][0] if c["top_tags"] else "")).strip()
        c["rationale"] = str(out.get("rationale") or "").strip()[:120]
        print(f"  candidate: {c['theme']}  ({c['member_count']} notes)", flush=True)

    date_str = datetime.date.today().isoformat()
    summary = {"clusters": len(groups), "candidates": len(candidates), "covered": covered}
    report = render_cluster_report(date_str, summary, candidates)

    if args.dry_run:
        print("\n--- DRY RUN (report not written) ---\n")
        print(report)
        return 0

    for c in candidates:
        nm = c["nearest_moc"][0] if c["nearest_moc"] else None
        sc = c["nearest_moc"][1] if c["nearest_moc"] else 0.0
        conn.execute("INSERT OR IGNORE INTO moc_candidates VALUES(?,?,?,?,?,?,?,?)",
                     (c["anchor"], c["theme"], c["tag"], c["member_count"], nm, sc, now, "new"))
    conn.commit()

    out_dir = os.path.join(vault, "_claude-output", "clusters")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.md")
    if os.path.exists(out_path):
        out_path = os.path.join(out_dir, f"{date_str}-{time.strftime('%H%M')}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nMOC candidates → {out_path}")
    print(f"  {len(candidates)} new candidate(s) · {covered} covered · {len(groups)} cluster(s)")
    return 0
```

- [ ] **Step 2: Add the CLI subparser**

In `main`, after the `apply` subparser block (`pa.add_argument(...)`), add:

```python
    pcl = sub.add_parser("cluster", help="detect MOC-candidate clusters -> review report")
    pcl.add_argument("--scope", help="comma-separated path prefixes (default: z/)")
    pcl.add_argument("--limit", type=int, help="max candidates to name/report (0 = no cap)")
    pcl.add_argument("--gen-model", dest="gen_model", help="judgment model for naming")
    pcl.add_argument("--force", action="store_true", help="ignore the seen-ledger")
    pcl.add_argument("--dry-run", action="store_true",
                     help="print report; write nothing (no ledger, no file)")
```

And add `cluster` to the dispatch dict (the `return {...}[args.cmd](cfg, args)` block):

```python
        "apply": cmd_apply, "cluster": cmd_cluster,
```

- [ ] **Step 3: Verify the suite still passes**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py`
Expected: `OK` (all unit + integration tests). Then check the CLI parses:
Run: `~/Developer/trellis/.venv/bin/python3 trellis.py cluster --help`
Expected: usage text listing `--scope --limit --gen-model --force --dry-run`

- [ ] **Step 4: Real dry-run smoke test**

Run: `~/Developer/trellis/.venv/bin/python3 trellis.py cluster --dry-run --limit 5`
Expected: prints "clustering N notes…", then a `# MOC candidates — <date>` report with up to 5 named candidates and a `covered` count. No file written, no ledger rows. (Requires the Ollama app running and a populated `index.db`.)

- [ ] **Step 5: Commit**

```bash
git add trellis.py
git commit -m "feat: cmd_cluster orchestration + cluster CLI"
```

---

## Task 13: Docs + nightly wrapper venv switch

**Files:**
- Modify: `README.md`
- Modify: `run-garden.sh`
- Modify: `com.trellis.garden.plist`

- [ ] **Step 1: Update `run-garden.sh` to use the venv python**

Replace the hardcoded `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` invocation with `~/Developer/trellis/.venv/bin/python3` (expand to the absolute `~/Developer/trellis/.venv/bin/python3`). Leave the index-then-garden sequence unchanged.

- [ ] **Step 2: Update `com.trellis.garden.plist`**

Change the `ProgramArguments` python path to `~/Developer/trellis/.venv/bin/python3`. (No new schedule for `cluster` — it stays manual per the spec.)

- [ ] **Step 3: Add a Phase 3 section to `README.md`**

Document, under a new "### Auto-MOC detection (Phase 3)" heading: the venv setup (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`), the commands:

```sh
.venv/bin/python3 trellis.py cluster                 # detect candidates -> review report
.venv/bin/python3 trellis.py cluster --dry-run       # print, write nothing
.venv/bin/python3 trellis.py cluster --limit 10      # cap candidates this run
.venv/bin/python3 trellis.py cluster --force         # ignore the seen-ledger
```

Note that reports land in `_claude-output/clusters/YYYY-MM-DD.md`, that the `moc_candidates` ledger keeps repeat runs quiet, that building a MOC for a theme makes the coverage test drop it automatically, and that `cover_sim_threshold` / `hdbscan_min_cluster_size` are the two dials to tune. Also update the Roadmap to mark Phase 3 done and the Requirements section to mention the venv for clustering.

- [ ] **Step 4: Verify**

Run: `~/Developer/trellis/.venv/bin/python3 tests/test_trellis.py`
Expected: `OK`. Visually confirm `run-garden.sh` and the plist reference the venv python.

- [ ] **Step 5: Commit**

```bash
git add README.md run-garden.sh com.trellis.garden.plist
git commit -m "docs: Phase 3 usage + point nightly job at venv python"
```

---

## Task 14: Tune thresholds against the first real run

**Files:** none (config-only, may touch `trellis.toml`)

- [ ] **Step 1: Full run**

Run: `~/Developer/trellis/.venv/bin/python3 trellis.py cluster`
Inspect `_claude-output/clusters/<date>.md`.

- [ ] **Step 2: Judge and adjust**

- If too much is noise (few/no clusters): lower `hdbscan_min_cluster_size` (e.g. 6) and/or raise `umap_neighbors` (e.g. 20) in `trellis.toml`.
- If real gaps are marked "covered": lower `cover_sim_threshold` (e.g. 0.50).
- If junk blobs slip through as candidates: raise `cover_sim_threshold` or `hdbscan_min_cluster_size`.
- Re-run with `--force --dry-run` to re-evaluate without touching the ledger.

- [ ] **Step 3: Commit any config changes**

```bash
git add trellis.toml
git commit -m "chore: tune cluster thresholds after first run"
```

---

## Self-Review

**Spec coverage:** detection+report (Tasks 9,12) ✓ · UMAP→HDBSCAN venv (Tasks 1,10) ✓ · hybrid coverage = semantic gate + link-coverage context (Tasks 5,6,12) ✓ · z/ scope (Task 2 config, Task 12 filter) ✓ · LLM naming with fallback (Tasks 8,12) ✓ · anchor seen-ledger + auto-drop-when-MOC-built (Tasks 7,11,12) ✓ · dated report w/ collision guard + dry-run (Task 12) ✓ · lazy clustering imports keep other commands numpy-only (Task 10) ✓ · manual (no nightly schedule) (Task 13) ✓ · config keys (Task 2) ✓ · tests pure + one skippable integration (Tasks 3–11) ✓ · threshold tuning flagged (Task 14) ✓.

**Placeholder scan:** none — every code step shows complete code; Task 13 doc edits are descriptive but the commands are concrete.

**Type consistency:** candidate dict keys (`anchor, theme, tag, rationale, member_count, link_coverage, nearest_moc, repr_titles, member_titles, top_tags`) are defined in Task 9 and produced identically in Task 12. `reduce_and_cluster` params dict keys match between Task 10 and Task 12. Helper names (`cluster_members, centroid, rank_by_centrality, coverage_score, moc_linked_targets, link_coverage, filter_unseen, build_cluster_naming_prompt, render_cluster_report, reduce_and_cluster, _ensure_cluster_tables`) are consistent across tasks.

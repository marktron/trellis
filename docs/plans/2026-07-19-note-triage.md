# New-Note Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `trellis triage` command that detects notes newly added to `z/`, suggests tags / MOC placement / Product-Idea links into the shared dated review file, and extend `trellis apply` to write the checked MOC and idea items.

**Architecture:** Everything lives in the single file `trellis.py` (house style — one module, pure helpers unit-tested, commands thin). Triage reuses the existing embedding index, gen-model plumbing (`generate_json`), the `suggestions` seen-ledger, and the review-file grammar. Garden and triage write into one dated review file per day (`# Review — YYYY-MM-DD` in `gardener_dir`) with append-if-pending semantics. Spec: `docs/specs/2026-07-19-note-triage-design.md`.

**Tech Stack:** Python 3.11+ stdlib + numpy only (no new deps). Ollama via existing `embed`/`generate_json` helpers. Tests: `unittest` in `tests/test_trellis.py`.

## Global Constraints

- All code goes in `trellis.py`; all tests go in `tests/test_trellis.py`. Do not create new modules.
- Tests must be pure: no network, no Ollama, no real vault. sqlite via `sqlite3.connect(":memory:")`, files via `tempfile.mkdtemp()`.
- Run the suite with `python3 -m unittest discover -s tests -q` from the repo root. Baseline before this plan: **116 tests, OK (skipped=1)**. Every task ends green.
- Follow existing naming: pure helpers are plain functions, private I/O helpers start with `_`, commands are `cmd_<name>(cfg, args)`.
- Commit messages: plain imperative mood (like `Add CI; make MOC scope and report dirs configurable`), ending with a blank line then `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- New config keys (added in Task 7, used throughout): `triage_scope=["z/"]`, `idea_scope=["Areas/Product Ideas/"]`, `triage_bulk_min=8`, `triage_tag_skip_threshold=3`, `moc_place_threshold=0.55`, `idea_link_threshold=0.55`. Review files share the existing `gardener_dir`.
- Suggestion-ledger kinds: existing `link`/`tag`, new `moc` (value = MOC title) and `idea` (value = idea title).

---

### Task 1: New-note detection helpers

Pure detection logic ported from the note-triage skill: trust `created:`/`published:` frontmatter over mtime (iCloud sync bumps mtimes of old notes), bulk-touch guard on the mtime fallback.

**Files:**
- Modify: `trellis.py` (add after `extract_tags`, near line 203)
- Test: `tests/test_trellis.py` (append new test classes)

**Interfaces:**
- Produces: `extract_created(frontmatter: str) -> datetime.date | None`
- Produces: `detect_new_notes(entries, cutoff, triaged, bulk_min) -> tuple[list[str], list[tuple[str, list[str]]]]` where `entries` is `[(rel, created_date_or_None, mtime_float)]`, `cutoff` is a `datetime.datetime`, `triaged` a `set[str]` of rels. Returns `(sorted candidate rels, [(minute_str, sorted rels)] suspected bulk buckets)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trellis.py`:

```python
class TestExtractCreated(unittest.TestCase):
    def test_created_date(self):
        import datetime
        self.assertEqual(t.extract_created("created: 2026-07-15\ntags: [a]"),
                         datetime.date(2026, 7, 15))

    def test_published_fallback_and_quotes(self):
        import datetime
        self.assertEqual(t.extract_created("published: '2026-07-01'"),
                         datetime.date(2026, 7, 1))

    def test_datetime_value_keeps_date_part(self):
        import datetime
        self.assertEqual(t.extract_created("created: 2026-07-15T09:30:00"),
                         datetime.date(2026, 7, 15))

    def test_absent_or_invalid(self):
        self.assertIsNone(t.extract_created(""))
        self.assertIsNone(t.extract_created("title: created note"))
        self.assertIsNone(t.extract_created("created: not-a-date"))


class TestDetectNewNotes(unittest.TestCase):
    CUT = __import__("datetime").datetime(2026, 7, 1, 12, 0, 0)

    def _e(self, rel, created=None, mtime=0.0):
        return (rel, created, mtime)

    def test_created_after_cutoff_is_candidate(self):
        import datetime
        cands, bulk = t.detect_new_notes(
            [self._e("z/new.md", datetime.date(2026, 7, 2))],
            self.CUT, set(), 8)
        self.assertEqual(cands, ["z/new.md"])
        self.assertEqual(bulk, [])

    def test_created_before_cutoff_ignored_even_with_fresh_mtime(self):
        import datetime
        fresh = self.CUT.timestamp() + 9999
        cands, _ = t.detect_new_notes(
            [self._e("z/old.md", datetime.date(2026, 6, 1), fresh)],
            self.CUT, set(), 8)
        self.assertEqual(cands, [])  # created: trumps mtime

    def test_mtime_fallback_when_no_created(self):
        fresh = self.CUT.timestamp() + 60
        cands, _ = t.detect_new_notes(
            [self._e("z/nofm.md", None, fresh),
             self._e("z/stale.md", None, self.CUT.timestamp() - 60)],
            self.CUT, set(), 8)
        self.assertEqual(cands, ["z/nofm.md"])

    def test_bulk_bucket_excluded_and_reported(self):
        base = self.CUT.timestamp() + 3600
        entries = [self._e(f"z/bulk{i}.md", None, base) for i in range(8)]
        entries.append(self._e("z/lone.md", None, base + 300))
        cands, bulk = t.detect_new_notes(entries, self.CUT, set(), 8)
        self.assertEqual(cands, ["z/lone.md"])
        self.assertEqual(len(bulk), 1)
        self.assertEqual(len(bulk[0][1]), 8)

    def test_already_triaged_excluded(self):
        import datetime
        cands, _ = t.detect_new_notes(
            [self._e("z/done.md", datetime.date(2026, 7, 2))],
            self.CUT, {"z/done.md"}, 8)
        self.assertEqual(cands, [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trellis.TestExtractCreated tests.test_trellis.TestDetectNewNotes -v`
Expected: ERROR — `AttributeError: module 'trellis' has no attribute 'extract_created'`

- [ ] **Step 3: Implement**

In `trellis.py`, directly after `extract_tags` (line ~203), add:

```python
_CREATED_RE = re.compile(r"(?mi)^(?:created|published)\s*:\s*['\"]?(\d{4}-\d{2}-\d{2})")


def extract_created(frontmatter: str) -> "datetime.date | None":
    """First created:/published: date in frontmatter, or None. Preferred over
    mtime for triage — iCloud sync and bulk scripts bump mtimes of old notes."""
    m = _CREATED_RE.search(frontmatter or "")
    if not m:
        return None
    try:
        return datetime.date.fromisoformat(m.group(1))
    except ValueError:
        return None


def detect_new_notes(entries, cutoff, triaged, bulk_min):
    """Pure new-note detection. entries: [(rel, created_date|None, mtime)].

    A note with a created: date is judged by that date alone (mtime ignored).
    Notes without one fall back to mtime, grouped into mtime-minute buckets:
    a bucket of >= bulk_min files is a suspected sync/bulk touch — excluded
    from candidates and returned separately for manual review.
    Returns (sorted candidate rels, [(minute_str, sorted rels)])."""
    cut_ts = cutoff.timestamp()
    cut_date = cutoff.date()
    candidates, mtime_only = [], []
    for rel, created, mtime in entries:
        if rel in triaged:
            continue
        if created is not None:
            if created > cut_date:
                candidates.append(rel)
        elif mtime > cut_ts:
            mtime_only.append((rel, mtime))
    buckets = collections.defaultdict(list)
    for rel, mtime in mtime_only:
        key = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        buckets[key].append(rel)
    suspected = []
    for key in sorted(buckets):
        rels = buckets[key]
        if len(rels) >= bulk_min:
            suspected.append((key, sorted(rels)))
        else:
            candidates.extend(rels)
    return sorted(candidates), suspected
```

Note: `datetime` is already imported at the top of `trellis.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 126 tests … OK (skipped=1)` (116 + 10 new)

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "Add new-note detection helpers for triage

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Triage state table and skill-state seed

**Files:**
- Modify: `trellis.py` (add after `_ensure_garden_tables`, line ~988)
- Test: `tests/test_trellis.py`

**Interfaces:**
- Consumes: `meta_set(conn, key, value)` / `meta_get(conn, key, default=None)` (existing).
- Produces: `_ensure_triage_tables(conn)` — creates `triage_state(path TEXT PRIMARY KEY, triaged_at REAL)`.
- Produces: `seed_triage_state(conn, state: dict, prefix: str = "z/") -> int` — one-time import of the skill's `triage-state.json` dict (`{"last_run_iso": ..., "triaged": [filenames]}`); no-op returning 0 if `triage_state` already has rows; sets meta key `triage_last_run` from `last_run_iso`.

- [ ] **Step 1: Write the failing tests**

```python
class TestTriageState(unittest.TestCase):
    def _conn(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        t._ensure_triage_tables(conn)
        return conn

    def test_seed_imports_names_with_prefix_and_last_run(self):
        conn = self._conn()
        n = t.seed_triage_state(
            conn, {"last_run_iso": "2026-07-12T09:00:00", "triaged": ["A.md", "B.md"]})
        self.assertEqual(n, 2)
        rows = {r[0] for r in conn.execute("SELECT path FROM triage_state")}
        self.assertEqual(rows, {"z/A.md", "z/B.md"})
        self.assertEqual(t.meta_get(conn, "triage_last_run"), "2026-07-12T09:00:00")

    def test_seed_noop_when_state_exists(self):
        conn = self._conn()
        conn.execute("INSERT INTO triage_state VALUES('z/X.md', 0)")
        n = t.seed_triage_state(conn, {"last_run_iso": "2026-07-12T09:00:00",
                                       "triaged": ["A.md"]})
        self.assertEqual(n, 0)
        self.assertIsNone(t.meta_get(conn, "triage_last_run"))

    def test_seed_tolerates_missing_keys(self):
        conn = self._conn()
        self.assertEqual(t.seed_triage_state(conn, {}), 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trellis.TestTriageState -v`
Expected: ERROR — no attribute `_ensure_triage_tables`

- [ ] **Step 3: Implement**

After `_ensure_garden_tables` in `trellis.py`:

```python
def _ensure_triage_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS triage_state (
                        path TEXT PRIMARY KEY, triaged_at REAL)""")
    conn.commit()


def seed_triage_state(conn, state: dict, prefix: str = "z/") -> int:
    """One-time import of the note-triage skill's _workspace/triage-state.json
    (filenames + last_run_iso), so skill-era triaged notes are never reprocessed.
    No-op once triage_state has any rows. Returns rows imported."""
    if conn.execute("SELECT COUNT(*) FROM triage_state").fetchone()[0]:
        return 0
    names = state.get("triaged") or []
    now = time.time()
    conn.executemany("INSERT OR IGNORE INTO triage_state VALUES(?,?)",
                     [(prefix + n, now) for n in names])
    if state.get("last_run_iso"):
        meta_set(conn, "triage_last_run", state["last_run_iso"])
    conn.commit()
    return len(names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 129 tests … OK (skipped=1)`

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "Add triage state table with one-time skill-state seed

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Unified review file — neutral title + append-if-pending writer

Garden and triage share one dated review file. Applied files are archived away by `_archive_review`, so "file exists in `gardener_dir`" ⇒ pending ⇒ append.

**Files:**
- Modify: `trellis.py` — `render_report` (line ~941), `cmd_garden` report write (lines ~1171-1173), new helpers near `_dated_report_path` (line ~1472)
- Test: `tests/test_trellis.py` — update `TestRenderReport.test_render_includes_sections_and_checkboxes` (line ~254), add `TestReviewWriter`

**Interfaces:**
- Produces: `strip_review_header(md: str) -> str` — drops the `# ` H1 line and the `> Check the boxes` hint line; returns remaining content stripped, with trailing newline (or `""`).
- Produces: `append_or_create_review(out_dir: str, date_str: str, report: str) -> str` — creates `out_dir/date_str.md` with `report`, or if that file exists appends `\n---\n\n` + `strip_review_header(report)`. Returns the path. (Task 7's `cmd_triage` uses this too.)
- Changes: `render_report` H1 becomes `# Review — {date}`; summary line becomes `_Garden: …_`.

- [ ] **Step 1: Update the existing render test and write failing writer tests**

In `TestRenderReport.test_render_includes_sections_and_checkboxes`, change:

```python
        self.assertIn("# Gardener review — 2026-06-15", md)
```

to:

```python
        self.assertIn("# Review — 2026-06-15", md)
        self.assertIn("_Garden:", md)
```

Append:

```python
class TestReviewWriter(unittest.TestCase):
    REPORT = ("# Review — 2026-07-19\n\n_Triage: 1 new note(s)._\n\n"
              "> Check the boxes you want, then run `trellis apply <this file>`. "
              "Nothing here has been written to your notes.\n\n"
              "## Tag suggestions\n\n- [ ] [[A]] → `x`\n")

    def test_strip_review_header(self):
        body = t.strip_review_header(self.REPORT)
        self.assertNotIn("# Review", body)
        self.assertNotIn("> Check the boxes", body)
        self.assertIn("_Triage: 1 new note(s)._", body)
        self.assertIn("- [ ] [[A]] → `x`", body)

    def test_create_when_absent(self):
        import tempfile
        d = tempfile.mkdtemp()
        path = t.append_or_create_review(d, "2026-07-19", self.REPORT)
        self.assertEqual(path, os.path.join(d, "2026-07-19.md"))
        with open(path) as fh:
            self.assertEqual(fh.read(), self.REPORT)

    def test_append_when_pending(self):
        import tempfile
        d = tempfile.mkdtemp()
        t.append_or_create_review(d, "2026-07-19", self.REPORT)
        second = self.REPORT.replace("Triage", "Garden").replace("`x`", "`y`")
        path = t.append_or_create_review(d, "2026-07-19", second)
        content = open(path).read()
        self.assertEqual(content.count("# Review — 2026-07-19"), 1)  # one H1
        self.assertEqual(content.count("> Check the boxes"), 1)      # one hint
        self.assertIn("\n---\n", content)
        self.assertIn("_Garden: 1 new note(s)._", content)
        self.assertIn("`y`", content)
        # merged file still parses as one review
        self.assertEqual(len(t.parse_review(content)["links"]), 0)

    def test_archived_same_day_file_does_not_block_create(self):
        import tempfile
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "applied"))
        with open(os.path.join(d, "applied", "2026-07-19.md"), "w") as fh:
            fh.write("old")
        path = t.append_or_create_review(d, "2026-07-19", self.REPORT)
        self.assertEqual(path, os.path.join(d, "2026-07-19.md"))
        self.assertEqual(open(path).read(), self.REPORT)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trellis.TestRenderReport tests.test_trellis.TestReviewWriter -v`
Expected: `test_render_includes_sections_and_checkboxes` FAILS (old title); `TestReviewWriter` ERRORS (no attribute)

- [ ] **Step 3: Implement**

In `render_report` (line ~944), change the first two appended lines:

```python
    L = [f"# Review — {date_str}", ""]
    L.append(
        f"_Garden: {summary['processed']} note(s) processed · "
        f"{summary['new_links']} new link suggestion(s) · "
        f"{summary['new_tags']} tag suggestion(s) · "
        f"{summary['orphans']} orphan(s) in scope._")
```

Near `_dated_report_path` (line ~1472), add:

```python
def strip_review_header(md: str) -> str:
    """Drop the H1 title and the checkbox-hint blockquote — the parts already
    present when appending a report to an existing pending review file."""
    keep = [ln for ln in md.splitlines()
            if not ln.startswith("# ") and not ln.startswith("> Check the boxes")]
    out = "\n".join(keep).strip()
    return out + "\n" if out else ""


def append_or_create_review(out_dir: str, date_str: str, report: str) -> str:
    """Write a report into the shared review queue. Today's dated file still
    sitting in out_dir means it is pending (applied files are archived away),
    so this report's body is appended under a rule instead of creating a
    suffixed sibling."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{date_str}.md")
    if os.path.exists(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n---\n\n{strip_review_header(report)}")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
    return path
```

In `cmd_garden` (lines ~1171-1173), replace:

```python
    out_path = _dated_report_path(os.path.join(vault, cfg["gardener_dir"]), date_str)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
```

with:

```python
    out_path = append_or_create_review(
        os.path.join(vault, cfg["gardener_dir"]), date_str, report)
```

(`_dated_report_path` stays — `cmd_cluster` still uses it for its own report dir.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 133 tests … OK (skipped=1)`

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "Unify review files: neutral title, append-if-pending writer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: parse_review — MOC placement and Product-idea sections

**Files:**
- Modify: `trellis.py` — regexes near line 1303, `parse_review` (line ~1308)
- Test: `tests/test_trellis.py` — extend `TestParseReview`

**Interfaces:**
- Changes: `parse_review(md)` returns `{"links": [(source, target)], "tags": [(source, [tags])], "mocs": [(note, moc, section)], "ideas": [(note, idea, reason)]}`. New line grammars (checked forms):
  - `- [x] [[Note]] → [[Some MOC]] § Section name — reason` under a `## MOC placements` heading
  - `- [x] [[Note]] → [[Some Idea]] — reason` under a `## Product idea links` heading
- Callers updated in Task 5 (`_apply_review_file` reads the new keys); until then the new keys are simply present and empty for old files.

- [ ] **Step 1: Write the failing tests**

Extend `TestParseReview.SAMPLE` — insert before the `## Orphans in scope` line:

```python
## MOC placements

- [x] [[Note A]] → [[Cycling MOC]] § Training physiology — mechanism note
- [ ] [[Note B]] → [[Strategy MOC]] § Positioning — unchecked, ignore

## Product idea links

- [x] [[Note A]] → [[Training Load Dashboard]] — evidence for aerobic-base metric
- [ ] [[Note B]] → [[Some Idea]] — unchecked, ignore

```

Add test methods:

```python
    def test_moc_placements_only_checked(self):
        r = t.parse_review(self.SAMPLE)
        self.assertEqual(r["mocs"],
                         [("Note A", "Cycling MOC", "Training physiology")])

    def test_idea_links_only_checked_with_reason(self):
        r = t.parse_review(self.SAMPLE)
        self.assertEqual(r["ideas"],
                         [("Note A", "Training Load Dashboard",
                           "evidence for aerobic-base metric")])

    def test_old_files_have_empty_new_keys(self):
        r = t.parse_review("# Review — x\n\n## Link suggestions\n")
        self.assertEqual(r["mocs"], [])
        self.assertEqual(r["ideas"], [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trellis.TestParseReview -v`
Expected: FAIL — `KeyError: 'mocs'`

- [ ] **Step 3: Implement**

Next to the existing check regexes (line ~1303), add:

```python
_CHECK_MOC_RE = re.compile(
    r"^-\s+\[[xX]\]\s+\[\[(.+?)\]\]\s*→\s*\[\[(.+?)\]\]\s*§\s*(.+?)\s*—")
_CHECK_IDEA_RE = re.compile(
    r"^-\s+\[[xX]\]\s+\[\[(.+?)\]\]\s*→\s*\[\[(.+?)\]\]\s*—\s*(.*)$")
```

In `parse_review`: initialize `mocs: list[tuple[str, str, str]] = []` and
`ideas: list[tuple[str, str, str]] = []`; extend the section sniffing:

```python
        if line.startswith("## "):
            low = line.lower()
            section = ("link" if "link suggestion" in low
                       else "tag" if "tag suggestion" in low
                       else "moc" if "moc placement" in low
                       else "idea" if "idea link" in low else None)
            src = None
            continue
```

and add the two branches after the `tag` branch:

```python
        elif section == "moc":
            m = _CHECK_MOC_RE.match(line)
            if m:
                mocs.append((m.group(1).strip(), m.group(2).strip(),
                             m.group(3).strip()))
        elif section == "idea":
            m = _CHECK_IDEA_RE.match(line)
            if m:
                ideas.append((m.group(1).strip(), m.group(2).strip(),
                              m.group(3).strip()))
```

Return `{"links": links, "tags": tags, "mocs": mocs, "ideas": ideas}` and update the docstring accordingly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 136 tests … OK (skipped=1)`

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "Parse MOC placement and product-idea sections in reviews

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Apply writers — MOC section insert and idea related-notes append

**Files:**
- Modify: `trellis.py` — new pure helpers after `merge_frontmatter_tags` (line ~1470), wire into `_apply_review_file` (line ~1533) and `cmd_apply` (line ~1495)
- Test: `tests/test_trellis.py`

**Interfaces:**
- Consumes: Task 4's `parse_review` keys `mocs`/`ideas`; existing `_LINK_RE`, `title_to_rel` mapping inside `_apply_review_file`.
- Produces: `insert_into_section(content: str, section: str, line: str) -> str | None` — insert `line` as the last item of the named `##`/`###` section (case-insensitive heading match); `None` if the heading is missing; unchanged content if the line's `[[target]]` already appears in that section (idempotent).
- Produces: `append_related_note(content: str, note_title: str, reason: str, date_str: str) -> str` — append `- [[note]] — reason` under the first heading matching `## Related notes` (prefix match, so legacy "added by Claude" variants are reused); creates `## Related notes (added YYYY-MM-DD)` at EOF if absent; idempotent per note title.
- Changes: `_apply_review_file` returns `(links, tags, mocs, ideas, sources)`; `cmd_apply` totals updated.

- [ ] **Step 1: Write the failing tests**

```python
class TestInsertIntoSection(unittest.TestCase):
    MOC = ("---\ntags: [moc]\n---\n# Cycling MOC\n\n## Training physiology\n\n"
           "- [[Existing note]]\n\n## Gear\n\n- [[Some bike]]\n")

    def test_inserts_at_end_of_section(self):
        out = t.insert_into_section(self.MOC, "Training physiology", "- [[New note]]")
        self.assertIsNotNone(out)
        idx_new = out.index("[[New note]]")
        self.assertGreater(idx_new, out.index("[[Existing note]]"))
        self.assertLess(idx_new, out.index("## Gear"))

    def test_heading_match_case_insensitive(self):
        self.assertIsNotNone(
            t.insert_into_section(self.MOC, "training PHYSIOLOGY", "- [[N]]"))

    def test_missing_heading_returns_none(self):
        self.assertIsNone(t.insert_into_section(self.MOC, "Nutrition", "- [[N]]"))

    def test_idempotent(self):
        once = t.insert_into_section(self.MOC, "Gear", "- [[Some bike]]")
        self.assertEqual(once, self.MOC)  # already present in that section

    def test_last_section_of_file(self):
        out = t.insert_into_section(self.MOC, "Gear", "- [[New saddle]]")
        self.assertTrue(out.rstrip().endswith("- [[New saddle]]"))


class TestAppendRelatedNote(unittest.TestCase):
    def test_creates_section_at_eof(self):
        out = t.append_related_note("# Idea\n\nBody.\n", "Note A", "why", "2026-07-19")
        self.assertIn("## Related notes (added 2026-07-19)", out)
        self.assertIn("- [[Note A]] — why", out)

    def test_reuses_legacy_claude_section(self):
        content = ("# Idea\n\n## Related notes (added by Claude on 2026-05-01)\n\n"
                   "- [[Old note]] — old why\n")
        out = t.append_related_note(content, "Note A", "why", "2026-07-19")
        self.assertEqual(out.count("## Related notes"), 1)
        self.assertIn("- [[Note A]] — why", out)

    def test_idempotent_per_note(self):
        content = "# Idea\n\n## Related notes\n\n- [[Note A]] — earlier why\n"
        self.assertEqual(
            t.append_related_note(content, "Note A", "new why", "2026-07-19"),
            content)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trellis.TestInsertIntoSection tests.test_trellis.TestAppendRelatedNote -v`
Expected: ERROR — no attribute `insert_into_section`

- [ ] **Step 3: Implement the pure helpers**

After `merge_frontmatter_tags` in `trellis.py`:

```python
_SECTION_HDR_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_RELATED_HDR_RE = re.compile(r"(?m)^##\s+(Related notes\b[^\n]*)$")


def insert_into_section(content: str, section: str, line: str) -> str | None:
    """Insert `line` as the last item of the named ##/### section. Returns None
    if no such heading exists (never guess a section); returns content
    unchanged if the line's [[target]] already appears in the section."""
    lines = content.splitlines()
    start = level = None
    want = section.strip().lower()
    for i, ln in enumerate(lines):
        m = _SECTION_HDR_RE.match(ln)
        if m and m.group(2).strip().lower() == want:
            start, level = i, len(m.group(1))
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"^(#{1,6})\s+", lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    probe = re.search(r"\[\[([^\]|#]+)", line)
    if probe:
        needle = f"[[{probe.group(1).strip().lower()}"
        if any(needle in ln.lower() for ln in lines[start:end]):
            return content
    last = start
    for j in range(start + 1, end):
        if lines[j].strip():
            last = j
    lines.insert(last + 1, line)
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def append_related_note(content: str, note_title: str, reason: str,
                        date_str: str) -> str:
    """Append '- [[note]] — reason' under the first '## Related notes…' heading
    (prefix match — legacy 'added by Claude' sections are reused, not
    duplicated), creating the section at EOF if absent. Idempotent per note."""
    line = f"- [[{note_title}]] — {reason}" if reason else f"- [[{note_title}]]"
    m = _RELATED_HDR_RE.search(content)
    if not m:
        base = content.rstrip()
        prefix = base + "\n\n" if base else ""
        return f"{prefix}## Related notes (added {date_str})\n\n{line}\n"
    updated = insert_into_section(content, m.group(1), line)
    return updated if updated is not None else content
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run: `python3 -m unittest tests.test_trellis.TestInsertIntoSection tests.test_trellis.TestAppendRelatedNote -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Wire into `_apply_review_file` and `cmd_apply`**

In `_apply_review_file`:

1. Empty check (line ~1537) becomes:

```python
    if not any(review[k] for k in ("links", "tags", "mocs", "ideas")):
```

2. After the `add_tags` grouping (line ~1555), add:

```python
    add_mocs: dict[str, list] = collections.defaultdict(list)   # moc -> [(note, section)]
    add_ideas: dict[str, list] = collections.defaultdict(list)  # idea -> [(note, reason)]
    for note_t, moc_t, section in review["mocs"]:
        add_mocs[moc_t].append((note_t, section))
    for note_t, idea_t, reason in review["ideas"]:
        add_ideas[idea_t].append((note_t, reason))
```

3. After the existing per-source loop (before the final dry-run/archive block, line ~1607), add:

```python
    applied_mocs = applied_ideas = 0
    date_str = datetime.date.today().isoformat()

    def _edit_target(title, kind, editor):
        """Apply editor(content) to the note titled `title`; count via return."""
        rel = title_to_rel.get(title.lower())
        if not rel:
            print(f"  ! {kind} target not found, skipping: {title}", file=sys.stderr)
            return 0
        full = os.path.join(cfg["vault"], rel)
        content = open(full, encoding="utf-8").read()
        updated, count = editor(rel, content)
        if count and not dry_run and updated != content:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(updated)
        return count

    for moc_t in sorted(add_mocs):
        def edit(rel, content, moc_t=moc_t):
            count = 0
            for note_t, section in add_mocs[moc_t]:
                updated = insert_into_section(content, section, f"- [[{note_t}]]")
                if updated is None:
                    print(f"  ! section '{section}' not found in {moc_t}; "
                          f"skipping [[{note_t}]]", file=sys.stderr)
                    continue
                if updated != content:
                    content = updated
                    count += 1
                    note_rel = title_to_rel.get(note_t.lower())
                    if note_rel and not dry_run:
                        conn.execute("UPDATE suggestions SET status='applied' "
                                     "WHERE path=? AND kind='moc' AND value=?",
                                     (note_rel, moc_t))
                    print(f"  {'~' if dry_run else '✓'} [[{note_t}]] → "
                          f"[[{moc_t}]] § {section}")
                else:
                    print(f"  = already in {moc_t}: [[{note_t}]]")
            return content, count
        applied_mocs += _edit_target(moc_t, "MOC", edit)

    for idea_t in sorted(add_ideas):
        def edit(rel, content, idea_t=idea_t):
            count = 0
            for note_t, reason in add_ideas[idea_t]:
                updated = append_related_note(content, note_t, reason, date_str)
                if updated != content:
                    content = updated
                    count += 1
                    note_rel = title_to_rel.get(note_t.lower())
                    if note_rel and not dry_run:
                        conn.execute("UPDATE suggestions SET status='applied' "
                                     "WHERE path=? AND kind='idea' AND value=?",
                                     (note_rel, idea_t))
                    print(f"  {'~' if dry_run else '✓'} [[{note_t}]] → "
                          f"[[{idea_t}]] (related note)")
                else:
                    print(f"  = already related to {idea_t}: [[{note_t}]]")
            return content, count
        applied_ideas += _edit_target(idea_t, "idea", edit)
```

4. Update the summary print and return:

```python
    head = "DRY RUN — would apply" if dry_run else "applied"
    print(f"\n{head}: {applied_links} link(s) · {applied_tags} tag(s) · "
          f"{applied_mocs} MOC placement(s) · {applied_ideas} idea link(s) "
          f"across {len(sources)} source note(s)")
```

and `return applied_links, applied_tags, applied_mocs, applied_ideas, len(sources)`.
Also update the early-return on the empty-check path to `return 0, 0, 0, 0, 0`.

5. In `cmd_apply`, update the accumulation (line ~1516-1529):

```python
    tot = [0, 0, 0, 0, 0]
    for path in paths:
        if multi:
            print(f"\n── {os.path.basename(path)} ──")
        for i, v in enumerate(_apply_review_file(cfg, path, args.dry_run)):
            tot[i] += v

    if multi:
        head = "DRY RUN — would apply" if args.dry_run else "applied"
        print(f"\nTOTAL {head}: {tot[0]} link(s) · {tot[1]} tag(s) · "
              f"{tot[2]} MOC placement(s) · {tot[3]} idea link(s) "
              f"across {tot[4]} source note(s) in {len(paths)} review(s)")
```

- [ ] **Step 6: Run the full suite**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 144 tests … OK (skipped=1)` — if any existing apply-related test asserts the 3-tuple return, update it to the 5-tuple.

- [ ] **Step 7: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "Apply MOC placements and product-idea links from reviews

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Triage renderer, MOC/idea prompts, heading extraction

**Files:**
- Modify: `trellis.py` — prompts near `TAG_PROMPT` (line ~680), `moc_headings` near `parse_outlinks`, `render_triage_report` next to `render_report`
- Test: `tests/test_trellis.py`

**Interfaces:**
- Produces: `moc_headings(body: str) -> list[str]` — text of all `##`/`###` headings, in order.
- Produces: `MOC_PLACE_PROMPT` with `{moc}`, `{sections}`, `{title}`, `{excerpt}` slots → model returns `{"section": <exact text or null>, "reason": ...}`.
- Produces: `IDEA_PROMPT` with `{idea}`, `{idea_excerpt}`, `{title}`, `{excerpt}` slots → model returns `{"related": bool, "reason": ...}`.
- Produces: `render_triage_report(date_str, summary, tag_items, moc_items, idea_items, untouched, suspected) -> str` where `summary` has keys `new_notes`/`new_tags`/`new_mocs`/`new_ideas`; `tag_items` = `[{"source", "tags"}]` (same shape garden uses); `moc_items` = `[{"source", "moc", "section", "reason"}]`; `idea_items` = `[{"source", "idea", "reason"}]`; `untouched` = `[(title, why)]`; `suspected` = `detect_new_notes`'s second return.

- [ ] **Step 1: Write the failing tests**

```python
class TestMocHeadings(unittest.TestCase):
    def test_extracts_h2_h3_in_order(self):
        body = "# Title\n\n## Alpha\ntext\n### Beta\n\n## Gamma\n#### too deep\n"
        self.assertEqual(t.moc_headings(body), ["Alpha", "Beta", "Gamma"])

    def test_empty(self):
        self.assertEqual(t.moc_headings("no headings here"), [])


class TestRenderTriageReport(unittest.TestCase):
    def _md(self):
        return t.render_triage_report(
            "2026-07-19",
            {"new_notes": 2, "new_tags": 2, "new_mocs": 1, "new_ideas": 1},
            tag_items=[{"source": "Note A", "tags": ["endurance", "physiology"]}],
            moc_items=[{"source": "Note A", "moc": "Cycling MOC",
                        "section": "Training physiology", "reason": "mechanism note"}],
            idea_items=[{"source": "Note A", "idea": "Training Load Dashboard",
                         "reason": "evidence for metric"}],
            untouched=[("Note B", "no MOC fit")],
            suspected=[("2026-07-16 09:41", ["z/old1.md", "z/old2.md"])])

    def test_sections_and_grammar_round_trip(self):
        md = self._md()
        self.assertIn("# Review — 2026-07-19", md)
        self.assertIn("_Triage: 2 new note(s)", md)
        # every suggestion line must round-trip through parse_review when checked
        checked = md.replace("- [ ]", "- [x]")
        r = t.parse_review(checked)
        self.assertEqual(r["tags"], [("Note A", ["endurance", "physiology"])])
        self.assertEqual(r["mocs"],
                         [("Note A", "Cycling MOC", "Training physiology")])
        self.assertEqual(r["ideas"],
                         [("Note A", "Training Load Dashboard", "evidence for metric")])

    def test_informational_sections_not_parsed(self):
        checked = self._md().replace("- [ ]", "- [x]")
        r = t.parse_review(checked)
        flat = [s for s, *_ in r["mocs"]] + [s for s, *_ in r["ideas"]]
        self.assertNotIn("Note B", flat)          # untouched list is inert

    def test_prompts_have_slots(self):
        p = t.MOC_PLACE_PROMPT.format(moc="M", sections="- A", title="T", excerpt="E")
        self.assertIn('"section"', p)
        p2 = t.IDEA_PROMPT.format(idea="I", idea_excerpt="X", title="T", excerpt="E")
        self.assertIn('"related"', p2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trellis.TestMocHeadings tests.test_trellis.TestRenderTriageReport -v`
Expected: ERROR — no attribute `moc_headings`

- [ ] **Step 3: Implement**

Near `parse_outlinks`:

```python
_MOC_HEADING_RE = re.compile(r"(?m)^(#{2,3})\s+(.+?)\s*$")


def moc_headings(body: str) -> list[str]:
    """Text of the ##/### headings in a MOC body — the placement targets the
    gen model chooses among."""
    return [m.group(2) for m in _MOC_HEADING_RE.finditer(body)]
```

After `TAG_PROMPT`:

```python
MOC_PLACE_PROMPT = """You maintain Maps of Content (MOCs) for a Zettelkasten. \
Decide whether the NOTE below clearly earns a place in this MOC, and if so \
under which existing section. Be conservative: most notes do NOT belong in a \
MOC; only place a note that is a strong fit for one specific section.

MOC: "{moc}"
SECTIONS:
{sections}

NOTE: "{title}"
{excerpt}

Return JSON only. section must be the exact text of one SECTIONS entry, or \
null if the note does not clearly belong:
{{"section": "<exact section text or null>", "reason": "<10 words max>"}}"""


IDEA_PROMPT = """You evaluate whether a Zettelkasten NOTE genuinely helps a \
PRODUCT IDEA — as evidence, positioning, or a competitive or philosophical \
companion. Be conservative: answer false unless the connection is direct and \
useful.

PRODUCT IDEA: "{idea}"
{idea_excerpt}

NOTE: "{title}"
{excerpt}

Return JSON only:
{{"related": true or false, "reason": "<10 words max>"}}"""
```

Next to `render_report`:

```python
def render_triage_report(date_str: str, summary: dict, tag_items: list,
                         moc_items: list, idea_items: list,
                         untouched: list, suspected: list) -> str:
    """Render the triage half of the shared review file. Pure for testing."""
    L = [f"# Review — {date_str}", ""]
    L.append(f"_Triage: {summary['new_notes']} new note(s) · "
             f"{summary['new_tags']} tag suggestion(s) · "
             f"{summary['new_mocs']} MOC placement(s) · "
             f"{summary['new_ideas']} idea link(s)._")
    L.append("")
    L.append("> Check the boxes you want, then run `trellis apply <this file>`. "
             "Nothing here has been written to your notes.")
    L.append("")
    if tag_items:
        L.append("## Tag suggestions\n")
        for it in tag_items:
            tags = " ".join(f"`{x}`" for x in it["tags"])
            L.append(f"- [ ] [[{it['source']}]] → {tags}")
        L.append("")
    if moc_items:
        L.append("## MOC placements\n")
        for it in moc_items:
            L.append(f"- [ ] [[{it['source']}]] → [[{it['moc']}]] "
                     f"§ {it['section']} — {it['reason']}")
        L.append("")
    if idea_items:
        L.append("## Product idea links\n")
        for it in idea_items:
            L.append(f"- [ ] [[{it['source']}]] → [[{it['idea']}]] — {it['reason']}")
        L.append("")
    if untouched:
        L.append("## Notes with no suggestions\n")
        for title, why in untouched:
            L.append(f"- [[{title}]] — {why}")
        L.append("")
    if suspected:
        L.append("## Suspected bulk-touch clusters (excluded — review manually)\n")
        for key, rels in suspected:
            L.append(f"- {key} · {len(rels)} file(s) (e.g. {rels[0]})")
        L.append("")
    if not (tag_items or moc_items or idea_items):
        L.append("_No suggestions for this batch._")
    return "\n".join(L)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 150 tests … OK (skipped=1)`

- [ ] **Step 5: Commit**

```bash
git add trellis.py tests/test_trellis.py
git commit -m "Add triage renderer, placement prompts, and MOC heading parser

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: cmd_triage, config defaults, CLI wiring

The command itself: mostly orchestration of Tasks 1–6 plus existing plumbing. Commands are not unit-tested in this repo (only pure helpers are), so verification is the suite staying green plus a live smoke test.

**Files:**
- Modify: `trellis.py` — `DEFAULTS` (line ~83, in the phase-3 block area), `_scan_vault` (line ~990), new `cmd_triage` before the CLI section (line ~1619), `main()` (line ~1625)

**Interfaces:**
- Consumes: `detect_new_notes`, `extract_created`, `seed_triage_state`, `_ensure_triage_tables` (Tasks 1–2); `append_or_create_review`, `render_triage_report`, `moc_headings`, `MOC_PLACE_PROMPT`, `IDEA_PROMPT` (Tasks 3, 6); existing `_scan_vault`, `_load_matrix`, `top_k`, `candidate_tags`, `classify_tag_suggestions`, `generate_json`, `read_note`, `split_frontmatter`.
- Produces: `trellis triage [--limit N] [--scope prefixes] [--gen-model M] [--force] [--dry-run]`.
- Changes: `_scan_vault` entries gain `"created"` (from `extract_created(fm)`) and `"mtime"` (`os.path.getmtime(full)`) keys.

- [ ] **Step 1: Add config defaults**

In `DEFAULTS`, after the phase-3 cluster block, add:

```python
    # --- triage (phase 4) ---
    "triage_scope": ["z/"],            # path prefixes triage watches for new notes
    "idea_scope": ["Areas/Product Ideas/"],  # where product-idea files live
    "triage_bulk_min": 8,              # mtime-minute bucket >= this ⇒ suspected bulk touch
    "triage_tag_skip_threshold": 3,    # skip tag step when a note already has >= this many
    "moc_place_threshold": 0.55,       # note↔MOC cosine gate (provisional; tune)
    "idea_link_threshold": 0.55,       # note↔idea cosine gate (provisional; tune)
```

Verify `moc_scope` already exists in `DEFAULTS` (used by `cmd_cluster`); if its default is commented out in favor of code-level fallback, mirror however `cmd_cluster` reads it.

- [ ] **Step 2: Extend `_scan_vault`**

In the `notes[rel] = {...}` dict, add two keys:

```python
            "created": extract_created(fm),
            "mtime": os.path.getmtime(full),
```

- [ ] **Step 3: Implement `cmd_triage`**

Insert before the CLI section (line ~1619):

```python
# --------------------------------------------------------------------------- #
# Triage (phase 4): new-note tags / MOC placement / idea links -> review queue
# --------------------------------------------------------------------------- #
def cmd_triage(cfg, args):
    if not _require_vault(cfg):
        return 1
    vault = cfg["vault"]
    scope = tuple(args.scope.split(",")) if args.scope else tuple(cfg["triage_scope"])
    gen_model = cfg["gen_model"]

    conn = connect(cfg["db_path"])
    _ensure_garden_tables(conn)
    _ensure_triage_tables(conn)

    # One-time seed from the note-triage skill's state file, if present.
    state_json = os.path.join(vault, "_workspace", "triage-state.json")
    if os.path.exists(state_json):
        try:
            with open(state_json, encoding="utf-8") as fh:
                imported = seed_triage_state(conn, json.load(fh))
            if imported:
                print(f"seeded triage state from triage-state.json ({imported} notes)")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"warning: could not read {state_json}: {e}", file=sys.stderr)

    last_run = meta_get(conn, "triage_last_run")
    if last_run is None:
        # First run, nothing to seed from: baseline only. Everything currently
        # in scope is treated as pre-existing; later notes get triaged.
        meta_set(conn, "triage_last_run", datetime.datetime.now().isoformat())
        conn.commit()
        print("no triage state found — baseline initialized; "
              "notes created from now on will be triaged")
        return 0
    cutoff = datetime.datetime.fromisoformat(last_run)

    paths, titles, mat = _load_matrix(conn)
    if not paths:
        print("index is empty — run:  trellis index", file=sys.stderr)
        return 1
    rel_to_idx = {p: i for i, p in enumerate(paths)}

    print("scanning vault…", flush=True)
    notes = _scan_vault(vault, set(cfg["exclude_dirs"]), cfg["max_chars"])
    vault_tags = {tag.lower() for n in notes.values() for tag in n["tags"]}
    triaged = (set() if args.force else
               {row[0] for row in conn.execute("SELECT path FROM triage_state")})

    entries = [(rel, n["created"], n["mtime"])
               for rel, n in notes.items() if rel.startswith(scope)]
    candidates, suspected = detect_new_notes(entries, cutoff, triaged,
                                             cfg["triage_bulk_min"])
    candidates = [r for r in candidates if r in rel_to_idx]
    limited = bool(args.limit) and len(candidates) > args.limit
    if limited:
        print(f"limiting to {args.limit} of {len(candidates)} candidates "
              "(rest queued for the next run)")
        candidates = candidates[:args.limit]

    for key, rels in suspected:
        print(f"suspected bulk touch at {key}: {len(rels)} file(s) excluded",
              file=sys.stderr)
    if not candidates:
        print(f"no new notes since {cutoff.date().isoformat()}")
        return 0
    print(f"scope {scope} · {len(candidates)} new note(s) since "
          f"{cutoff.date().isoformat()}\n", flush=True)

    moc_scope = tuple(cfg["moc_scope"])
    idea_scope = tuple(cfg["idea_scope"])
    moc_rows = [(i, titles[i]) for i, p in enumerate(paths) if p.startswith(moc_scope)]
    idea_rows = [(i, titles[i]) for i, p in enumerate(paths) if p.startswith(idea_scope)]
    if not moc_rows:
        print(f"note: no indexed notes under {moc_scope} — skipping MOC placement",
              file=sys.stderr)
    if not idea_rows:
        print(f"note: no indexed notes under {idea_scope} — skipping idea links",
              file=sys.stderr)

    seen = {(row[0], row[1], row[2]) for row in
            conn.execute("SELECT path, kind, value FROM suggestions").fetchall()}
    now = time.time()
    tag_items, moc_items, idea_items, untouched = [], [], [], []
    n_tags = n_mocs = n_ideas = 0

    for n, rel in enumerate(candidates, 1):
        note = notes[rel]
        idx = rel_to_idx[rel]
        got, why = 0, []

        # ---- tags (garden pipeline, no thin-note gate) ----
        if len(note["tags"]) >= cfg["triage_tag_skip_threshold"]:
            why.append("already tagged")
        else:
            neigh = [(titles[i], paths[i]) for i, _ in
                     top_k(mat[idx], mat, cfg["tag_candidate_neighbors"] + 1)
                     if i != idx]
            ntags = [notes.get(pp, {}).get("tags", []) for _, pp in neigh]
            cand_tags = candidate_tags(ntags, set(note["tags"]), 20)
            picked = []
            if cand_tags:
                prompt = TAG_PROMPT.format(title=note["title"],
                                           excerpt=note["excerpt"][:1200],
                                           tags=", ".join(cand_tags))
                try:
                    out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                        timeout=cfg["gen_timeout"],
                                        num_predict=cfg["gen_num_predict"])
                except OllamaError:
                    out = {}
                picked, _ = classify_tag_suggestions(
                    out.get("tags", []), [], cand_tags, vault_tags, note["tags"])
            fresh = [x for x in picked if (rel, "tag", x) not in seen]
            if fresh:
                for x in fresh:
                    seen.add((rel, "tag", x))
                    if not args.dry_run:
                        conn.execute("INSERT OR IGNORE INTO suggestions VALUES(?,?,?,?,?,?)",
                                     (rel, "tag", x, "", now, "new"))
                tag_items.append({"source": note["title"], "tags": fresh})
                n_tags += len(fresh)
                got += len(fresh)
            else:
                why.append("no tag fit")

        # ---- MOC placement ----
        placed = False
        if moc_rows:
            sim, mi, mtitle = max(
                (float(mat[idx] @ mat[i]), i, ttl) for i, ttl in moc_rows)
            if sim >= cfg["moc_place_threshold"] and (rel, "moc", mtitle) not in seen:
                moc_raw = read_note(os.path.join(vault, paths[mi]))
                headings = []
                if moc_raw:
                    _, moc_body = split_frontmatter(moc_raw.decode("utf-8", "replace"))
                    headings = moc_headings(moc_body)
                if headings:
                    prompt = MOC_PLACE_PROMPT.format(
                        moc=mtitle,
                        sections="\n".join(f"- {h}" for h in headings),
                        title=note["title"], excerpt=note["excerpt"][:1200])
                    try:
                        out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                            timeout=cfg["gen_timeout"],
                                            num_predict=cfg["gen_num_predict"])
                    except OllamaError:
                        out = {}
                    section = out.get("section")
                    by_lower = {h.lower(): h for h in headings}
                    if isinstance(section, str) and section.strip().lower() in by_lower:
                        section = by_lower[section.strip().lower()]
                        reason = " ".join(str(out.get("reason", "")).split()[:10])[:120]
                        seen.add((rel, "moc", mtitle))
                        if not args.dry_run:
                            conn.execute("INSERT OR IGNORE INTO suggestions VALUES(?,?,?,?,?,?)",
                                         (rel, "moc", mtitle,
                                          f"{section} — {reason}", now, "new"))
                        moc_items.append({"source": note["title"], "moc": mtitle,
                                          "section": section, "reason": reason})
                        n_mocs += 1
                        got += 1
                        placed = True
        if moc_rows and not placed:
            why.append("no MOC fit")

        # ---- Product idea links (top 3 ideas above the gate) ----
        linked = False
        if idea_rows:
            ranked = sorted(((float(mat[idx] @ mat[i]), i, ttl)
                             for i, ttl in idea_rows), reverse=True)
            for sim, ii, ititle in ranked[:3]:
                if sim < cfg["idea_link_threshold"] or (rel, "idea", ititle) in seen:
                    continue
                prompt = IDEA_PROMPT.format(
                    idea=ititle,
                    idea_excerpt=notes.get(paths[ii], {}).get("excerpt", "")[:600],
                    title=note["title"], excerpt=note["excerpt"][:1200])
                try:
                    out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                        timeout=cfg["gen_timeout"],
                                        num_predict=cfg["gen_num_predict"])
                except OllamaError:
                    out = {}
                if out.get("related") is True:
                    reason = " ".join(str(out.get("reason", "")).split()[:10])[:120]
                    seen.add((rel, "idea", ititle))
                    if not args.dry_run:
                        conn.execute("INSERT OR IGNORE INTO suggestions VALUES(?,?,?,?,?,?)",
                                     (rel, "idea", ititle, reason, now, "new"))
                    idea_items.append({"source": note["title"], "idea": ititle,
                                       "reason": reason})
                    n_ideas += 1
                    got += 1
                    linked = True
        if idea_rows and not linked:
            why.append("no idea fit")

        if not got:
            untouched.append((note["title"], "; ".join(why) or "nothing to add"))
        if not args.dry_run:
            conn.execute("INSERT OR IGNORE INTO triage_state VALUES(?,?)", (rel, now))
            conn.commit()
        print(f"  [{n}/{len(candidates)}] {note['title'][:50]}  (+{got})", flush=True)

    date_str = datetime.date.today().isoformat()
    summary = {"new_notes": len(candidates), "new_tags": n_tags,
               "new_mocs": n_mocs, "new_ideas": n_ideas}
    report = render_triage_report(date_str, summary, tag_items, moc_items,
                                  idea_items, untouched, suspected)

    if args.dry_run:
        print("\n--- DRY RUN (report not written) ---\n")
        print(report)
        return 0
    out_path = append_or_create_review(
        os.path.join(vault, cfg["gardener_dir"]), date_str, report)
    if not limited:
        # Advance the cutoff only when nothing was left behind — a --limit run
        # relies on triage_state alone so the remainder surfaces next time.
        meta_set(conn, "triage_last_run", datetime.datetime.now().isoformat())
    conn.commit()
    print(f"\nreview queue → {out_path}")
    print(f"  {n_tags} tag · {n_mocs} MOC placement · {n_ideas} idea link "
          f"suggestion(s) across {len(candidates)} new note(s)")
    return 0
```

- [ ] **Step 4: Wire the CLI**

In `main()`, after the `apply` subparser:

```python
    pt = sub.add_parser("triage",
                        help="triage new notes -> tag/MOC/idea suggestions")
    pt.add_argument("--limit", type=int,
                    help="max new notes this run (rest queued for next run)")
    pt.add_argument("--scope", help="comma-separated path prefixes (default: z/)")
    pt.add_argument("--gen-model", dest="gen_model", help="judgment model")
    pt.add_argument("--force", action="store_true",
                    help="ignore triage state; re-triage matching notes")
    pt.add_argument("--dry-run", action="store_true",
                    help="print the report; write nothing (no ledger, no state, no file)")
```

and add `"triage": cmd_triage,` to the dispatch dict.

- [ ] **Step 5: Run the full suite and a CLI sanity check**

Run: `python3 -m unittest discover -s tests -q`
Expected: `Ran 150 tests … OK (skipped=1)` (no new unit tests this task)

Run: `python3 trellis.py triage --help`
Expected: usage text listing `--limit`, `--scope`, `--gen-model`, `--force`, `--dry-run`

- [ ] **Step 6: Live smoke test (requires Ollama + configured vault; skip if unavailable)**

Run: `python3 trellis.py triage --dry-run --limit 2`
Expected: either `no new notes since <date>`, a baseline-initialization message, or a printed dry-run report with triage sections. Must not traceback. Nothing is written in any case.

- [ ] **Step 7: Commit**

```bash
git add trellis.py
git commit -m "Add trellis triage: new-note tags, MOC placement, idea links

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Nightly chaining and docs

**Files:**
- Modify: `run-garden.sh` (line 12)
- Modify: `trellis.toml.example` (append triage block)
- Modify: `README.md` (Features list + a short triage section near the garden/apply docs)

**Interfaces:**
- Consumes: the `triage` subcommand from Task 7. No code interfaces produced.

- [ ] **Step 1: Chain triage into the nightly wrapper**

In `run-garden.sh`, between the `index` and `garden` lines, insert (triage is non-fatal by construction — each line's failure doesn't stop the script):

```zsh
"$PY" "$DIR/trellis.py" triage >> "$LOG" 2>&1
```

and update the header comment to `# Nightly trellis run: refresh the embedding index, triage new notes, then garden.`

- [ ] **Step 2: Document config keys**

Append to `trellis.toml.example`:

```toml
# --- new-note triage (phase 4) ---
# Triage watches these prefixes for notes newly created since the last run
# (created:/published: frontmatter preferred; mtime fallback with a bulk-touch
# guard). Suggestions go into the shared gardener review file.
triage_scope        = ["z/"]
idea_scope          = ["Areas/Product Ideas/"]  # product-idea files (must be indexed)
triage_bulk_min     = 8      # >= this many notes sharing an mtime-minute => bulk touch, excluded
triage_tag_skip_threshold = 3  # skip tag suggestions when a note already has this many
moc_place_threshold = 0.55   # note<->MOC cosine gate (provisional; tune after first run)
idea_link_threshold = 0.55   # note<->idea cosine gate (provisional; tune after first run)
```

- [ ] **Step 3: Update the README**

Add a Features bullet after the "Nightly gardener" bullet:

```markdown
- **New-note triage** — detects notes newly added to `z/` (trusting `created:`
  frontmatter over sync-scrambled mtimes), then suggests frontmatter tags, a
  placement in the best-fitting MOC section, and links into relevant
  `Areas/Product Ideas/` files. Suggestions land in the same dated review file
  as the gardener's; `trellis apply` writes the checked ones.
```

In the usage/examples part of the README (near where `garden`/`apply` are shown), add a short subsection:

```markdown
### Triage new notes

```sh
trellis triage             # suggest tags / MOC placement / idea links for new notes
trellis triage --dry-run   # print the report without writing anything
trellis apply              # apply all checked items from pending reviews
```

The first run initializes a baseline (or imports `_workspace/triage-state.json`
if the earlier Claude-skill workflow left one); after a backlog import, run
with `--limit` a few times rather than triaging weeks of notes at once.
```

Also update the "Apply step" feature bullet to mention MOC placements and idea
links alongside tags/links, and the nightly-scheduler sentence to say the
launchd job runs `index → triage → garden`.

- [ ] **Step 4: Verify**

Run: `python3 -m unittest discover -s tests -q`
Expected: still green.

Run: `zsh -n run-garden.sh`
Expected: no output (syntax OK).

- [ ] **Step 5: Commit**

```bash
git add run-garden.sh trellis.toml.example README.md
git commit -m "Chain triage into the nightly run; document phase 4

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** detection incl. bulk guard (Task 1), db state + seed (Task 2), unified review file + append-if-pending (Task 3), review grammar + parser (Tasks 4, 6), apply writers incl. missing-heading skip and legacy Related-notes reuse (Task 5), shared ledger with `moc`/`idea` kinds (Tasks 5, 7), thresholds + config (Task 7), `--dry-run`/`--force`/`--limit` semantics incl. not advancing the cutoff on limited runs (Task 7), nightly chain + docs (Task 8). Spec's "no candidates ⇒ update nothing" honored (early return before any state write).
- **Type consistency:** `parse_review` keys `mocs`=(note, moc, section) / `ideas`=(note, idea, reason) match the Task 6 renderer grammar and Task 5 consumers; `_apply_review_file` 5-tuple matches `cmd_apply`; `summary` keys match between `render_triage_report` and `cmd_triage`.
- **Known judgment calls baked in:** MOC placement considers only the single best-scoring MOC per note; ideas consider the top 3 above the gate; ledger value for `moc` is the MOC title (a rejected placement won't be re-suggested for the same MOC under a different section — acceptable, matches "considered once" semantics).

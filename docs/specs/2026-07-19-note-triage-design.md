# Phase 4 — New-note triage (design)

Date: 2026-07-19
Status: draft, awaiting user review

## Goal

Bring the weekly note-triage workflow (today a Claude Code skill run against the
vault) into trellis: detect notes newly added to `z/` since the last run, and for
each one suggest (a) frontmatter tags, (b) MOC membership with a specific
sub-section, and (c) relevance to items in `Areas/Product Ideas/` — all emitted
as a dated checkbox review file that `trellis apply` writes back.

## Decisions (from brainstorming)

These were made autonomously (interactive questions unavailable this session) —
each is overridable before implementation.

| Question | Decision |
|---|---|
| Apply model | **Review queue, not apply-then-report.** The skill applies high-confidence changes directly; trellis's core promise is "never edits notes directly." Triage emits a review file; `trellis apply` writes checked items. A local 35B's judgment also warrants a human gate more than Claude's did. |
| Command shape | **New `trellis triage` command** plus extensions to `apply`. Not folded into `garden`: garden's lifecycle is "re-visit changed/thin notes," triage's is "visit each note exactly once when new" — different state, different cadence. |
| New-note detection | **Port the skill's logic:** trust `created:`/`published:` frontmatter date when present; fall back to mtime only for notes without it, with a bulk-touch guard (≥ `triage_bulk_min` notes sharing one mtime-minute ⇒ suspected sync/bulk event, reported but not triaged). This guard exists because iCloud sync and bulk scripts bump mtimes of old notes. |
| Triage state | **In `index.db`**, not a JSON file: table `triage_state(path PRIMARY KEY, triaged_at)` + `meta` key `triage_last_run`. One-time seed: if the table is empty and `_workspace/triage-state.json` exists, import its `triaged` list and `last_run_iso` so skill-era runs are respected. |
| When is a note "triaged"? | **When the review file is written**, not when applied. Emitting suggestions = the note has been considered; unchecked boxes = user declined. Avoids unapplied reviews blocking future runs. |
| MOC set | **Discovered from the index** via existing `moc_scope` config (default `MOCs/`), never hard-coded. The skill's hard-coded MOC list rotted twice already. |
| New-MOC candidates | **Out of scope** — `trellis cluster` already detects recurring themes with no covering MOC. The triage report links to it instead of duplicating the logic. |
| Tag suggestions | **Reuse the garden tag pipeline** (neighbor-tag candidate vocabulary + gen-model pick + `classify_tag_suggestions`), but run it for *every* new note, not only thin ones — new notes rarely have tags yet, and this replaces the skill's `tags-after.txt` vocabulary with the live vault vocabulary trellis already computes. |
| Shared suggestions ledger | **Triage reads and writes the same `suggestions` table as garden**, adding kinds `moc` and `idea` alongside `link`/`tag`. This is what guarantees cross-command dedup: whichever command surfaces a (path, kind, value) first owns it; the other filters it as seen. Load-bearing for the unified review file below. |
| Unified review file | **Triage and garden write sections into the same dated review file.** One output dir (existing `gardener_dir` — no new config key, no breaking rename), neutral `# Review — YYYY-MM-DD` title, each command appending its own summary line and sections. **Append-if-pending:** a command finding today's file still in the dir (applied files are archived away, so presence = pending) appends its sections instead of creating a suffixed sibling. `parse_review` is already section-driven, so the merged file needs no parser changes; no-arg `trellis apply` sweeps the shared dir as it does today. |
| Scheduling | **Chained into the nightly wrapper:** `run-garden.sh` runs `index → triage → garden`. Triage needs the new note's embedding (index first); garden's cold-start link pass then sweeps the same note the same night, so its links land beside its MOC/tag/idea suggestions in one file. A triage failure must not block garden (non-fatal step in the wrapper). Triage exits in milliseconds when there are no new notes. |

## Pipeline — `trellis triage`

Flags: `--dry-run`, `--force` (ignore triage state, re-triage), `--limit <n>`,
`--scope <prefixes>`.

1. **Detect new notes.** Scan `triage_scope` (default `z/`) via the existing
   `_scan_vault`. A note is a candidate if not in `triage_state` and:
   - its `created:`/`published:` frontmatter date (regex on frontmatter, same
     pattern as the skill) is after `triage_last_run`'s date — mtime ignored; or
   - it has no `created:` field and mtime > `triage_last_run`, and its
     mtime-minute bucket holds fewer than `triage_bulk_min` notes.

   Suspected bulk buckets are listed in the report header area ("excluded —
   review manually"), never auto-triaged. No candidates ⇒ print "no new notes
   since <date>", write nothing, update nothing.

2. **Per note — tags.** Same machinery as garden's tag step (`candidate_tags` →
   gen model → `classify_tag_suggestions`), unconditionally (no thin-note gate).
   Skip only notes that already have ≥ `triage_tag_skip_threshold` (default 3) tags.

3. **Per note — MOC placement.** Cosine similarity between the note's stored
   embedding and each MOC file's embedding (MOCs are already indexed under
   `moc_scope`). If the best score ≥ `moc_place_threshold` (default **0.55**,
   provisional — tune after first real run), one `generate_json` call: given the
   MOC's `##`/`###` heading list and the note's title + excerpt, return
   `{"section": "...", "reason": "..."}` or `{"section": null}` if it doesn't
   earn a place. Null/failed generation ⇒ no suggestion (the "clearly earns its
   place" bar from the skill, enforced by requiring the model to name a section).

4. **Per note — Product Idea links.** Same shape as step 3 against notes under
   `idea_scope` (default `Areas/Product Ideas/`, already indexed): similarity
   gate at `idea_link_threshold` (default **0.55**, provisional), then one
   gen-model call asking whether the note is evidence / positioning / a
   competitive or philosophical companion for the idea, returning a one-line
   reason or null.

5. **Emit review sections** into the shared dated review file in `gardener_dir`
   (create it, or append if today's file is pending — see Decisions), record new
   `moc`/`idea`/`tag` suggestions in the shared `suggestions` ledger, record each
   processed note in `triage_state`, set `triage_last_run` to now. `--dry-run`
   prints and writes nothing (no file, no ledger, no state), matching `garden`.

## Review file format

Extends the existing gardener grammar so `parse_review` stays one parser. In the
nightly chain, triage typically creates the file and garden appends its Link/Tag
sections below (garden's title line switches to the neutral form too):

```markdown
# Review — 2026-07-19

_Triage: 5 new note(s) · 9 tag suggestion(s) · 3 MOC placement(s) · 2 idea link(s)._

> Check the boxes you want, then run `trellis apply <this file>`.

## Tag suggestions
- [ ] [[Zone 2 and mitochondrial density]] → `endurance` `physiology`

## MOC placements
- [ ] [[Zone 2 and mitochondrial density]] → [[Cycling MOC]] § Training physiology — mechanism note for the aerobic-base cluster

## Product idea links
- [ ] [[Zone 2 and mitochondrial density]] → [[Training Load Dashboard]] — evidence for the aerobic-base metric

## Notes with no suggestions
- [[Some new note]] — no MOC above threshold; tags already present

## Suspected bulk-touch clusters (excluded — review manually)
- 2026-07-16 09:41 · 23 files (e.g. Old note.md)
```

The last two sections are informational; `apply` ignores them.

## Apply extensions

`parse_review` gains two sections; `_apply_review_file` gains two writers:

- **MOC placement:** insert `- [[note title]]` at the end of the named `##`/`###`
  section in the MOC file (before the next same-or-higher-level heading). If the
  heading no longer exists, warn and skip — never guess a section, never append
  to the file root.
- **Idea link:** append `- [[note title]] — <reason>` under a
  `## Related notes` section in the idea file, creating the section at EOF if
  absent. This matches the convention the skill established (existing
  `## Related notes (added by Claude …)` headings are matched by prefix, so old
  sections are reused, not duplicated).

Both are edits to non-`z/` files, consistent with the vault rule that `z/` note
bodies are never touched (tags still go through the frontmatter merge path).

## Configuration (new `trellis.toml` keys)

```toml
triage_scope        = ["z/"]
# review files share the existing gardener_dir — no separate triage_dir
idea_scope          = ["Areas/Product Ideas/"]
triage_bulk_min     = 8      # mtime-minute bucket size ⇒ suspected bulk touch
triage_tag_skip_threshold = 3
moc_place_threshold = 0.55   # provisional; tune after first run
idea_link_threshold = 0.55   # provisional; tune after first run
# moc_scope already exists (default "MOCs/")
```

## Testing

Following the house pattern — pure helpers, no network, join the fast suite:

- **New-note detection** as a pure function over `(name, created_date, mtime)`
  tuples + cutoff: created-trumps-mtime, bulk-bucket exclusion, already-triaged
  exclusion, missing-state first run.
- **`created:` frontmatter extraction** (both `created:` and `published:`, date
  formats seen in the vault).
- **MOC section insert**: end-of-section placement, `###` vs `##` boundaries,
  missing-heading skip, idempotency (re-apply doesn't duplicate the link).
- **Related-notes append**: section creation, prefix-matching legacy headings,
  dedup.
- **`parse_review`** round-trip for the two new sections, including a merged
  triage+garden file; renderer output.
- **Append-if-pending**: append vs. create decision (pending file present /
  absent / already archived), summary-line accumulation, and that a merged file
  round-trips through `parse_review` → apply → archive as one unit.
- **Prompt builders** for MOC placement and idea relevance (string-level).
- Gen-model calls stay behind `generate_json` (already mocked in existing tests).

## Scope / non-goals (YAGNI)

- No apply-then-report mode, no confidence-gated auto-apply.
- No new-MOC candidate logging (that's `trellis cluster`).
- No wikilink suggestions between new notes and z/ (that's `garden`; the natural
  cadence is to run `trellis garden` after triage, and new notes are exactly the
  cold-start notes garden's broad pass targets).
- No frontmatter *repair* (the skill fixed obviously-broken YAML in passing);
  trellis only merges tags via the existing frontmatter path. Broken frontmatter
  stays a manual fix.
- No migration of `_workspace/triage-log.md` — the dated review files + `applied/`
  archive are the log now.

## Open items (resolve empirically, not blockers)

- `moc_place_threshold` / `idea_link_threshold` defaults (0.55) need tuning
  against a real run — MOC files embed differently from atomic notes.
- Whether the gen model reliably names an *existing* section from the heading
  list, or needs the heading list echoed back with IDs to pick from. Decide
  during implementation with a few live probes.
- **First-run batch size.** The seed import takes `last_run_iso` from the skill's
  state file, so the first `trellis triage` covers everything since the last
  skill run — could be weeks of notes. Run it manually with `--limit` (and
  `--dry-run` first) before enabling the nightly chain.
- **Local-model judgment quality vs. Claude.** MOC section placement and idea
  relevance were previously Claude's calls. Eyeball the first real run's review
  file before trusting the thresholds; the checkbox gate contains the blast
  radius either way.

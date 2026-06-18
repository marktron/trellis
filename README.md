# trellis

A local-LLM gardener for an Obsidian vault. Structure for your notes to grow on.

Everything runs locally against [Ollama](https://ollama.com). No note ever leaves
the machine.

## Roadmap

- **Phase 1 — index + search** *(done)*: an incremental embedding index over
  the vault, plus semantic search and per-note neighbor lookup.
- **Phase 2 — nightly gardener** *(done)*: link suggestions (embeddings for
  recall → small model for precision), tag suggestions for thin notes
  (controlled vocabulary), orphan detection. Emits a dated, checkbox review
  queue to `_claude-output/gardener/` — never edits notes directly.
- **Phase 2b — apply step** *(done)*: read back the checked boxes in a review
  file and apply approved tags/links to notes (links append to the body; tags
  fold into frontmatter via the vault's idempotent `migrate_tags`).
- **Phase 3 — auto-MOC detection** *(next)*: cluster the index (HDBSCAN), find
  dense thematic groups with no covering MOC, hand candidates to the `/moc`
  skill.

## Requirements

- Python 3.11+ with `numpy` (already present on this machine; otherwise
  `pip install numpy`). Everything else is the standard library.
- Ollama running with an embedding model pulled:

  ```sh
  ollama pull qwen3-embedding:0.6b
  ```

## Usage

```sh
python3 trellis.py index               # incremental (re)index — only embeds changed notes
python3 trellis.py index --rebuild     # force a full re-embed
python3 trellis.py search "spaced repetition for habits"
python3 trellis.py neighbors "Dichotomy of Control"   # related-note preview
python3 trellis.py status
```

### Gardener (Phase 2)

```sh
python3 trellis.py garden                  # tend up to `garden_limit` notes -> dated review queue
python3 trellis.py garden --dry-run        # print the report, write nothing
python3 trellis.py garden --limit 0        # no cap — drains the whole backlog (~3h for 877 notes)
python3 trellis.py garden --scope z/,MOCs  # restrict to path prefixes
python3 trellis.py garden --force          # re-garden notes even if unchanged
```

Reports land in `<vault>/_claude-output/gardener/YYYY-MM-DD.md` as checkbox lists.
Two ledgers in the DB make repeat runs cheap and quiet: `garden_state` skips
notes unchanged since they were last gardened, and `suggestions` prevents
re-surfacing an idea you've already seen. Notes are processed
most-disconnected-first, so orphans get attention before well-linked notes.
**Nothing is ever written to your notes** — the (future) apply step will read
back the checked boxes.

### Nightly scheduler (launchd)

`run-garden.sh` refreshes the index then gardens; `com.trellis.garden.plist`
runs it nightly at 03:00. Install:

```sh
cp ~/Developer/trellis/com.trellis.garden.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.trellis.garden.plist
launchctl kickstart -k gui/$(id -u)/com.trellis.garden   # optional: run once now
```

Requires the Ollama app to be running (it autostarts at login). Output is logged
to `garden.log`. To stop: `launchctl bootout gui/$(id -u)/com.trellis.garden`.

### Applying a review (Phase 2b)

After checking the boxes you want in a review file (and editing tag lists / links
freely — the apply step reads the file as edited, not the original suggestions):

```sh
python3 trellis.py apply 2026-06-15.md            # bare filename resolves in _claude-output/gardener/
python3 trellis.py apply 2026-06-15.md --dry-run  # preview; writes nothing
```

Links are appended to each source note's body under `Added by Claude on <date>:`;
tags are folded into YAML frontmatter via the vault's `migrate_tags.migrate_content`.
Already-present links/tags are skipped (safe to re-run), and applied items are
marked `status='applied'` in the ledger. Tag application is skipped with a warning
if `migrate_tags.py` can't be loaded (links still apply).

Any non-dry-run `apply` moves the review file to a sibling `applied/` folder —
even if nothing was checked — so running `apply` on a file always retires it and
`gardener/` only shows reviews still pending. History is preserved (nothing is
deleted), and the archived file is the only human-readable record of suggestions
you *declined* (the seen-ledger keeps those from re-appearing). Use `--dry-run`
to apply nothing and leave the file in place.

Optional convenience alias:

```sh
alias trellis='python3 ~/Developer/trellis/trellis.py'
```

## How it works

Each changed note is embedded as `title + tags + body` (frontmatter stripped).
Vectors are stored as float32 blobs in SQLite (`index.db`); search loads them into
a numpy matrix and ranks by cosine similarity. Change detection is by **content
hash**, not mtime — deliberately, because iCloud sync makes mtimes unreliable.
Switching `embed_model` triggers an automatic full rebuild (vectors from different
models aren't comparable).

Configuration lives in `trellis.toml`; CLI flags override it.

## Model notes

- **Embeddings:** `qwen3-embedding:0.6b` — 32K context (no truncation on long
  notes/MOCs), strong MTEB for its size. Bump to `:4b` if recall disappoints;
  this machine has the headroom. `embeddinggemma` is the smaller-footprint
  alternative (2K context).
- **Generation/judgment (Phase 2):** `qwen3.6:35b-a3b` (fast MoE) or
  `gemma4` — both already pulled locally.

## Tests

```sh
python3 tests/test_trellis.py
```

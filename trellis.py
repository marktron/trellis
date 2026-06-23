#!/usr/bin/env python3
"""trellis — a local-LLM gardener for an Obsidian vault.

Phase 1: an incremental embedding index + semantic search over the vault.
Everything runs locally against Ollama; nothing leaves the machine.

Usage (no install needed — system python3 + numpy):
    python3 trellis.py index               # incremental (re)index the vault
    python3 trellis.py search "query"      # semantic search
    python3 trellis.py neighbors "Note"    # notes most related to a given note
    python3 trellis.py status              # index stats

Config precedence: CLI flags > trellis.toml > built-in defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULTS = {
    # Set your vault path in trellis.toml (copy trellis.toml.example) or via the
    # TRELLIS_VAULT environment variable.
    "vault": os.environ.get("TRELLIS_VAULT", ""),
    "embed_model": "qwen3-embedding:0.6b",
    "db_path": os.path.join(HERE, "index.db"),
    "ollama_url": "http://localhost:11434",
    "exclude_dirs": [
        ".obsidian", ".trash", ".git", "node_modules",
        "_workspace", "templates", "Templates", ".smart-env",
    ],
    "batch_size": 16,
    # qwen3-embedding's Ollama runner crashes (EOF) on dense inputs above ~7k
    # chars; 6000 is a safe margin. Tune with --rebuild (hash is of raw content).
    "max_chars": 6000,
    # --- gardener (phase 2) ---
    "gen_model": "qwen3.6:35b-a3b",   # judgment model for link/tag suggestions
    "gen_timeout": 120,                # seconds per generation before skipping a note
    "gen_num_predict": 1024,           # output token cap (prevents greedy runaway)
    "garden_scope": ["z/"],            # path prefixes the gardener tends
    "garden_limit": 30,                # max notes per run (review burden, not speed)
    "link_candidates": 8,              # semantic neighbors considered per note
    "max_link_suggestions": 5,         # cap accepted links per note
    "tag_thin_threshold": 1,           # suggest tags only when a note has <= this many
    "tag_candidate_neighbors": 15,     # neighbors whose tags form the candidate vocab
    # --- auto-MOC clustering (phase 3) ---
    "cluster_scope": ["z/"],          # path prefixes clustered for MOC candidates
    "umap_components": 5,             # UMAP target dimensionality
    "umap_neighbors": 15,            # UMAP n_neighbors
    "umap_min_dist": 0.0,            # UMAP min_dist (0 = tightest clusters)
    "hdbscan_min_cluster_size": 8,   # smallest group worth a MOC
    "cover_sim_threshold": 0.60,     # centroid≥this to an MOC embedding ⇒ covered
    "moc_link_cover_threshold": 0.70,  # ≥this fraction already MOC-linked ⇒ covered
    "cluster_repr_notes": 8,         # representative notes shown per candidate
    "random_state": 42,              # seed UMAP for run-to-run stability
}

# qwen3-embedding works best with a task instruction on the QUERY side only.
QWEN_QUERY_INSTRUCT = (
    "Instruct: Given a note or query, retrieve other notes that are "
    "conceptually related.\nQuery: "
)


def load_config(cli_overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    toml_path = os.path.join(HERE, "trellis.toml")
    if os.path.exists(toml_path):
        try:
            import tomllib
            with open(toml_path, "rb") as fh:
                cfg.update({k: v for k, v in tomllib.load(fh).items() if v is not None})
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not read trellis.toml ({e}); using defaults", file=sys.stderr)
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})
    return cfg


# --------------------------------------------------------------------------- #
# Pure helpers (covered by tests/test_trellis.py)
# --------------------------------------------------------------------------- #
_FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_text, body). Empty frontmatter if none present."""
    m = _FM_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end():]


def extract_tags(frontmatter: str) -> list[str]:
    """Minimal tag extractor — handles `tags: [a, b]` and indented `- a` lists.

    Deliberately dependency-free; tags are a bonus embedding signal, so a miss
    just means slightly less context, never a crash.
    """
    if not frontmatter:
        return []
    lines = frontmatter.splitlines()
    tags: list[str] = []
    for i, line in enumerate(lines):
        m = re.match(r"^tags\s*:\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        inline = m.group(1).strip()
        if inline.startswith("["):  # tags: [a, b, c]
            tags += [t.strip().strip("'\"") for t in inline.strip("[]").split(",")]
        elif inline and inline not in ("|", ">"):  # tags: a (single, unusual)
            tags.append(inline.strip("'\""))
        # following indented "- tag" lines
        for follow in lines[i + 1:]:
            fm = re.match(r"^\s*-\s+(.*)$", follow)
            if fm:
                tags.append(fm.group(1).strip().strip("'\""))
            elif follow.strip() and not follow.startswith(" "):
                break
        break
    return [t for t in (t.strip() for t in tags) if t]


def content_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_embed_text(title: str, tags: list[str], body: str, max_chars: int) -> str:
    parts = [title]
    if tags:
        parts.append("tags: " + ", ".join(tags))
    parts.append(body.strip())
    return ("\n\n".join(p for p in parts if p))[:max_chars]


CONNECTED_HEADER = "### Connected notes added by Trellis"

# Matches trellis's own appended link blocks — the current section header and the
# legacy "Added by Claude on <date>:" form — plus their bullet lists and any blank
# lines in front. Used to keep these out of embeddings/eligibility hashes and to
# consolidate them into one section.
_TRELLIS_BLOCK_RE = re.compile(
    r"\n*^(?:###[ \t]+Connected notes added by Trellis[ \t]*"
    r"|Added by Claude on [^\n]*:)[ \t]*\n"
    r"(?P<links>(?:[ \t]*-[ \t]*\[\[[^\]\n]+\]\][^\n]*\n?)+)",
    re.MULTILINE)


def strip_trellis_blocks(text: str) -> str:
    """Remove trellis's own appended link blocks so they never pollute an
    embedding or trigger re-gardening (both header forms)."""
    return _TRELLIS_BLOCK_RE.sub("", text)


# Detects a legacy "Added by Claude on <date>:" marker line. Used only to REPORT
# prose blocks left untouched by migration — migration itself keys on
# _TRELLIS_BLOCK_RE, which matches only marker + [[wikilink]]-list blocks.
_LEGACY_MARKER_RE = re.compile(r"^Added by Claude on [^\n]*:[ \t]*$", re.MULTILINE)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Cosine top-k. Assumes query_vec and matrix rows are L2-normalized."""
    if matrix.shape[0] == 0:
        return []
    # NumPy's float32 matmul SIMD path emits spurious divide/overflow warnings
    # even on finite, normalized inputs; suppress them. nan_to_num keeps any
    # genuinely bad score sorting last rather than corrupting the ranking.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = matrix @ query_vec
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #
class OllamaError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise OllamaError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {url} ({e.reason}). Is `ollama serve` running?"
        ) from e
    except TimeoutError as e:
        # Socket read timeout (a slow/stuck generation). NOT a URLError subclass,
        # so it must be caught explicitly or it crashes the whole run.
        raise OllamaError(f"Ollama request to {url} timed out after {timeout}s") from e


def embed(texts: list[str], model: str, base_url: str) -> np.ndarray:
    """Return an (n, dim) float32 array of embeddings for `texts`."""
    resp = _post(f"{base_url}/api/embed", {"model": model, "input": texts})
    vecs = resp.get("embeddings")
    if not vecs:
        raise OllamaError(
            f"No embeddings returned for model '{model}'. "
            f"Pull it first:  ollama pull {model}"
        )
    return np.asarray(vecs, dtype=np.float32)


def embed_resilient(text: str, model: str, base_url: str, floor: int = 1000) -> np.ndarray:
    """Embed one text, halving it on runner crashes down to `floor` chars.

    Some notes tokenize densely enough to crash the embedding runner even under
    the char cap; embedding the head of the note beats skipping it entirely.
    """
    t = text
    while True:
        try:
            return embed([t], model, base_url)[0]
        except OllamaError:
            if len(t) <= floor:
                raise
            t = t[: max(floor, len(t) // 2)]


def generate_json(prompt: str, model: str, base_url: str,
                  timeout: float = 120.0, num_predict: int = 1024) -> dict:
    """Call a generation model and parse its JSON response.

    Uses Ollama's JSON-constrained output and disables 'thinking' so qwen3 models
    return JSON directly. `num_predict` caps output tokens so a greedy repetition
    loop can't run away (the legit JSON is < ~300 tokens). Returns {} on a parse
    failure (caller treats as no result).
    """
    resp = _post(
        f"{base_url}/api/generate",
        {"model": model, "prompt": prompt, "stream": False,
         "format": "json", "think": False,
         "options": {"temperature": 0, "num_predict": num_predict}},
        timeout=timeout,
    )
    text = resp.get("response", "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS notes (
               path TEXT PRIMARY KEY,
               title TEXT,
               hash TEXT,
               mtime REAL,
               model TEXT,
               dim INTEGER,
               embedding BLOB,
               indexed_at REAL
           )"""
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    return conn


def meta_get(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=?",
        (key, str(value), str(value)),
    )


# --------------------------------------------------------------------------- #
# Vault walking
# --------------------------------------------------------------------------- #
def iter_markdown(vault: str, exclude: set[str]):
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in exclude and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".md"):
                full = os.path.join(root, fn)
                yield full, os.path.relpath(full, vault)


def read_note(full_path: str) -> bytes | None:
    """Read a note, materializing iCloud 'dataless' files if needed."""
    try:
        with open(full_path, "rb") as fh:
            raw = fh.read()
        if raw == b"" and os.path.getsize(full_path) > 0:
            raise OSError("dataless")
        return raw
    except OSError:
        # Best-effort: ask iCloud to download, then retry once.
        try:
            import subprocess
            subprocess.run(["brctl", "download", full_path], timeout=30,
                           capture_output=True, check=False)
            with open(full_path, "rb") as fh:
                return fh.read()
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _require_vault(cfg) -> bool:
    """True if the configured vault exists; else print a helpful error."""
    if not (cfg.get("vault") and os.path.isdir(cfg["vault"])):
        print("error: vault not set or not found. Copy trellis.toml.example to "
              "trellis.toml and set `vault` (or set the TRELLIS_VAULT env var).",
              file=sys.stderr)
        return False
    return True


def cmd_index(cfg, args):
    vault = cfg["vault"]
    model = cfg["embed_model"]
    exclude = set(cfg["exclude_dirs"])
    if not _require_vault(cfg):
        return 1

    conn = connect(cfg["db_path"])
    stored_model = meta_get(conn, "embed_model")
    model_changed = stored_model is not None and stored_model != model
    rebuild = args.rebuild or model_changed
    if model_changed and not args.rebuild:
        print(f"note: embed model changed ({stored_model} -> {model}); rebuilding index.")
    if rebuild:
        conn.execute("DELETE FROM notes")
        conn.commit()

    existing = {
        row[0]: row[1]
        for row in conn.execute("SELECT path, hash FROM notes").fetchall()
    }
    seen: set[str] = set()
    pending: list[tuple] = []  # (relpath, title, hash, mtime, embed_text)

    files = list(iter_markdown(vault, exclude))
    if args.limit:
        files = files[: args.limit]

    for full, rel in files:
        seen.add(rel)
        raw = read_note(full)
        if raw is None:
            print(f"  skip (unreadable): {rel}", file=sys.stderr)
            continue
        h = content_hash(raw)
        if existing.get(rel) == h:
            continue  # unchanged
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        fm, body = split_frontmatter(text)
        title = os.path.splitext(os.path.basename(rel))[0]
        et = build_embed_text(title, extract_tags(fm), strip_trellis_blocks(body), cfg["max_chars"])
        pending.append((rel, title, h, os.path.getmtime(full), et))

    # Delete rows for files removed from the vault.
    removed = [p for p in existing if p not in seen]
    for p in removed:
        conn.execute("DELETE FROM notes WHERE path=?", (p,))
    if removed:
        print(f"removed {len(removed)} deleted note(s) from index")

    print(f"{len(files)} notes scanned · {len(pending)} new/changed to embed", flush=True)
    if not pending:
        meta_set(conn, "embed_model", model)
        meta_set(conn, "last_indexed", time.time())
        conn.commit()
        print("index up to date.")
        return 0

    bs = cfg["batch_size"]
    now = time.time()
    done = 0
    skipped: list[str] = []

    def store(rec, vec):
        rel, title, h, mtime, _ = rec
        conn.execute(
            """INSERT INTO notes(path,title,hash,mtime,model,dim,embedding,indexed_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                 title=excluded.title, hash=excluded.hash, mtime=excluded.mtime,
                 model=excluded.model, dim=excluded.dim, embedding=excluded.embedding,
                 indexed_at=excluded.indexed_at""",
            (rel, title, h, mtime, model, int(vec.shape[0]),
             vec.astype(np.float32).tobytes(), now),
        )

    for i in range(0, len(pending), bs):
        batch = pending[i:i + bs]
        try:
            vecs = embed([b[4] for b in batch], model, cfg["ollama_url"])
            for rec, vec in zip(batch, vecs):
                store(rec, vec)
        except OllamaError:
            # A note in this batch crashed the runner. Re-embed one at a time so
            # one poison note can't abort the run; skip (and log) any that fail.
            for rec in batch:
                try:
                    store(rec, embed_resilient(rec[4], model, cfg["ollama_url"]))
                except OllamaError as e:
                    skipped.append(rec[0])
                    print(f"\n  skip (embed failed): {rec[0]}\n    {str(e)[:160]}",
                          file=sys.stderr)
        done += len(batch)
        conn.commit()
        print(f"  embedded {done - len(skipped)}/{len(pending)}", end="\r", flush=True)

    meta_set(conn, "embed_model", model)
    meta_set(conn, "last_indexed", now)
    conn.commit()
    print(f"\nindexed {done - len(skipped)} note(s) with {model}.")
    if skipped:
        print(f"skipped {len(skipped)} note(s) the embedder could not handle:")
        for p in skipped:
            print(f"  - {p}")
    return 0


def _load_matrix(conn):
    rows = conn.execute("SELECT path, title, dim, embedding FROM notes").fetchall()
    if not rows:
        return [], [], np.zeros((0, 0), dtype=np.float32)
    paths = [r[0] for r in rows]
    titles = [r[1] for r in rows]
    dim = rows[0][2]
    mat = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float32).reshape(len(rows), dim)
    return paths, titles, l2_normalize(mat.copy())


def _embed_query(text, cfg):
    model = cfg["embed_model"]
    q = (QWEN_QUERY_INSTRUCT + text) if model.startswith("qwen3-embedding") else text
    vec = embed([q], model, cfg["ollama_url"])[0]
    n = np.linalg.norm(vec)
    return vec / (n if n else 1.0)


def cmd_search(cfg, args):
    conn = connect(cfg["db_path"])
    paths, titles, mat = _load_matrix(conn)
    if not paths:
        print("index is empty — run:  python3 trellis.py index", file=sys.stderr)
        return 1
    try:
        qvec = _embed_query(args.query, cfg)
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    results = top_k(qvec, mat, args.k * 4 if args.path else args.k)
    shown = 0
    for idx, score in results:
        if args.path and args.path.lower() not in paths[idx].lower():
            continue
        print(f"{score:5.3f}  {titles[idx]}\n       {paths[idx]}")
        shown += 1
        if shown >= args.k:
            break
    return 0


def resolve_note(query: str, paths: list[str], titles: list[str]) -> str | None:
    """Resolve a user-supplied note reference to a single indexed path.

    Matches on exact title (case-insensitive, trailing `.md` stripped) or path
    substring. On multiple hits an exact title match wins, else the first match.
    Returns the path, or None if nothing matches."""
    q = query.lower().removesuffix(".md")
    matches = [p for p, tt in zip(paths, titles)
               if q == tt.lower() or q in p.lower()]
    if not matches:
        return None
    if len(matches) > 1:
        exact = [p for p, tt in zip(paths, titles) if tt.lower() == q]
        matches = exact or matches
    return matches[0]


def cmd_neighbors(cfg, args):
    conn = connect(cfg["db_path"])
    paths, titles, mat = _load_matrix(conn)
    if not paths:
        print("index is empty — run:  python3 trellis.py index", file=sys.stderr)
        return 1
    target = resolve_note(args.note, paths, titles)
    if target is None:
        print(f"no indexed note matches '{args.note}'", file=sys.stderr)
        return 1
    src = paths.index(target)
    print(f"neighbors of: {titles[src]}  ({paths[src]})\n")
    for idx, score in top_k(mat[src], mat, args.k + 1):
        if idx == src:
            continue
        print(f"{score:5.3f}  {titles[idx]}\n       {paths[idx]}")
    return 0


def cmd_status(cfg, args):
    conn = connect(cfg["db_path"])
    n = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    dim = conn.execute("SELECT dim FROM notes LIMIT 1").fetchone()
    last = meta_get(conn, "last_indexed")
    last_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(last))) if last else "never"
    size = os.path.getsize(cfg["db_path"]) / 1e6 if os.path.exists(cfg["db_path"]) else 0
    print(f"vault        : {cfg['vault']}")
    print(f"embed model  : {meta_get(conn, 'embed_model') or cfg['embed_model']}")
    print(f"indexed notes: {n}")
    print(f"vector dim   : {dim[0] if dim else '-'}")
    print(f"db           : {cfg['db_path']}  ({size:.1f} MB)")
    print(f"last indexed : {last_str}")
    return 0


# --------------------------------------------------------------------------- #
# Gardener (phase 2): link + tag suggestions -> review queue
# --------------------------------------------------------------------------- #
import collections  # noqa: E402
import datetime  # noqa: E402

_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

LINK_PROMPT = """You maintain a Zettelkasten. Given a SOURCE note and CANDIDATE \
notes (found by semantic similarity), choose which candidates are genuinely worth \
linking from the source — only real conceptual connections a reader of the source \
would benefit from following. Be selective; choosing none is fine.

SOURCE: "{title}"
{excerpt}

CANDIDATES:
{candidates}

Return JSON only:
{{"links": [{{"title": "<exact candidate title>", "reason": "<8 words max>"}}]}}"""

TAG_PROMPT = """You maintain a tag vocabulary for a Zettelkasten. Suggest tags for \
the NOTE below, choosing ONLY from the EXISTING TAGS list so the vocabulary stays \
consistent. You may propose at most one genuinely new tag if nothing fits.

NOTE: "{title}"
{excerpt}

EXISTING TAGS (choose from these): {tags}

Return JSON only:
{{"tags": ["<existing tag>", ...], "proposed_new": ["<new tag or omit>"]}}"""


def parse_outlinks(body: str) -> set[str]:
    """Return the set of wikilink targets in a note body, normalized to lowercase
    basenames (alias and #heading stripped)."""
    out = set()
    for m in _LINK_RE.findall(body):
        target = m.split("|")[0].split("#")[0].strip()
        if target:
            out.add(target.lower())
    return out


def build_link_graph(notes: dict) -> tuple[dict, collections.Counter]:
    """notes: rel -> {title, out}. Returns (title_lower -> rel, inbound counts)."""
    title_to_rel: dict[str, str] = {}
    for rel, n in notes.items():
        title_to_rel.setdefault(n["title"].lower(), rel)
    inbound: collections.Counter = collections.Counter()
    for rel, n in notes.items():
        for tgt in n["out"]:
            dest = title_to_rel.get(tgt)
            if dest and dest != rel:
                inbound[dest] += 1
    return title_to_rel, inbound


def resolved_outlinks(note: dict, self_rel: str, title_to_rel: dict) -> set[str]:
    return {tgt for tgt in note["out"]
            if tgt in title_to_rel and title_to_rel[tgt] != self_rel}


def candidate_tags(neighbor_tag_lists: list[list[str]], own_tags: set[str],
                   top_n: int) -> list[str]:
    """Aggregate neighbors' tags into a frequency-ranked candidate vocabulary,
    excluding tags the note already has."""
    counts: collections.Counter = collections.Counter()
    for tags in neighbor_tag_lists:
        counts.update(tags)
    ranked = [tag for tag, _ in counts.most_common() if tag not in own_tags]
    return ranked[:top_n]


def cluster_members(labels):
    """Map each non-noise cluster label to its member row indices (-1 = noise)."""
    out = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab < 0:
            continue
        out.setdefault(lab, []).append(i)
    return out


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


def is_covered(sim_score, link_cov, sim_threshold, link_threshold):
    """A cluster is already covered if an MOC is semantically close OR most of
    its notes are already linked from some MOC (structural coverage)."""
    return sim_score >= sim_threshold or link_cov >= link_threshold


def filter_unseen(candidates, seen_anchors):
    """Drop candidates whose anchor path is already in seen_anchors."""
    return [c for c in candidates if c["anchor"] not in seen_anchors]


def normalize_tag(tag):
    """Normalize a suggested tag to vault convention: lowercase, no leading '#',
    spaces/underscores → hyphens, nested '/' preserved."""
    if not tag:
        return ""
    t = re.sub(r"[\s_]+", "-", tag.strip().lstrip("#").strip().lower())
    return re.sub(r"-{2,}", "-", t).strip("-")


CLUSTER_NAME_PROMPT = """You organize a Zettelkasten into topic maps (MOCs). \
Below is a cluster of related notes found by semantic similarity. Name the single \
coherent theme they share, in a few words suitable as a MOC title.

COMMON TAGS: {tags}

REPRESENTATIVE NOTES:
{titles}

Return JSON only:
{{"theme": "<short title>", "suggested_tag": "<one lowercase tag, nested ok>", "rationale": "<8 words max>"}}"""


def build_cluster_naming_prompt(top_tags, repr_titles):
    tags = ", ".join(top_tags) if top_tags else "(none)"
    titles = "\n".join(f"- {x}" for x in repr_titles)
    return CLUSTER_NAME_PROMPT.format(tags=tags, titles=titles)


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


def _ensure_cluster_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS moc_candidates (
                        anchor_path TEXT PRIMARY KEY, theme TEXT, tag TEXT,
                        member_count INTEGER, nearest_moc TEXT, score REAL,
                        first_seen REAL, status TEXT DEFAULT 'new')""")
    conn.commit()


def classify_tag_suggestions(picked_raw: list, proposed_raw: list,
                             cand_tags: list, vault_tags, own_tags) -> tuple:
    """Split a model's tag response into (existing_picks, genuinely_new).

    The model is shown only this note's neighbors' tags (`cand_tags`), not the
    whole vault, so it routinely flags a tag that exists elsewhere as 'new'.
    Validate `proposed_new` against the full vault vocabulary (case-insensitive),
    mirroring the link path's hallucination guard:
      - already on the note  -> dropped (not a suggestion),
      - exists in the vault   -> reclassified as a normal pick (NOT 'new'),
      - otherwise             -> a genuine new tag (capped to one).
    """
    # Some models emit explicit JSON null (e.g. {"proposed_new": null}); dict.get
    # returns that None rather than the default, so coerce before iterating.
    picked_raw = picked_raw or []
    proposed_raw = proposed_raw or []
    cand_set = set(cand_tags)
    own = {t.lower() for t in own_tags}
    vault = {t.lower() for t in vault_tags}
    picked = [t for t in picked_raw if t in cand_set]
    have = {t.lower() for t in picked} | own
    new_tag: list = []
    for t in proposed_raw:
        if not t:
            continue
        low = t.lower()
        if low in have:        # already on the note or already picked
            continue
        if low in vault:       # real existing tag the model mislabeled as new
            picked.append(t)
            have.add(low)
        else:
            new_tag.append(t)
    return picked, new_tag[:1]


def render_report(date_str: str, summary: dict, link_items: list, tag_items: list,
                  orphans: list) -> str:
    """Render the gardener review queue as markdown. Pure (no I/O) for testing."""
    L = [f"# Gardener review — {date_str}", ""]
    L.append(
        f"_Processed {summary['processed']} note(s) · "
        f"{summary['new_links']} new link suggestion(s) · "
        f"{summary['new_tags']} tag suggestion(s) · "
        f"{summary['orphans']} orphan(s) in scope._")
    L.append("")
    L.append("> Check the boxes you want, then run `trellis apply <this file>`. "
             "Nothing here has been written to your notes.")
    L.append("")
    if link_items:
        L.append("## Link suggestions\n")
        for it in link_items:
            L.append(f"### [[{it['source']}]]")
            for s in it["suggestions"]:
                L.append(f"- [ ] link → [[{s['title']}]] — {s['reason']}")
            L.append("")
    if tag_items:
        L.append("## Tag suggestions\n")
        for it in tag_items:
            tags = " ".join(f"`{x}`" for x in it["tags"])
            L.append(f"- [ ] [[{it['source']}]] → {tags}")
            if it.get("proposed_new"):
                pn = " ".join(f"`{x}`" for x in it["proposed_new"])
                L.append(f"  - [ ] (new tag, use sparingly) {pn}")
        L.append("")
    if orphans:
        L.append(f"## Orphans in scope ({len(orphans)} total — "
                 f"showing up to 40)\n")
        L.append("_No inbound or outbound links. The link suggestions above "
                 "target the most disconnected first._\n")
        for rel in orphans[:40]:
            L.append(f"- [[{os.path.splitext(os.path.basename(rel))[0]}]]")
        L.append("")
    if not (link_items or tag_items):
        L.append("_No new suggestions this run._")
    return "\n".join(L)


def _ensure_garden_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS garden_state (
                        path TEXT PRIMARY KEY, hash TEXT, gardened_at REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS suggestions (
                        path TEXT, kind TEXT, value TEXT, reason TEXT,
                        first_seen REAL, status TEXT DEFAULT 'new',
                        PRIMARY KEY (path, kind, value))""")
    conn.commit()


def _scan_vault(vault, exclude, max_chars):
    """One pass: rel -> {title, tags, out, excerpt, hash}."""
    notes = {}
    for full, rel in iter_markdown(vault, exclude):
        raw = read_note(full)
        if raw is None:
            continue
        text = raw.decode("utf-8", "replace")
        fm, body = split_frontmatter(text)
        notes[rel] = {
            "title": os.path.splitext(os.path.basename(rel))[0],
            "tags": extract_tags(fm),
            "out": parse_outlinks(body),
            "excerpt": " ".join(body.split()),
            # Hash the note WITHOUT trellis's own appended blocks, so applying
            # links doesn't re-trigger gardening on an otherwise-unchanged note.
            "hash": content_hash(strip_trellis_blocks(text).encode("utf-8")),
        }
    return notes


def cmd_garden(cfg, args):
    if not _require_vault(cfg):
        return 1
    vault = cfg["vault"]
    scope = tuple(args.scope.split(",")) if args.scope else tuple(cfg["garden_scope"])
    gen_model = cfg["gen_model"]
    limit = args.limit if args.limit is not None else cfg["garden_limit"]

    conn = connect(cfg["db_path"])
    _ensure_garden_tables(conn)
    paths, titles, mat = _load_matrix(conn)
    if not paths:
        print("index is empty — run:  python3 trellis.py index", file=sys.stderr)
        return 1
    rel_to_idx = {p: i for i, p in enumerate(paths)}

    print("scanning vault…", flush=True)
    notes = _scan_vault(vault, set(cfg["exclude_dirs"]), cfg["max_chars"])
    title_to_rel, inbound = build_link_graph(notes)
    # Full vault tag vocabulary — lets us reject "new tag" suggestions for tags
    # that already exist outside this note's local neighbor candidate set.
    vault_tags = {tag.lower() for n in notes.values() for tag in n["tags"]}

    if getattr(args, "note", None):
        # Single-note mode: garden exactly one note, ignoring scope/limit and
        # the unchanged-since-last-run ledger (targeting it implies --force).
        target = resolve_note(args.note, paths, titles)
        if target is None or target not in notes:
            print(f"no indexed note matches '{args.note}'", file=sys.stderr)
            return 1
        in_scope = eligible = [target]
        orphans = [target] if (not resolved_outlinks(notes[target], target, title_to_rel)
                               and inbound[target] == 0) else []
        print(f"single note · {notes[target]['title']}  ({target})\n", flush=True)
    else:
        in_scope = [r for r in notes if r.startswith(scope) and r in rel_to_idx]
        orphans = [r for r in in_scope
                   if not resolved_outlinks(notes[r], r, title_to_rel) and inbound[r] == 0]

        gardened = {row[0]: row[1] for row in
                    conn.execute("SELECT path, hash FROM garden_state").fetchall()}
        eligible = [r for r in in_scope
                    if args.force or gardened.get(r) != notes[r]["hash"]]
        # Most-disconnected first: orphans, then by (inbound+outbound) ascending.
        eligible.sort(key=lambda r: (inbound[r] + len(resolved_outlinks(notes[r], r, title_to_rel))))
        if limit:
            eligible = eligible[:limit]

        print(f"scope {scope} · {len(in_scope)} notes · {len(orphans)} orphans · "
              f"{len(eligible)} to process this run\n", flush=True)

    seen = {(row[0], row[1], row[2]) for row in
            conn.execute("SELECT path, kind, value FROM suggestions").fetchall()}
    now = time.time()
    link_items, tag_items = [], []
    new_links = new_tags = 0

    for n, rel in enumerate(eligible, 1):
        note = notes[rel]
        idx = rel_to_idx[rel]
        already_linked = note["out"]

        # ---- link suggestions ----
        neigh = [(titles[i], paths[i]) for i, _ in
                 top_k(mat[idx], mat, cfg["link_candidates"] + 1) if i != idx]
        cand = [(tt, pp) for tt, pp in neigh[:cfg["link_candidates"]]
                if tt.lower() not in already_linked]
        suggestions = []
        if cand:
            block = "\n".join(
                f'{i+1}. "{tt}" — {notes.get(pp, {}).get("excerpt", "")[:300]}'
                for i, (tt, pp) in enumerate(cand))
            prompt = LINK_PROMPT.format(
                title=note["title"], excerpt=note["excerpt"][:1200], candidates=block)
            try:
                out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                    timeout=cfg["gen_timeout"],
                                    num_predict=cfg["gen_num_predict"])
            except OllamaError as e:
                print(f"  link gen failed for {rel}: {str(e)[:120]}", file=sys.stderr)
                out = {}
            valid_titles = {tt.lower() for tt, _ in cand}
            for s in (out.get("links") or [])[:cfg["max_link_suggestions"]]:
                title = str(s.get("title", "")).strip()
                if title.lower() not in valid_titles:
                    continue  # model hallucinated a title not in candidates
                key = (rel, "link", title)
                if key in seen:
                    continue
                seen.add(key)
                reason = str(s.get("reason", "")).strip()[:120]
                suggestions.append({"title": title, "reason": reason})
                if not args.dry_run:
                    conn.execute("INSERT OR IGNORE INTO suggestions VALUES(?,?,?,?,?,?)",
                                 (rel, "link", title, reason, now, "new"))
        if suggestions:
            link_items.append({"source": note["title"], "suggestions": suggestions})
            new_links += len(suggestions)

        # ---- tag suggestions (thin notes only) ----
        if len(note["tags"]) <= cfg["tag_thin_threshold"]:
            ntags = [notes.get(pp, {}).get("tags", [])
                     for _, pp in neigh[:cfg["tag_candidate_neighbors"]]]
            cand_tags = candidate_tags(ntags, set(note["tags"]), 20)
            if cand_tags:
                prompt = TAG_PROMPT.format(
                    title=note["title"], excerpt=note["excerpt"][:1200],
                    tags=", ".join(cand_tags))
                try:
                    out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                    timeout=cfg["gen_timeout"],
                                    num_predict=cfg["gen_num_predict"])
                except OllamaError:
                    out = {}
                picked, new_tag = classify_tag_suggestions(
                    out.get("tags", []), out.get("proposed_new", []),
                    cand_tags, vault_tags, note["tags"])
                fresh = [t for t in picked if (rel, "tag", t) not in seen]
                if fresh or new_tag:
                    for t in fresh:
                        seen.add((rel, "tag", t))
                        if not args.dry_run:
                            conn.execute("INSERT OR IGNORE INTO suggestions VALUES(?,?,?,?,?,?)",
                                         (rel, "tag", t, "", now, "new"))
                    tag_items.append({"source": note["title"], "tags": fresh,
                                      "proposed_new": new_tag})
                    new_tags += len(fresh) + len(new_tag)

        if not args.dry_run:
            conn.execute(
                "INSERT INTO garden_state VALUES(?,?,?) ON CONFLICT(path) "
                "DO UPDATE SET hash=excluded.hash, gardened_at=excluded.gardened_at",
                (rel, note["hash"], now))
        conn.commit()
        print(f"  [{n}/{len(eligible)}] {note['title'][:50]}"
              f"  (+{len(suggestions)} links)", flush=True)

    date_str = datetime.date.today().isoformat()
    summary = {"processed": len(eligible), "new_links": new_links,
               "new_tags": new_tags, "orphans": len(orphans)}
    report = render_report(date_str, summary, link_items, tag_items, orphans)

    if args.dry_run:
        print("\n--- DRY RUN (report not written) ---\n")
        print(report)
        return 0
    out_dir = os.path.join(vault, "_workspace", "gardener")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.md")
    if os.path.exists(out_path):  # don't clobber an earlier run on the same day
        out_path = os.path.join(out_dir, f"{date_str}-{time.strftime('%H%M')}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nreview queue → {out_path}")
    print(f"  {new_links} new link · {new_tags} tag suggestion(s) "
          f"across {len(eligible)} note(s)")
    return 0


# --------------------------------------------------------------------------- #
# Cluster (phase 3): detect MOC-candidate clusters -> review report
# --------------------------------------------------------------------------- #
def cmd_cluster(cfg, args):
    if not _require_vault(cfg):
        return 1
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
        member_paths = [paths[i] for i in members]
        link_cov = link_coverage(member_paths, moc_linked)
        if is_covered(score, link_cov, cfg["cover_sim_threshold"],
                      cfg["moc_link_cover_threshold"]):
            covered += 1
            continue                                          # an MOC already covers it
        nearest = (titles[moc_idx[j]], score) if j >= 0 else None

        member_titles = [titles[i] for i in ranked]
        repr_titles = member_titles[:cfg["cluster_repr_notes"]]
        top_tags = candidate_tags([notes.get(p, {}).get("tags", []) for p in member_paths],
                                  set(), 6)

        candidates.append({
            "anchor": anchor_rel, "theme": "", "tag": "", "rationale": "",
            "member_count": len(members), "link_coverage": link_cov,
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
        c["tag"] = normalize_tag(str(out.get("suggested_tag") or (c["top_tags"][0] if c["top_tags"] else "")))
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

    out_dir = os.path.join(vault, "_workspace", "clusters")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.md")
    if os.path.exists(out_path):
        out_path = os.path.join(out_dir, f"{date_str}-{time.strftime('%H%M')}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nMOC candidates → {out_path}")
    print(f"  {len(candidates)} new candidate(s) · {covered} covered · {len(groups)} cluster(s)")
    return 0


# --------------------------------------------------------------------------- #
# Apply (phase 2b): write checked review items back into notes
# --------------------------------------------------------------------------- #
_CHECK_LINK_RE = re.compile(r"^-\s+\[[xX]\]\s+link\s*→\s*\[\[(.+?)\]\]")
_CHECK_TAG_RE = re.compile(r"^-\s+\[[xX]\]\s+\[\[(.+?)\]\]\s*→\s*(.*)$")
_SRC_HDR_RE = re.compile(r"^###\s+\[\[(.+?)\]\]")


def parse_review(md: str) -> dict:
    """Extract CHECKED items from a gardener review file.

    Returns {"links": [(source, target)], "tags": [(source, [tags])]}.
    Unchecked ([ ]) items are ignored; the user's edits to the file win.
    """
    links: list[tuple[str, str]] = []
    tags: list[tuple[str, list[str]]] = []
    section = None
    src = None
    for line in md.splitlines():
        if line.startswith("## "):
            low = line.lower()
            section = ("link" if "link suggestion" in low
                       else "tag" if "tag suggestion" in low else None)
            src = None
            continue
        if section == "link":
            h = _SRC_HDR_RE.match(line)
            if h:
                src = h.group(1).strip()
                continue
            m = _CHECK_LINK_RE.match(line)
            if m and src:
                links.append((src, m.group(1).strip()))
        elif section == "tag":
            m = _CHECK_TAG_RE.match(line)
            if m:
                found = [x.strip() for x in re.findall(r"`([^`]+)`", m.group(2))]
                if found:
                    tags.append((m.group(1).strip(), found))
    return {"links": links, "tags": tags}


def _load_migrate_content(cfg):
    """Dynamically load the vault's idempotent tag-migration helper."""
    import importlib.util
    p = os.path.join(cfg["vault"], "_workspace", "scripts", "migrate_tags.py")
    if not os.path.exists(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("migrate_tags", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.migrate_content
    except Exception:  # noqa: BLE001
        return None


def _archive_review(path: str) -> str | None:
    """Move a processed review file into an `applied/` sibling dir so the
    gardener folder only shows pending reviews. Never clobbers; non-fatal."""
    archive_dir = os.path.join(os.path.dirname(path), "applied")
    try:
        os.makedirs(archive_dir, exist_ok=True)
        dest = os.path.join(archive_dir, os.path.basename(path))
        if os.path.exists(dest):
            stem, ext = os.path.splitext(os.path.basename(path))
            dest = os.path.join(archive_dir, f"{stem}-{time.strftime('%H%M%S')}{ext}")
        os.replace(path, dest)
        return dest
    except OSError as e:
        print(f"warning: could not archive review file ({e})", file=sys.stderr)
        return None


def consolidate_connected(content: str, new_targets) -> str:
    """Fold new link targets into a single Connected-notes section, absorbing any
    existing section and legacy 'Added by Claude' blocks (deduped, order-preserving)."""
    existing = []
    for m in _TRELLIS_BLOCK_RE.finditer(content):
        for raw in _LINK_RE.findall(m.group("links")):
            tgt = raw.split("|")[0].split("#")[0].strip()
            if tgt:
                existing.append(tgt)
    base = _TRELLIS_BLOCK_RE.sub("", content).rstrip()
    seen, ordered = set(), []
    for tgt in [*existing, *new_targets]:
        k = tgt.lower()
        if k and k not in seen:
            seen.add(k)
            ordered.append(tgt)
    if not ordered:
        return (base + "\n") if base else ""
    block = "\n".join(f"- [[{x}]]" for x in ordered)
    prefix = (base + "\n\n") if base else ""
    return f"{prefix}{CONNECTED_HEADER}\n{block}\n"


def cmd_apply(cfg, args):
    if not _require_vault(cfg):
        return 1
    path = args.file
    if not os.path.exists(path):
        alt = os.path.join(cfg["vault"], "_workspace", "gardener", path)
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        print(f"error: review file not found: {args.file}", file=sys.stderr)
        return 1

    review = parse_review(open(path, encoding="utf-8").read())
    if not review["links"] and not review["tags"]:
        print(f"no checked items in {path} — nothing to apply")
        if not args.dry_run:  # explicit apply = retire the review anyway
            archived = _archive_review(path)
            if archived:
                print(f"archived review → {archived}")
        return 0

    notes = _scan_vault(cfg["vault"], set(cfg["exclude_dirs"]), cfg["max_chars"])
    title_to_rel: dict[str, str] = {}
    for rel, n in notes.items():
        title_to_rel.setdefault(n["title"].lower(), rel)

    add_links: dict[str, list] = collections.defaultdict(list)
    add_tags: dict[str, list] = collections.defaultdict(list)
    for s, tgt in review["links"]:
        add_links[s].append(tgt)
    for s, tg in review["tags"]:
        add_tags[s].extend(tg)
    sources = sorted(set(add_links) | set(add_tags))

    migrate = _load_migrate_content(cfg) if any(add_tags.values()) else None
    if add_tags and migrate is None:
        print("warning: migrate_tags.py not loadable — tags will be SKIPPED "
              "(links still applied)", file=sys.stderr)

    conn = connect(cfg["db_path"])
    _ensure_garden_tables(conn)
    applied_links = applied_tags = 0

    for s in sources:
        rel = title_to_rel.get(s.lower())
        if not rel:
            print(f"  ! source not found, skipping: {s}", file=sys.stderr)
            continue
        note = notes[rel]
        new_links = [t for t in dict.fromkeys(add_links.get(s, []))
                     if t.lower() not in note["out"]]
        new_tags = [t for t in dict.fromkeys(add_tags.get(s, []))
                    if migrate and t not in note["tags"]]
        if not new_links and not new_tags:
            print(f"  = up to date: {s}")
            continue

        if args.dry_run:
            print(f"  {rel}")
            for t in new_tags:
                print(f"      + tag   {t}")
            for t in new_links:
                print(f"      + link  [[{t}]]")
        else:
            full = os.path.join(cfg["vault"], rel)
            content = open(full, encoding="utf-8").read()
            if new_tags:  # fold into frontmatter via the vault's migration trick
                staged = content if content.endswith("\n") else content + "\n"
                staged += "\n" + "\n".join(f"#{t}" for t in new_tags) + "\n"
                migrated = migrate(staged)[0]
                if migrated:
                    content = migrated
                else:
                    print(f"  ! tag merge no-op for {s}; leaving tags", file=sys.stderr)
                    new_tags = []
            if new_links:  # fold into the single Connected-notes section
                content = consolidate_connected(content, new_links)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)
            for t in new_links:
                conn.execute("UPDATE suggestions SET status='applied' "
                             "WHERE path=? AND kind='link' AND value=?", (rel, t))
            for t in new_tags:
                conn.execute("UPDATE suggestions SET status='applied' "
                             "WHERE path=? AND kind='tag' AND value=?", (rel, t))
            print(f"  ✓ {s}  (+{len(new_tags)} tags, +{len(new_links)} links)")
        applied_links += len(new_links)
        applied_tags += len(new_tags)

    if not args.dry_run:
        conn.commit()
    head = "DRY RUN — would apply" if args.dry_run else "applied"
    print(f"\n{head}: {applied_links} link(s) · {applied_tags} tag(s) "
          f"across {len(sources)} source note(s)")
    if not args.dry_run:
        archived = _archive_review(path)
        if archived:
            print(f"archived review → {archived}")
    return 0


# --------------------------------------------------------------------------- #
# Migrate: legacy "Added by Claude on <date>:" link blocks -> one section
# --------------------------------------------------------------------------- #
def cmd_migrate(cfg, args):
    if not _require_vault(cfg):
        return 1
    vault = cfg["vault"]
    scope = tuple(args.scope.split(",")) if args.scope else None
    exclude = set(cfg["exclude_dirs"])
    converted, prose_left = [], []
    for full, rel in iter_markdown(vault, exclude):
        if scope and not rel.startswith(scope):
            continue
        raw = read_note(full)
        if raw is None:
            continue
        text = raw.decode("utf-8", "replace")
        # Block-precise: convert only marker + [[link]]-list blocks. Prose blocks
        # under an "Added by Claude" marker don't match _TRELLIS_BLOCK_RE and are
        # left exactly as-is, even when they sit in the same note as a link block.
        after = consolidate_connected(text, []) if _TRELLIS_BLOCK_RE.search(text) else text
        if after != text:
            if args.apply:
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(after)
            converted.append(rel)
        if _LEGACY_MARKER_RE.search(after):
            prose_left.append(rel)                    # a manual prose marker remains

    head = "migrated" if args.apply else "DRY RUN — would migrate"
    print(f"{head}: {len(converted)} note(s) -> single '{CONNECTED_HEADER}' section")
    for rel in converted:
        print(f"  ✓ {rel}")
    if prose_left:
        print(f"\nleft {len(prose_left)} note(s) with a manual 'Added by Claude' prose "
              f"block untouched (some may also appear above — their link block "
              f"was still converted):")
        for rel in prose_left:
            print(f"  · {rel}")
    if not args.apply:
        print("\nnothing written. Re-run with --apply to write these changes.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(prog="trellis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vault", help="path to the Obsidian vault")
    p.add_argument("--embed-model", dest="embed_model", help="Ollama embedding model")
    p.add_argument("--db", dest="db_path", help="path to the index database")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="incrementally (re)index the vault")
    pi.add_argument("--rebuild", action="store_true", help="force a full re-embed")
    pi.add_argument("--limit", type=int, help="only process first N notes (testing)")

    ps = sub.add_parser("search", help="semantic search")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=10, help="number of results")
    ps.add_argument("--path", help="only results whose path contains this substring")

    pn = sub.add_parser("neighbors", help="notes most related to a given note")
    pn.add_argument("note", help="note title or path substring")
    pn.add_argument("-k", type=int, default=10)

    sub.add_parser("status", help="show index stats")

    pg = sub.add_parser("garden", help="suggest links/tags -> dated review queue")
    pg.add_argument("--note", help="garden a single note (title or path substring); "
                                   "ignores --scope/--limit and implies --force")
    pg.add_argument("--limit", type=int, help="max notes this run (0 = no cap)")
    pg.add_argument("--scope", help="comma-separated path prefixes (default: z/)")
    pg.add_argument("--gen-model", dest="gen_model", help="judgment model")
    pg.add_argument("--force", action="store_true",
                    help="re-garden notes even if unchanged since last run")
    pg.add_argument("--dry-run", action="store_true",
                    help="print the report; write nothing (no ledger, no file)")

    pa = sub.add_parser("apply", help="write checked items from a review file into notes")
    pa.add_argument("file", help="path to a gardener review .md (or just its filename)")
    pa.add_argument("--dry-run", action="store_true",
                    help="show what would change; write nothing")

    pcl = sub.add_parser("cluster", help="detect MOC-candidate clusters -> review report")
    pcl.add_argument("--scope", help="comma-separated path prefixes (default: z/)")
    pcl.add_argument("--limit", type=int, help="max candidates to name/report (0 = no cap)")
    pcl.add_argument("--gen-model", dest="gen_model", help="judgment model for naming")
    pcl.add_argument("--force", action="store_true", help="ignore the seen-ledger")
    pcl.add_argument("--dry-run", action="store_true",
                     help="print report; write nothing (no ledger, no file)")

    pm = sub.add_parser(
        "migrate",
        help="convert legacy 'Added by Claude on <date>:' link blocks to one Connected-notes section")
    pm.add_argument("--scope", help="comma-separated path prefixes (default: whole vault)")
    pm.add_argument("--apply", action="store_true", help="write changes (default: dry run)")

    args = p.parse_args(argv)
    cfg = load_config({"vault": args.vault, "embed_model": args.embed_model,
                       "db_path": args.db_path,
                       "gen_model": getattr(args, "gen_model", None)})
    return {
        "index": cmd_index, "search": cmd_search,
        "neighbors": cmd_neighbors, "status": cmd_status, "garden": cmd_garden,
        "apply": cmd_apply, "cluster": cmd_cluster, "migrate": cmd_migrate,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())

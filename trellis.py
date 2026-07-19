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
import fnmatch
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

__version__ = "0.1.0"

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULTS = {
    # Set your vault path in trellis.toml (copy trellis.toml.example) or via the
    # TRELLIS_VAULT environment variable.
    "vault": os.environ.get("TRELLIS_VAULT", ""),
    "embed_model": "qwen3-embedding:0.6b",
    # db_path has no static default here — it depends on where trellis.toml (if
    # any) was found; see _config_path() and load_config().
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
    "gardener_dir": "_workspace/gardener",  # vault-relative review-queue dir; keep it
                                       # under a folder in exclude_dirs or trellis
                                       # will index its own review files
    "garden_limit": 30,                # max notes per run (review burden, not speed)
    "link_candidates": 8,              # semantic neighbors considered per note
    "max_link_suggestions": 5,         # cap accepted links per note
    # cold-start: first-pass or sparsely-linked notes get a wider net (see
    # wants_broad_links). Set broad values equal to the normal ones to disable.
    "link_thin_threshold": 3,          # <= this many existing links ⇒ broad treatment
    "link_candidates_broad": 15,       # neighbors considered for a cold-start note
    "max_link_suggestions_broad": 10,  # cap accepted links for a cold-start note
    # Link-target hygiene: candidates are drawn from the whole index, so
    # operational/scaffold notes leak in and get rejected. link_target_scope (if
    # set) restricts targets to these path prefixes; link_target_exclude drops
    # candidates whose basename matches any glob (case-insensitive).
    "link_target_scope": [],           # [] = any prefix; e.g. ["z/", "MOCs"]
    "link_target_exclude": [
        "_*",                          # scaffold: _decisions, _snapshot, _parked, …
        "readme", "capture", "untitled*",
        "timeline", "design notes", "dev notes", "idea triage", "feature ideas",
        "[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]*",  # dated daily/review notes
    ],
    "tag_thin_threshold": 1,           # suggest tags only when a note has <= this many
    "tag_candidate_neighbors": 15,     # neighbors whose tags form the candidate vocab
    # --- auto-MOC clustering (phase 3) ---
    "cluster_scope": ["z/"],          # path prefixes clustered for MOC candidates
    "moc_scope": ["MOCs/"],           # path prefixes holding your Maps of Content
                                      # (the coverage reference for candidates)
    "clusters_dir": "_workspace/clusters",  # vault-relative candidate-report dir;
                                      # same exclude_dirs caveat as gardener_dir
    "umap_components": 5,             # UMAP target dimensionality
    "umap_neighbors": 15,            # UMAP n_neighbors
    "umap_min_dist": 0.0,            # UMAP min_dist (0 = tightest clusters)
    "hdbscan_min_cluster_size": 8,   # smallest group worth a MOC
    "cover_sim_threshold": 0.60,     # centroid≥this to an MOC embedding ⇒ covered
    "moc_link_cover_threshold": 0.70,  # ≥this fraction already MOC-linked ⇒ covered
    "cluster_repr_notes": 8,         # representative notes shown per candidate
    "random_state": 42,              # seed UMAP for run-to-run stability
    # --- triage (phase 4) ---
    "triage_scope": ["z/"],            # path prefixes triage watches for new notes
    "idea_scope": ["Areas/Product Ideas/"],  # where product-idea files live
    "triage_bulk_min": 8,              # mtime-minute bucket >= this ⇒ suspected bulk touch
    "triage_tag_skip_threshold": 3,    # skip tag step when a note already has >= this many
    "moc_place_threshold": 0.55,       # note↔MOC cosine gate (provisional; tune)
    "idea_link_threshold": 0.55,       # note↔idea cosine gate (provisional; tune)
}

# qwen3-embedding works best with a task instruction on the QUERY side only.
QWEN_QUERY_INSTRUCT = (
    "Instruct: Given a note or query, retrieve other notes that are "
    "conceptually related.\nQuery: "
)


def _config_path() -> str | None:
    """Locate trellis.toml, first hit wins:

    1. $TRELLIS_CONFIG (explicit path; if set but missing, warn and keep looking)
    2. ./trellis.toml (current working directory)
    3. <HERE>/trellis.toml (repo-checkout behavior, preserved for existing users)
    4. $XDG_CONFIG_HOME/trellis/trellis.toml, falling back to ~/.config/trellis/trellis.toml

    Returns None if none of the above exist.
    """
    env_path = os.environ.get("TRELLIS_CONFIG")
    if env_path:
        if os.path.exists(env_path):
            return env_path
        print(f"warning: TRELLIS_CONFIG={env_path} does not exist; "
              "continuing config search", file=sys.stderr)

    cwd_path = os.path.join(os.getcwd(), "trellis.toml")
    if os.path.exists(cwd_path):
        return cwd_path

    here_path = os.path.join(HERE, "trellis.toml")
    if os.path.exists(here_path):
        return here_path

    xdg_base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    xdg_path = os.path.join(xdg_base, "trellis", "trellis.toml")
    if os.path.exists(xdg_path):
        return xdg_path

    return None


def load_config(cli_overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    toml_path = _config_path()
    # Default db_path lives next to whatever config file was found (so an
    # installed user's db sits next to their config); with no config file
    # found anywhere, fall back to the repo-checkout behavior (HERE/index.db).
    # A db_path key from the TOML itself, or --db on the CLI, still wins below.
    cfg["db_path"] = (os.path.join(os.path.dirname(toml_path), "index.db")
                       if toml_path else os.path.join(HERE, "index.db"))
    if toml_path:
        try:
            import tomllib
            with open(toml_path, "rb") as fh:
                cfg.update({k: v for k, v in tomllib.load(fh).items() if v is not None})
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not read trellis.toml ({toml_path}): {e}; "
                  "using defaults", file=sys.stderr)
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
            if created >= cut_date:  # inclusive: same-day notes stay visible; triaged-set dedupes
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

Return JSON only. The reason must be 8 words max and name the specific shared idea \
— avoid vague fillers like "related", "similar", or "provides context":
{{"links": [{{"title": "<exact candidate title>", "reason": "<8 words max>"}}]}}"""

LINK_PROMPT_BROAD = """You maintain a Zettelkasten. The SOURCE note below is new or \
barely linked, so it's starting from a blank slate — favor recall over precision. \
Given the CANDIDATE notes (found by semantic similarity), suggest every candidate \
with a genuine conceptual connection a reader of the source would benefit from \
following. Cast a wide net; only drop candidates that are clearly off-topic.

SOURCE: "{title}"
{excerpt}

CANDIDATES:
{candidates}

Return JSON only. The reason must be 8 words max and name the specific shared idea \
— avoid vague fillers like "related", "similar", or "provides context":
{{"links": [{{"title": "<exact candidate title>", "reason": "<8 words max>"}}]}}"""


def wants_broad_links(rel: str, gardened: dict, link_count: int, threshold: int) -> bool:
    """Whether a note should get the broader (cold-start) link treatment: true on
    its first pass (no garden_state entry) or while it stays sparsely linked
    (<= threshold resolved outlinks)."""
    return rel not in gardened or link_count <= threshold


def eligible_link_target(path: str, scope, exclude_globs) -> bool:
    """Whether a candidate note may be SUGGESTED as a link target. Filters out
    operational/scaffold notes (underscore files, READMEs, dated notes, etc.)
    that reviewers consistently reject, and — when scope is non-empty — anything
    outside it. Globs match the basename (sans .md), case-insensitively."""
    if scope and not path.startswith(tuple(scope)):
        return False
    stem = os.path.basename(path)
    if stem.lower().endswith(".md"):
        stem = stem[:-3]
    s = stem.lower()
    return not any(fnmatch.fnmatchcase(s, g.lower()) for g in exclude_globs)


TAG_PROMPT = """You maintain a tag vocabulary for a Zettelkasten. Suggest tags for \
the NOTE below, choosing ONLY from the EXISTING TAGS list so the vocabulary stays \
consistent. Do not invent new tags; if nothing fits, return an empty list.

NOTE: "{title}"
{excerpt}

EXISTING TAGS (choose from these): {tags}

Return JSON only:
{{"tags": ["<existing tag>", ...]}}"""


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


_MOC_HEADING_RE = re.compile(r"(?m)^(#{2,3})\s+(.+?)\s*$")


def moc_headings(body: str) -> list[str]:
    """Text of the ##/### headings in a MOC body — the placement targets the
    gen model chooses among."""
    return [m.group(2) for m in _MOC_HEADING_RE.finditer(body)]


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


def moc_linked_targets(notes, title_to_rel, moc_prefixes):
    """Set of rel paths that are wikilink targets from any note whose path
    starts with one of `moc_prefixes` (the configured `moc_scope`)."""
    linked = set()
    for rel, n in notes.items():
        if not rel.startswith(tuple(moc_prefixes)):
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
    L = [f"# Review — {date_str}", ""]
    L.append(
        f"_Garden: {summary['processed']} note(s) processed · "
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


def _ensure_garden_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS garden_state (
                        path TEXT PRIMARY KEY, hash TEXT, gardened_at REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS suggestions (
                        path TEXT, kind TEXT, value TEXT, reason TEXT,
                        first_seen REAL, status TEXT DEFAULT 'new',
                        PRIMARY KEY (path, kind, value))""")
    conn.commit()


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
            "created": extract_created(fm),
            "mtime": os.path.getmtime(full),
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

    gardened = {row[0]: row[1] for row in
                conn.execute("SELECT path, hash FROM garden_state").fetchall()}

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
    target_scope = tuple(cfg["link_target_scope"])
    target_exclude = cfg["link_target_exclude"]

    for n, rel in enumerate(eligible, 1):
        note = notes[rel]
        idx = rel_to_idx[rel]
        already_linked = note["out"]

        # ---- link suggestions ----
        # Cold-start notes (first pass or sparsely linked) cast a wider net.
        broad = wants_broad_links(rel, gardened,
                                  len(resolved_outlinks(note, rel, title_to_rel)),
                                  cfg["link_thin_threshold"])
        n_cand = cfg["link_candidates_broad"] if broad else cfg["link_candidates"]
        max_sug = cfg["max_link_suggestions_broad"] if broad else cfg["max_link_suggestions"]
        link_prompt = LINK_PROMPT_BROAD if broad else LINK_PROMPT
        # Over-fetch neighbors so target-hygiene filtering still yields n_cand.
        pool = min(len(paths), n_cand * 4 + 1)
        neigh = [(titles[i], paths[i]) for i, _ in
                 top_k(mat[idx], mat, pool) if i != idx]
        cand = [(tt, pp) for tt, pp in neigh
                if tt.lower() not in already_linked
                and eligible_link_target(pp, target_scope, target_exclude)][:n_cand]
        suggestions = []
        if cand:
            block = "\n".join(
                f'{i+1}. "{tt}" — {notes.get(pp, {}).get("excerpt", "")[:300]}'
                for i, (tt, pp) in enumerate(cand))
            prompt = link_prompt.format(
                title=note["title"], excerpt=note["excerpt"][:1200], candidates=block)
            try:
                out = generate_json(prompt, gen_model, cfg["ollama_url"],
                                    timeout=cfg["gen_timeout"],
                                    num_predict=cfg["gen_num_predict"])
            except OllamaError as e:
                print(f"  link gen failed for {rel}: {str(e)[:120]}", file=sys.stderr)
                out = {}
            valid_titles = {tt.lower() for tt, _ in cand}
            for s in (out.get("links") or [])[:max_sug]:
                title = str(s.get("title", "")).strip()
                if title.lower() not in valid_titles:
                    continue  # model hallucinated a title not in candidates
                key = (rel, "link", title)
                if key in seen:
                    continue
                seen.add(key)
                reason = " ".join(str(s.get("reason", "")).split()[:8])[:120]
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
                # New-tag proposals were almost never accepted in review, so we no
                # longer ask for them — restrict to the existing vocabulary.
                picked, _ = classify_tag_suggestions(
                    out.get("tags", []), [],
                    cand_tags, vault_tags, note["tags"])
                fresh = [t for t in picked if (rel, "tag", t) not in seen]
                if fresh:
                    for t in fresh:
                        seen.add((rel, "tag", t))
                        if not args.dry_run:
                            conn.execute("INSERT OR IGNORE INTO suggestions VALUES(?,?,?,?,?,?)",
                                         (rel, "tag", t, "", now, "new"))
                    tag_items.append({"source": note["title"], "tags": fresh})
                    new_tags += len(fresh)

        if not args.dry_run:
            conn.execute(
                "INSERT INTO garden_state VALUES(?,?,?) ON CONFLICT(path) "
                "DO UPDATE SET hash=excluded.hash, gardened_at=excluded.gardened_at",
                (rel, note["hash"], now))
        conn.commit()
        print(f"  [{n}/{len(eligible)}] {note['title'][:50]}"
              f"  (+{len(suggestions)} links){'  (broad)' if broad else ''}", flush=True)

    date_str = datetime.date.today().isoformat()
    summary = {"processed": len(eligible), "new_links": new_links,
               "new_tags": new_tags, "orphans": len(orphans)}
    report = render_report(date_str, summary, link_items, tag_items, orphans)

    if args.dry_run:
        print("\n--- DRY RUN (report not written) ---\n")
        print(report)
        return 0
    out_path = append_or_create_review(
        os.path.join(vault, cfg["gardener_dir"]), date_str, report)
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
    moc_prefixes = tuple(cfg["moc_scope"])
    moc_idx = [i for i, p in enumerate(paths) if p.startswith(moc_prefixes)]
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
    moc_linked = moc_linked_targets(notes, title_to_rel, moc_prefixes)

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

    out_path = _dated_report_path(os.path.join(vault, cfg["clusters_dir"]), date_str)
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
_CHECK_MOC_RE = re.compile(
    r"^-\s+\[[xX]\]\s+\[\[(.+?)\]\]\s*→\s*\[\[(.+?)\]\]\s*§\s*(.+?)\s*—")
_CHECK_IDEA_RE = re.compile(
    r"^-\s+\[[xX]\]\s+\[\[(.+?)\]\]\s*→\s*\[\[(.+?)\]\]\s*—\s*(.*)$")


def parse_review(md: str) -> dict:
    """Extract CHECKED items from a gardener review file.

    Returns {"links": [(source, target)], "tags": [(source, [tags])],
             "mocs": [(note, moc, section)], "ideas": [(note, idea, reason)]}.
    Unchecked ([ ]) items are ignored; the user's edits to the file win.
    """
    links: list[tuple[str, str]] = []
    tags: list[tuple[str, list[str]]] = []
    mocs: list[tuple[str, str, str]] = []
    ideas: list[tuple[str, str, str]] = []
    section = None
    src = None
    for line in md.splitlines():
        if line.startswith("## "):
            low = line.lower()
            section = ("link" if "link suggestion" in low
                       else "tag" if "tag suggestion" in low
                       else "moc" if "moc placement" in low
                       else "idea" if "idea link" in low else None)
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
    return {"links": links, "tags": tags, "mocs": mocs, "ideas": ideas}


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


_FM_TAGS_KEY_RE = re.compile(r"^tags\s*:")
_FM_TAGS_INLINE_LIST_RE = re.compile(r"^tags\s*:\s*\[(.*)\]\s*$")
_FM_TAGS_INLINE_VALUE_RE = re.compile(r"^tags\s*:\s*([^\s\[#].*?)\s*$")
_FM_LIST_ITEM_RE = re.compile(r"^\s*-\s+(.+?)\s*$")


def _find_tags_field(fm_lines: list[str]) -> tuple[list[str] | None, int | None, int | None]:
    """Locate the `tags:` field within a list of frontmatter lines. Handles the
    three shapes seen in the wild: block list, inline list (`[a, b]`, incl.
    empty `[]`), and inline single value. Returns (tags, start, end) with `end`
    exclusive, or (None, None, None) if there's no tags field at all."""
    for i, line in enumerate(fm_lines):
        if not _FM_TAGS_KEY_RE.match(line):
            continue
        m = _FM_TAGS_INLINE_LIST_RE.match(line)
        if m:
            inner = m.group(1).strip()
            if not inner:
                return [], i, i + 1
            return [x.strip().strip("\"'") for x in inner.split(",") if x.strip()], i, i + 1
        m = _FM_TAGS_INLINE_VALUE_RE.match(line)
        if m:
            return [m.group(1).strip().strip("\"'")], i, i + 1
        # Block list form: tags:\n  - a\n  - b (blank lines inside are tolerated).
        tags: list[str] = []
        end = i + 1
        for j in range(i + 1, len(fm_lines)):
            item = _FM_LIST_ITEM_RE.match(fm_lines[j])
            if item:
                tags.append(item.group(1).strip().strip("\"'"))
                end = j + 1
            elif fm_lines[j].strip() == "":
                end = j + 1
            else:
                break
        return tags, i, end
    return None, None, None


def merge_frontmatter_tags(content: str, new_tags: list[str]) -> str | None:
    """Merge `new_tags` into a note's frontmatter `tags:` field, order-preserving
    and deduped (existing tags first), always writing the result back as a
    2-space block list. Returns None if the frontmatter is malformed (an
    opening `---` with no closing one) so the caller can skip tags for that
    note rather than corrupt it. Body is never touched; trailing-newline
    presence/absence is preserved.

    A pure, stdlib-only reimplementation of the vault's migrate_tags.py merge
    semantics — trellis ships this natively so tag application doesn't degrade
    to links-only for anyone without that script on disk.
    """
    has_trailing_newline = content.endswith("\n")
    raw_lines = content.split("\n")
    if has_trailing_newline and raw_lines and raw_lines[-1] == "":
        raw_lines = raw_lines[:-1]

    if not raw_lines or raw_lines[0].strip() != "---":
        # No frontmatter at all — prepend a brand-new block, body untouched.
        block = "\n".join(["tags:"] + [f"  - {t}" for t in dict.fromkeys(new_tags)])
        return f"---\n{block}\n---\n{content}"

    close_idx = None
    for i in range(1, len(raw_lines)):
        if raw_lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx is None:
        return None  # unclosed frontmatter — caller warns and skips tags

    fm_lines = raw_lines[1:close_idx]
    body_lines = raw_lines[close_idx + 1:]

    existing_tags, start, end = _find_tags_field(fm_lines)
    merged = list(dict.fromkeys((existing_tags or []) + new_tags))
    tags_block = ["tags:"] + [f"  - {t}" for t in merged]

    if start is not None:
        new_fm_lines = fm_lines[:start] + tags_block + fm_lines[end:]
    else:
        new_fm_lines = list(fm_lines)
        while new_fm_lines and new_fm_lines[-1].strip() == "":
            new_fm_lines.pop()
        new_fm_lines.extend(tags_block)

    new_content = "\n".join(["---"] + new_fm_lines + ["---"] + body_lines)
    if has_trailing_newline:
        new_content += "\n"
    return new_content


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
        want = probe.group(1).strip().lower()
        seg = "\n".join(lines[start:end])
        existing = {m.split("|")[0].split("#")[0].strip().lower()
                    for m in _LINK_RE.findall(seg)}
        if want in existing:
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


def _dated_report_path(out_dir: str, date_str: str) -> str:
    """Path for a dated report in out_dir (created if missing), timestamp-suffixed
    so an earlier run on the same day is never clobbered."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{date_str}.md")
    if os.path.exists(path):
        path = os.path.join(out_dir, f"{date_str}-{time.strftime('%H%M')}.md")
    return path


def _pending_reviews(cfg) -> list[str]:
    """List pending gardener review files: top-level `.md` files in the configured
    gardener folder. The `applied/` archive subdir (and anything below it) is ignored."""
    gdir = os.path.join(cfg["vault"], cfg["gardener_dir"])
    if not os.path.isdir(gdir):
        return []
    return sorted(
        os.path.join(gdir, f)
        for f in os.listdir(gdir)
        if f.endswith(".md") and os.path.isfile(os.path.join(gdir, f))
    )


def cmd_apply(cfg, args):
    if not _require_vault(cfg):
        return 1

    if args.file:
        path = args.file
        if not os.path.exists(path):
            alt = os.path.join(cfg["vault"], cfg["gardener_dir"], path)
            if os.path.exists(alt):
                path = alt
        if not os.path.exists(path):
            print(f"error: review file not found: {args.file}", file=sys.stderr)
            return 1
        paths = [path]
    else:
        paths = _pending_reviews(cfg)
        if not paths:
            print("no pending reviews in the gardener folder — nothing to apply")
            return 0
        print(f"applying {len(paths)} pending review(s) from the gardener folder")

    multi = len(paths) > 1
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
    return 0


def _apply_review_file(cfg, path, dry_run) -> tuple[int, int, int, int, int]:
    """Apply the checked items from a single review file into the vault's notes.
    Returns (applied_links, applied_tags, applied_mocs, applied_ideas,
    source_note_count)."""
    review = parse_review(open(path, encoding="utf-8").read())
    if not any(review[k] for k in ("links", "tags", "mocs", "ideas")):
        print(f"no checked items in {path} — nothing to apply")
        if not dry_run:  # explicit apply = retire the review anyway
            archived = _archive_review(path)
            if archived:
                print(f"archived review → {archived}")
        return 0, 0, 0, 0, 0

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

    add_mocs: dict[str, list] = collections.defaultdict(list)   # moc -> [(note, section)]
    add_ideas: dict[str, list] = collections.defaultdict(list)  # idea -> [(note, reason)]
    for note_t, moc_t, section in review["mocs"]:
        add_mocs[moc_t].append((note_t, section))
    for note_t, idea_t, reason in review["ideas"]:
        add_ideas[idea_t].append((note_t, reason))

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
                    if t not in note["tags"]]
        if not new_links and not new_tags:
            print(f"  = up to date: {s}")
            continue

        if dry_run:
            print(f"  {rel}")
            for t in new_tags:
                print(f"      + tag   {t}")
            for t in new_links:
                print(f"      + link  [[{t}]]")
        else:
            full = os.path.join(cfg["vault"], rel)
            content = open(full, encoding="utf-8").read()
            if new_tags:  # fold into frontmatter natively
                merged = merge_frontmatter_tags(content, new_tags)
                if merged is None:
                    print(f"  ! tag merge failed for {s} (malformed frontmatter); "
                          "leaving tags", file=sys.stderr)
                    new_tags = []
                else:
                    content = merged
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

    applied_mocs = applied_ideas = 0
    date_str = datetime.date.today().isoformat()

    def _edit_target(title, kind, editor):
        """Apply editor(content) to the note titled `title`; count via return."""
        rel = title_to_rel.get(title.lower())
        if not rel:
            print(f"  ! {kind} target not found, skipping: {title}", file=sys.stderr)
            return 0
        scopes = tuple(cfg.get("moc_scope", ["MOCs/"])) if kind == "MOC" \
            else tuple(cfg.get("idea_scope", ["Areas/Product Ideas/"]))
        if scopes and not rel.startswith(scopes):
            print(f"  ! {kind} target [[{title}]] resolves outside {scopes} "
                  f"({rel}); skipping", file=sys.stderr)
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

    if not dry_run:
        conn.commit()
    head = "DRY RUN — would apply" if dry_run else "applied"
    print(f"\n{head}: {applied_links} link(s) · {applied_tags} tag(s) · "
          f"{applied_mocs} MOC placement(s) · {applied_ideas} idea link(s) "
          f"across {len(sources)} source note(s)")
    if not dry_run:
        archived = _archive_review(path)
        if archived:
            print(f"archived review → {archived}")
    return applied_links, applied_tags, applied_mocs, applied_ideas, len(sources)


# --------------------------------------------------------------------------- #
# Migrate: legacy "Added by Claude on <date>:" link blocks -> one section
# --------------------------------------------------------------------------- #
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
    skill_state = None
    if os.path.exists(state_json):
        try:
            with open(state_json, encoding="utf-8") as fh:
                skill_state = json.load(fh)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"warning: could not read {state_json}: {e}", file=sys.stderr)
    if skill_state is not None and not args.dry_run:
        imported = seed_triage_state(conn, skill_state)
        if imported:
            print(f"seeded triage state from triage-state.json ({imported} notes)")

    last_run = meta_get(conn, "triage_last_run")
    dry_seed_triaged: set = set()
    if last_run is None and args.dry_run and skill_state is not None:
        # Dry run must not persist the seed; derive the same view in memory.
        last_run = skill_state.get("last_run_iso")
        dry_seed_triaged = {"z/" + n for n in (skill_state.get("triaged") or [])}
    if last_run is None:
        if args.dry_run:
            print("no triage state found — a real run would initialize the "
                  "baseline (nothing written in dry run)")
            return 0
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
    if not args.force:
        triaged |= dry_seed_triaged

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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(prog="trellis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    pa.add_argument("file", nargs="?",
                    help="path to a gardener review .md (or just its filename); "
                         "omit to apply all pending reviews in the gardener folder")
    pa.add_argument("--dry-run", action="store_true",
                    help="show what would change; write nothing")

    pcl = sub.add_parser("cluster", help="detect MOC-candidate clusters -> review report")
    pcl.add_argument("--scope", help="comma-separated path prefixes (default: z/)")
    pcl.add_argument("--limit", type=int, help="max candidates to name/report (0 = no cap)")
    pcl.add_argument("--gen-model", dest="gen_model", help="judgment model for naming")
    pcl.add_argument("--force", action="store_true", help="ignore the seen-ledger")
    pcl.add_argument("--dry-run", action="store_true",
                     help="print report; write nothing (no ledger, no file)")

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

    args = p.parse_args(argv)
    cfg = load_config({"vault": args.vault, "embed_model": args.embed_model,
                       "db_path": args.db_path,
                       "gen_model": getattr(args, "gen_model", None)})
    return {
        "index": cmd_index, "search": cmd_search,
        "neighbors": cmd_neighbors, "status": cmd_status, "garden": cmd_garden,
        "apply": cmd_apply, "cluster": cmd_cluster, "triage": cmd_triage,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""embedx: embeddings-powered semantic search over a project folder.

usage:
  python3 embedx.py index <dir> [--exts .py,.md] [--index PATH]
  python3 embedx.py search "query" [--dir .] [--top 8]
  python3 embedx.py status [--dir .]

config: --base-url (or EMBEDX_BASE_URL / OPENAI_BASE_URL),
        --api-key (or EMBEDX_API_KEY / OPENAI_API_KEY),
        --model (or EMBEDX_MODEL, default text-embedding-3-small).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import urllib.request

CHUNK_LINES = 40
OVERLAP = 10
BATCH = 64
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv"}
INDEX_NAME = ".embedx.json"


class EmbedxError(Exception):
    pass


def base_url_default():
    return (os.environ.get("EMBEDX_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1")


def api_key_default():
    return (os.environ.get("EMBEDX_API_KEY")
            or os.environ.get("OPENAI_API_KEY") or "")


def model_default():
    return os.environ.get("EMBEDX_MODEL", "text-embedding-3-small")


def is_binary(path):
    # null bytes in the head mean binary, skip it
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def read_text(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def iter_files(root, exts):
    # walk the tree, pruning hidden and junk dirs
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in SKIP_DIRS]
        for name in filenames:
            if name.startswith("."):
                continue
            if exts and not any(name.endswith(e) for e in exts):
                continue
            full = os.path.join(dirpath, name)
            if is_binary(full):
                continue
            found.append(full)
    return sorted(found)


def chunk_lines(lines, n=CHUNK_LINES, overlap=OVERLAP):
    # sliding windows of n lines, 1-based line numbers
    step = max(1, n - overlap)
    chunks = []
    i = 0
    while i < len(lines):
        start = i + 1
        end = min(i + n, len(lines))
        chunks.append((start, end, "".join(lines[i:end])))
        if end == len(lines):
            break
        i += step
    return chunks


def embed(texts, base_url, api_key, model, batch=BATCH):
    # post batches to /v1/embeddings, return one vector per text
    if not api_key:
        raise EmbedxError("set OPENAI_API_KEY or pass --api-key")
    url = base_url.rstrip("/") + "/embeddings"
    vecs = []
    for i in range(0, len(texts), batch):
        payload = json.dumps({"model": model, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + api_key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            raise EmbedxError("embeddings request failed: %s" % e)
        try:
            items = sorted(body["data"], key=lambda d: d["index"])
            vecs.extend(d["embedding"] for d in items)
        except (KeyError, TypeError):
            raise EmbedxError("bad embeddings response")
    if len(vecs) != len(texts):
        raise EmbedxError("embedding count mismatch")
    return vecs


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def index_path_for(args, d):
    return args.index if args.index else os.path.join(d, INDEX_NAME)


def load_index(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise EmbedxError("cannot read index %s: %s" % (path, e))


def file_hash(path):
    # sha256 of the raw bytes, used to skip unchanged files on re-index
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for buf in iter(lambda: f.read(65536), b""):
                h.update(buf)
    except OSError:
        return None
    return h.hexdigest()


def cmd_index(args):
    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        raise EmbedxError("not a directory: %s" % args.dir)
    exts = None
    if args.exts:
        exts = [e if e.startswith(".") else "." + e
                for e in args.exts.split(",") if e.strip()]
    files = iter_files(root, exts)
    ipath = index_path_for(args, root)
    # never index our own output file
    ipath_abs = os.path.abspath(ipath)
    files = [f for f in files if os.path.abspath(f) != ipath_abs]
    old = None
    if os.path.exists(ipath):
        try:
            old = load_index(ipath)
        except EmbedxError:
            old = None  # corrupt index, rebuild from scratch
    old_chunks = {}
    old_hashes = {}
    if isinstance(old, dict):
        if not args.force and old.get("model", args.model) == args.model:
            old_hashes = old.get("hashes") or {}
            for c in old.get("chunks") or []:
                old_chunks.setdefault(c["file"], []).append(c)
        # a changed model re-embeds everything: old vectors belong to a
        # different model and would poison search results
    chunks = []
    todo = []  # (chunk index, text) pairs that still need embedding
    hashes = {}
    reused = 0
    for path in files:
        text = read_text(path)
        if text is None:
            continue
        rel = os.path.relpath(path, root)
        digest = file_hash(path)
        hashes[rel] = digest
        if digest and old_hashes.get(rel) == digest and rel in old_chunks:
            # unchanged since the last index, keep the stored chunks as-is
            chunks.extend(old_chunks[rel])
            reused += len(old_chunks[rel])
            continue
        for start, end, ctext in chunk_lines(text.splitlines(keepends=True)):
            todo.append((len(chunks), ctext))
            chunks.append({"file": rel, "start": start, "end": end,
                           "text": ctext})
    if not chunks and old is None:
        print("embedx: nothing to index in %s" % root)
        return 0
    if todo:
        vecs = embed([t for _, t in todo], args.base_url, args.api_key,
                     args.model, batch=args.batch)
        for (i, _), v in zip(todo, vecs):
            chunks[i]["vec"] = v
    dims = 0
    for c in chunks:
        if "vec" in c:
            dims = len(c["vec"])
            break
    if not dims:
        dims = (old or {}).get("dims", 0)
    data = {"version": 2, "model": args.model, "dims": dims,
            "files": sorted({c["file"] for c in chunks}),
            "hashes": hashes, "chunks": chunks}
    with open(ipath, "w") as f:
        json.dump(data, f)
    print("embedx: %d chunks (%d reused, %d embedded) from %d files -> %s"
          % (len(chunks), reused, len(todo), len(data["files"]), ipath))
    return 0


def cmd_search(args):
    root = os.path.abspath(args.dir)
    data = load_index(index_path_for(args, root))
    qvec = embed([args.query], args.base_url, args.api_key,
                 data.get("model") or args.model)[0]
    scored = []
    for c in data["chunks"]:
        scored.append((cosine(qvec, c["vec"]), c))
    scored.sort(key=lambda s: s[0], reverse=True)
    for score, c in scored[:args.top]:
        print("%s:%d-%d  %.4f" % (c["file"], c["start"], c["end"], score))
        for line in c["text"].splitlines()[:2]:
            line = line.strip()
            if line:
                print("    %s" % line[:100])
                break
    return 0


def cmd_status(args):
    root = os.path.abspath(args.dir)
    ipath = index_path_for(args, root)
    data = load_index(ipath)
    size = os.path.getsize(ipath)
    print("index: %s" % ipath)
    print("files: %d" % len(data.get("files", [])))
    print("chunks: %d" % len(data.get("chunks", [])))
    print("model: %s" % data.get("model", "?"))
    print("dims: %s" % data.get("dims", "?"))
    print("size: %.1f KB" % (size / 1024))
    return 0


def add_common(p):
    p.add_argument("--base-url", default=base_url_default())
    p.add_argument("--api-key", default=api_key_default())
    p.add_argument("--model", default=model_default())
    p.add_argument("--index", default=None,
                   help="index file path (default <dir>/.embedx.json)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="embeddings-powered semantic search over a folder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="chunk a dir and embed it")
    pi.add_argument("dir")
    pi.add_argument("--exts", default=None,
                    help="comma list of extensions, e.g. .py,.md")
    pi.add_argument("--batch", type=int, default=BATCH)
    pi.add_argument("--force", action="store_true",
                    help="re-embed everything, ignore stored file hashes")
    add_common(pi)

    ps = sub.add_parser("search", help="search the index")
    ps.add_argument("query")
    ps.add_argument("--dir", default=".")
    ps.add_argument("--top", type=int, default=8)
    add_common(ps)

    pt = sub.add_parser("status", help="show index stats")
    pt.add_argument("--dir", default=".")
    add_common(pt)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "index":
            return cmd_index(args)
        if args.cmd == "search":
            return cmd_search(args)
        return cmd_status(args)
    except EmbedxError as e:
        print("embedx: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

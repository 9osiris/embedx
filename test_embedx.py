#!/usr/bin/env python3
# tests for embedx.py against a fake local embeddings server. no real api.
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embedx

EMBEDX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedx.py")


def vec_for(text):
    # scripted vectors: keyword texts land on unit axes, else hash noise
    t = text.lower()
    if "apple" in t:
        return [1.0, 0.0, 0.0, 0.0]
    if "banana" in t:
        return [0.0, 1.0, 0.0, 0.0]
    if "cherry" in t:
        return [0.0, 0.0, 1.0, 0.0]
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:4]]


class Handler(BaseHTTPRequestHandler):
    count = 0  # embedding requests served, so tests can assert skip behavior

    def do_POST(self):
        Handler.count += 1
        assert self.path == "/v1/embeddings", self.path
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [{"object": "embedding", "index": i,
                 "embedding": vec_for(t)} for i, t in enumerate(inputs)]
        resp = json.dumps({"object": "list", "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


def start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def reset_count():
    Handler.count = 0


BASE = None
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL: %s" % name)


def write(root, rel, data):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(p, mode) as f:
        f.write(data)
    return p


def sample_tree(root):
    # three keyword files plus noise to exercise the chunker
    write(root, "a.py", "\n".join("apple line %d" % i for i in range(100)))
    write(root, "b.py", "\n".join("banana line %d" % i for i in range(50)))
    write(root, "docs/c.md", "# cherry docs\n\ncherry pie recipe\n")
    write(root, ".hidden.py", "apple hidden\n")
    write(root, "bin.dat", b"\x00\x01\x02binary")
    os.makedirs(os.path.join(root, ".git"))
    write(root, ".git/HEAD", "apple in git\n")


def run_cli(*argv, env=None):
    e = dict(os.environ)
    e["OPENAI_API_KEY"] = "x"
    e["EMBEDX_BASE_URL"] = BASE
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, EMBEDX] + list(argv),
                       capture_output=True, text=True, env=e, timeout=60)
    return r


def test_chunk_windows():
    lines = ["l%d\n" % i for i in range(100)]
    chunks = embedx.chunk_lines(lines)
    check("chunk count for 100 lines", len(chunks) == 3)
    check("first window bounds", chunks[0][0] == 1 and chunks[0][1] == 40)
    check("overlap of 10", chunks[1][0] == 31)
    check("last window ends at 100", chunks[-1][1] == 100)
    short = embedx.chunk_lines(["a\n", "b\n"])
    check("short file one chunk", len(short) == 1 and short[0][1] == 2)


def test_iter_files():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        files = embedx.iter_files(d, None)
        rels = [os.path.relpath(f, d) for f in files]
        check("finds py and md", "a.py" in rels and "docs/c.md" in rels)
        check("skips hidden", ".hidden.py" not in rels)
        check("skips binary", "bin.dat" not in rels)
        check("skips .git", not any(r.startswith(".git") for r in rels))
        py = embedx.iter_files(d, [".py"])
        check("ext filter", all(f.endswith(".py") for f in py)
              and len(py) == 2)


def test_cosine():
    check("identical is 1", abs(embedx.cosine([1, 0], [1, 0]) - 1.0) < 1e-9)
    check("orthogonal is 0", abs(embedx.cosine([1, 0], [0, 1])) < 1e-9)
    check("zero vec is 0", embedx.cosine([0, 0], [1, 1]) == 0.0)


def test_embed_batch():
    vecs = embedx.embed(["apple x", "banana y", "cherry z"], BASE, "k",
                        "m", batch=2)
    check("batch returns all vecs", len(vecs) == 3)
    check("apple maps to x axis", vecs[0] == [1.0, 0.0, 0.0, 0.0])
    check("banana maps to y axis", vecs[1] == [0.0, 1.0, 0.0, 0.0])
    try:
        embedx.embed(["q"], BASE, "", "m")
        check("empty key raises", False)
    except embedx.EmbedxError:
        check("empty key raises", True)


def test_index_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        r = run_cli("index", d, "--index", ipath)
        check("index exit 0", r.returncode == 0)
        data = json.load(open(ipath))
        check("index version", data["version"] == 2)
        check("index model", data["model"] == "text-embedding-3-small")
        check("index dims", data["dims"] == 4)
        check("index files", set(data["files"]) == {"a.py", "b.py", "docs/c.md"})
        check("chunk count", len(data["chunks"]) == 3 + 2 + 1)
        c0 = data["chunks"][0]
        check("chunk fields", c0["file"] == "a.py" and c0["start"] == 1
              and c0["end"] == 40 and len(c0["vec"]) == 4)
        # reload through the loader the search path uses
        data2 = embedx.load_index(ipath)
        check("round trip stable", data2["chunks"] == data["chunks"])


def test_search_ranking():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        r = run_cli("search", "apple pie", "--index", ipath, "--top", "3")
        check("search exit 0", r.returncode == 0)
        lines = r.stdout.splitlines()
        check("search prints hits", len(lines) >= 3)
        check("apple chunk ranks first", lines[0].startswith("a.py:"))
        check("score shown", "1.0000" in lines[0])
        r2 = run_cli("search", "banana bread", "--index", ipath, "--top", "1")
        check("banana ranks first", r2.stdout.splitlines()[0].startswith("b.py:"))


def test_search_default_index():
    # search finds .embedx.json in --dir by default
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        r = run_cli("index", d)
        check("default index path", os.path.exists(os.path.join(d, ".embedx.json")))
        r2 = run_cli("search", "cherry tart", "--dir", d, "--top", "1")
        check("default index search", r2.stdout.splitlines()[0].startswith("docs/c.md"))


def test_status():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        r = run_cli("status", "--dir", d, "--index", ipath)
        check("status exit 0", r.returncode == 0)
        check("status files", "files: 3" in r.stdout)
        check("status chunks", "chunks: 6" in r.stdout)
        check("status model", "model: text-embedding-3-small" in r.stdout)
        check("status dims", "dims: 4" in r.stdout)


def test_cli_errors():
    r = run_cli("index", "/no/such/dir")
    check("bad dir nonzero", r.returncode == 1)
    with tempfile.TemporaryDirectory() as d:
        r = run_cli("search", "q", "--dir", d)
        check("missing index nonzero", r.returncode == 1)


def vecs_by_file(data):
    out = {}
    for c in data["chunks"]:
        out.setdefault(c["file"], []).append(c["vec"])
    return out


def test_reindex_skips_unchanged():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        r = run_cli("index", d, "--index", ipath)
        check("first index ok", r.returncode == 0)
        first = json.load(open(ipath))
        reset_count()
        r2 = run_cli("index", d, "--index", ipath, "--batch", "1")
        check("reindex exit 0", r2.returncode == 0)
        check("no api calls on unchanged tree", Handler.count == 0)
        check("reuse reported", "reused" in r2.stdout)
        second = json.load(open(ipath))
        check("chunks identical", second["chunks"] == first["chunks"])
        check("hashes stored", len(second["hashes"]) == 3)


def test_reindex_changed_file():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        before = vecs_by_file(json.load(open(ipath)))
        write(d, "docs/c.md", "# cherry docs\n\napple pie recipe\n")
        reset_count()
        r = run_cli("index", d, "--index", ipath, "--batch", "1")
        check("reindex after edit ok", r.returncode == 0)
        check("only changed file embedded", Handler.count == 1)
        after = vecs_by_file(json.load(open(ipath)))
        check("unchanged vecs kept",
              after["a.py"] == before["a.py"] and after["b.py"] == before["b.py"])
        check("changed chunk re-embedded", after["docs/c.md"] != before["docs/c.md"])
        check("chunk count stable", len(after["docs/c.md"]) == 1)


def test_reindex_deleted_file():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        os.remove(os.path.join(d, "b.py"))
        reset_count()
        r = run_cli("index", d, "--index", ipath, "--batch", "1")
        check("reindex after delete ok", r.returncode == 0)
        check("no embeds for a delete", Handler.count == 0)
        data = json.load(open(ipath))
        check("deleted chunks dropped",
              all(c["file"] != "b.py" for c in data["chunks"]))
        check("files list updated", set(data["files"]) == {"a.py", "docs/c.md"})


def test_force_reembeds():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        reset_count()
        r = run_cli("index", d, "--index", ipath, "--force", "--batch", "1")
        check("force exit 0", r.returncode == 0)
        check("force re-embeds all", Handler.count == 6)


def test_model_change_reembeds():
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        reset_count()
        r = run_cli("index", d, "--index", ipath, "--batch", "1",
                    "--model", "other-model")
        check("model switch exit 0", r.returncode == 0)
        check("model change re-embeds all", Handler.count == 6)
        data = json.load(open(ipath))
        check("model updated in index", data["model"] == "other-model")


def test_hashless_index_reembeds():
    # a v1 index has no hashes, the first re-index embeds everything once
    with tempfile.TemporaryDirectory() as d:
        sample_tree(d)
        ipath = os.path.join(d, "idx.json")
        run_cli("index", d, "--index", ipath)
        data = json.load(open(ipath))
        del data["hashes"]
        data["version"] = 1
        json.dump(data, open(ipath, "w"))
        reset_count()
        r = run_cli("index", d, "--index", ipath, "--batch", "1")
        check("hashless index exit 0", r.returncode == 0)
        check("hashless index re-embeds all", Handler.count == 6)
        check("hashes added on rewrite", "hashes" in json.load(open(ipath)))


if __name__ == "__main__":
    srv = start_server()
    BASE = "http://127.0.0.1:%d/v1" % srv.server_address[1]
    try:
        test_chunk_windows()
        test_iter_files()
        test_cosine()
        test_embed_batch()
        test_index_roundtrip()
        test_search_ranking()
        test_search_default_index()
        test_status()
        test_cli_errors()
        test_reindex_skips_unchanged()
        test_reindex_changed_file()
        test_reindex_deleted_file()
        test_force_reembeds()
        test_model_change_reembeds()
        test_hashless_index_reembeds()
    finally:
        srv.shutdown()
    print("%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)

# embedx

Embeddings-powered semantic search over a project folder. Chunk a directory,
embed the chunks with any OpenAI-compatible embeddings API, then search by
meaning instead of keywords. Stdlib only, one file.

## usage

```sh
# index a project (writes .embedx.json into the dir)
python3 embedx.py index ~/code/myproj --exts .py,.md

# search it
python3 embedx.py search "where is retry logic handled" --dir ~/code/myproj --top 5

# index stats
python3 embedx.py status --dir ~/code/myproj
```

search output looks like:

```
src/client.py:41-80  0.7312
    def _request_with_retry(self, req):
src/agent.py:112-151  0.6844
    for attempt in range(max_retries):
```

## config

Flags beat env vars. The API key is required for index and search.

| flag | env | default |
|---|---|
| `--base-url` | `EMBEDX_BASE_URL` or `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `--api-key` | `EMBEDX_API_KEY` or `OPENAI_API_KEY` | (none) |
| `--model` | `EMBEDX_MODEL` | `text-embedding-3-small` |
| `--index` | | `<dir>/.embedx.json` |

`index` also takes `--exts .py,.md` to filter files and `--batch N`
(default 64) for the embeddings request batch size.

Chunking: text files are split into 40-line windows with 10 lines of
overlap. Binaries (null bytes), hidden files, and `.git` / `node_modules`
/ `__pycache__` and friends are skipped.

Re-running `index` is incremental: each file's sha256 is stored in the
index, and unchanged files keep their stored chunks with no new API
calls. Only new or edited files get embedded, and files deleted from
disk drop out of the index. `--force` re-embeds everything. Changing
`--model` also rebuilds the whole index, because old vectors belong to
a different model.

## tests

```sh
python3 test_embedx.py
```

Runs against a fake local embeddings server with scripted vectors, no
network and no API key. Covers chunking, file filtering, cosine math,
batched embedding calls, index round-trip, search ranking order,
incremental re-indexing (unchanged files skipped, edits re-embedded,
deletes dropped, --force, model changes, hashless v1 indexes), status,
and CLI error paths.

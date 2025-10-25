# RAG Indexing Service

Ingest files into `TheBrain` (a Weaviate collection) with local embeddings.

- Recursive discovery
- Parsing and chunking
- Per-file summaries and metadata
- Duplicate detection via SHA-256
- Weaviate storage with Ollama embeddings

> Prereqs: `ollama serve` and `make awaken_hive`.

## 1) Ingest Script

```bash
make ingest INGEST_DIR="/path/to/your/docs"
```

Defaults (override via env):
- `WEAVIATE_COLLECTION="TheBrain"`
- `EMBED_MODEL="bge-m3"`
- `SUMMARY_MODEL="qwen3:14b"`

## 2) RAG Querying (CLI)

```bash
make ask QUERY="Based on TheBrain, what are the primary project goals?"
```

## 3) RAG Querying (Notebook)

```python
from hivemind import HiveMind

hive = HiveMind()
hive.add_drone(
    name="Cortex",
    model="qwen3:14b",
    persona="You are a senior systems architect focused on robust, scalable infrastructure."
)

hive.brainscan(
    drone_name="Cortex",
    query="Based on the documents I provided, what are the key security concerns for the project?"
)
```

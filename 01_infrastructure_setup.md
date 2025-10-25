# mdLLMngr Control Tower - Infrastructure Setup

Sets up the mdLLMngr control stack with:
- Ollama (local LLM runtime)
- Weaviate (local vector DB for RAG)
- theLab Docker container (for safe code execution)
- Python project managed by `uv`

---

## 0) Prerequisites

- macOS, Linux, or Windows with WSL2
- Git
- Docker Desktop (for RAG and the execution lab)
- `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1) Project Layout

```
mdllmngr/
|-- compose/
|   |-- docker-compose.yml
|   `-- the_lab/
|-- hivemind/
|   |-- __init__.py
|   |-- session.py
|   `-- resources/
|       |-- __init__.py
|       |-- lab.py
|       `-- codex.py
|-- tools/
|   |-- ingest.py
|   `-- brainscan.py
|-- chats/
|-- the_wormhole/
|-- Makefile
|-- pyproject.toml
`-- README.md
```

---

## 2) Python environment via uv

```bash
uv venv .venv
uv sync
```

---

## 3) Ollama

```bash
ollama serve
make pull-models
```

---

## 4) Optional Services: Weaviate & theLab

```bash
make awaken_hive  # starts Weaviate + theLab
```

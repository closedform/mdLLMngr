import os, hashlib, pathlib
from typing import Iterable, Dict, Any
import weaviate
from weaviate.classes.config import Configure
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm
import ollama
from hivemind.resources.codex import extract_ipynb_text, try_export_mathematica_nb_to_md

OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "TheBrain")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "qwen3:14b")

def _connect():
    return weaviate.connect_to_local(grpc_port=50051, http_host="localhost", http_port=8080)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def summarize_text(text: str, fname: str) -> str:
    prompt = (f"Summarize this file for an index. Include title guess, topics, "
              f"notable functions/classes if code, and 1-2 tags.\n\n"
              f"FILE: {fname}\n\nCONTENT:\n{text[:8000]}")
    r = ollama.chat(model=SUMMARY_MODEL, messages=[{"role":"user","content":prompt}], options={"temperature":0.2})
    return r["message"]["content"].strip()

def chunks_from_path(p: pathlib.Path) -> Iterable[Dict[str, Any]]:
    if p.suffix.lower() == ".ipynb":
        ipy = extract_ipynb_text(str(p))
        CH = 1200; OL = 150
        blocks = [ipy[i:i+CH] for i in range(0, len(ipy), CH-OL)]
        for b in blocks:
            yield {"text": b, "source": str(p), "section": "notebook", "page": None}
        return
    if p.suffix.lower() == ".nb":
        md = try_export_mathematica_nb_to_md(str(p))
        if md and os.path.exists(md):
            for c in chunks_from_path(pathlib.Path(md)):
                c["source"] = str(p); yield c
            try: os.remove(md)
            except: pass
            return
    try:
        elements = partition(filename=str(p), strategy="auto")
        chunks = chunk_by_title(elements, max_characters=1200, new_after_n_chars=900, overlap=150)
        for ch in chunks:
            yield {"text": ch.text, "source": str(p), "section": getattr(ch.metadata, "category", None), "page": getattr(ch.metadata, "page_number", None)}
        return
    except Exception:
        pass
    st = p.stat()
    desc = (f"[BINARY or unsupported] name={p.name} ext={p.suffix} size={st.st_size} "
            f"mtime={int(st.st_mtime)} path={p}")
    yield {"text": desc, "source": str(p), "section": "binary", "page": None}

def ensure_collection(client):
    try:
        return client.collections.get(COLLECTION)
    except Exception:
        return client.collections.create(
            name=COLLECTION,
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(api_endpoint=OLLAMA_ENDPOINT, model=EMBED_MODEL),
            generative_config=Configure.Generative.ollama(api_endpoint=OLLAMA_ENDPOINT, model="llama3.1:8b"),
        )

def ingest_dir(root: str):
    client = _connect()
    coll = ensure_collection(client)
    to_insert, seen = [], {}
    files = list(pathlib.Path(root).rglob("*"))
    for p in tqdm(files, desc=f"Uploading knowledge to {COLLECTION}"):
        if not p.is_file(): continue
        try: filehash = sha256_file(str(p))
        except Exception: continue
        if filehash in seen:
            to_insert.append({"text": f"[DUPLICATE of {seen[filehash]}] {p.name}", "source": str(p), "section": "duplicate", "page": None, "hash": filehash})
            if len(to_insert) >= 256: coll.data.insert_many(to_insert); to_insert.clear()
            continue
        seen[filehash] = str(p)
        all_text = []
        for ch in chunks_from_path(p):
            all_text.append(ch["text"])
            to_insert.append(ch)
            if len(to_insert) >= 256: coll.data.insert_many(to_insert); to_insert.clear()
        if all_text:
            try:
                meta = summarize_text("\n\n".join(all_text[:8]), str(p))
                to_insert.append({"text": meta, "source": str(p), "section": "summary", "page": None, "hash": filehash})
            except Exception:
                pass
        if len(to_insert) >= 256: coll.data.insert_many(to_insert); to_insert.clear()
    if to_insert: coll.data.insert_many(to_insert)
    client.close()

if __name__ == "__main__":
    root = os.environ.get("INGEST_DIR", ".")
    print(f"Ingesting: {root}")
    ingest_dir(root)

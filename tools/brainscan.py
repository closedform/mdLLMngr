import os, sys, textwrap, weaviate, ollama

COLLECTION = os.environ.get("WEAVIATE_COLLECTION","TheBrain")
GEN_MODEL = os.environ.get("GEN_MODEL","qwen3:14b")
TOPK = int(os.environ.get("TOPK","8"))

def main():
    q = os.environ.get("QUERY") or (sys.argv[1] if len(sys.argv)>1 else f"Summarize the contents of {COLLECTION}.")
    client = weaviate.connect_to_local(grpc_port=50051, http_host="localhost", http_port=8080)
    try:
        docs = client.collections.get(COLLECTION)
        hits = docs.query.near_text(query=q, limit=TOPK).objects
        if not hits:
            print(f"No results in collection '{COLLECTION}'. Did you run `make ingest`?")
            return
        context = "\n\n---\n\n".join([h.properties.get("text","") for h in hits])
        prompt = f"""You are an assistant answering questions based on knowledge from TheBrain.
Use the provided context to answer the user query.

# Query
{q}

# Context from TheBrain (top {TOPK} chunks)
{context}
"""
        resp = ollama.chat(model=GEN_MODEL, messages=[{"role":"user","content":prompt}], options={"temperature":0.2})
        print(textwrap.fill(resp["message"]["content"], width=100))
    finally:
        client.close()

if __name__ == "__main__":
    main()

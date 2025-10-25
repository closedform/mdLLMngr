SHELL := /bin/bash

awaken_hive:
	@mkdir -p the_wormhole
	docker compose -f compose/docker-compose.yml up -d
	@echo "HiveMind support systems are online."
	@echo "Weaviate is at http://localhost:8080"

power_down_hive:
	docker compose -f compose/docker-compose.yml down

rebuild-thelab:
	@echo "Rebuilding theLab image..."
	docker compose -f compose/docker-compose.yml build the-lab
	@echo "Done. Restart with 'make power_down_hive' and 'make awaken_hive'."

pull-models:
	@echo "Pulling reasoning models..."
	ollama pull qwen3:14b
	ollama pull phi4:reasoning-14b || true
	ollama pull deepseek-r1:14b || true
	@echo "Pulling generalist and embedding models..."
	ollama pull llama3.1:8b-instruct
	ollama pull gemma2:9b-instruct || true
	ollama pull bge-m3 || ollama pull nomic-embed-text

ingest:
	INGEST_DIR?=.
	WEAVIATE_HOST=http://localhost:8080 \    	OLLAMA_ENDPOINT=http://localhost:11434 \    	WEAVIATE_COLLECTION=TheBrain \    	EMBED_MODEL=bge-m3 \    	SUMMARY_MODEL=qwen3:14b \    	uv run tools/ingest.py

ask:
	QUERY?="Summarize knowledge in TheBrain."
	WEAVIATE_HOST=http://localhost:8080 \    	WEAVIATE_COLLECTION=TheBrain \    	GEN_MODEL=qwen3:14b \    	uv run tools/brainscan.py

jlab:
	uv run jupyter lab

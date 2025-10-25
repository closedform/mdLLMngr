# Multi-Drone Conversation with HiveMind

Manage a swarm of AI Drones inside a Jupyter notebook.

- Swarm of Drones with names, models, and personas
- Address Drones with `@name`
- RAG via TheBrain
- Shared conversation history
- Optional code execution lab

> Requires: `ollama serve`. For RAG and code, `make awaken_hive`.

## Usage

```python
from hivemind import HiveMind

hive = HiveMind()
hive.add_drone("Cortex", "qwen3:14b", "Infra-focused architect.")
hive.add_drone("Cygnus", "gemma2:9b-instruct", "Creative strategist.")

hive.ask("@Cortex Outline a technical plan for a new real-time analytics service.")
hive.ask("@Cygnus What applications might be missing from Cortex's plan?")
```

## RAG

```python
hive.brainscan("Cortex", "What are the key security concerns for the project?")
```

# mdLLMngr Control Tower

**Start here.** mdLLMngr Control Tower is a local-first drone orchestration and retrieval stack built to let you play middle manager for your own fleet of LLM drones:
- `01_infrastructure_setup.md` - Base stack (Ollama, Docker, Makefile, Python project).
- `02_rag_indexing_service.md` - RAG pipeline for ingesting knowledge into `TheBrain`.
- `03_jupyter_hivemind.md` - A multi-agent Markdown command deck for AI `Drones`.
- `04_llm_thelab_execution.md` - theLab code execution within the `HiveMind`.

**Quick Start**

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd mdllmngr
    ```

2.  **Run Ollama**
    In a separate terminal, ensure the Ollama service is running.
    ```bash
    # Requires Ollama to be installed
    ollama serve
    ```

3.  **Set up the Environment & Tools**
    In your project terminal, set up the Python environment and start the support services.
    ```bash
    # Requires uv and Docker to be installed
    uv venv .venv && uv sync

    # Bring the Control Tower support systems online (Weaviate, theLab)
    make awaken_hive

    # Pull recommended LLMs
    make pull-models
    ```

4.  **Ingest Knowledge & Start Jupyter**
    ```bash
    # Ingest knowledge from the current directory into TheBrain
    make ingest INGEST_DIR="./"

    # Start the Jupyter interface
    make jlab
    ```

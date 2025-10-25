from __future__ import annotations

import json
import os
import pathlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ollama
import weaviate

from hivemind.resources import lab

try:
    from IPython.display import Markdown, display

    _HAS_IPY = True
except Exception:  # pragma: no cover - only triggered outside notebooks
    _HAS_IPY = False


@dataclass
class Drone:
    name: str
    model: str
    persona: str = "You are a helpful assistant."
    options: Dict = field(default_factory=dict)


class HiveMind:
    """Multi-drone orchestration with optional theLab execution and RAG access."""

    def __init__(self, execute: bool = False, mode: str = "moderated") -> None:
        self.id = str(uuid.uuid4())
        self.drones: Dict[str, Drone] = {}
        self.history: List[Dict[str, str]] = []
        self.execute = execute
        self.mode = mode  # Future placeholder for alternative coordination modes.
        self.workspace_dir = "the_wormhole"
        self.weaviate_collection = "TheBrain"
        pathlib.Path(self.workspace_dir).mkdir(exist_ok=True)
        self._weaviate_client: Optional[weaviate.WeaviateClient] = None

    # --------------------------------------------------------------------- #
    # Drone management
    # --------------------------------------------------------------------- #
    def add_drone(self, name: str, model: str, persona: str, options: Optional[Dict] = None) -> None:
        if name in self.drones:
            raise ValueError(f"Drone with name '{name}' already exists in the swarm.")
        if "Host" in name or "Brain" in name:
            raise ValueError("Drone name 'Host' or 'Brain' is reserved.")
        self.drones[name] = Drone(name=name, model=model, persona=persona, options=options or {})

    def list_drones(self) -> None:
        if not self.drones:
            print("No Drones in this swarm.")
            return
        print("Drones in the Swarm:")
        for name, drone in self.drones.items():
            print(f"- {name} (Model: {drone.model})")

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save_json(self, path: str) -> None:
        data = {
            "id": self.id,
            "history": self.history,
            "mode": self.mode,
            "execute": self.execute,
            "drones": {name: drone.__dict__ for name, drone in self.drones.items()},
        }
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "HiveMind":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        hive = cls(execute=data.get("execute", False), mode=data.get("mode", "moderated"))
        hive.id = data.get("id", str(uuid.uuid4()))
        hive.history = data.get("history", [])
        for name, payload in data.get("drones", {}).items():
            hive.drones[name] = Drone(**payload)
        return hive

    # --------------------------------------------------------------------- #
    # Display helpers
    # --------------------------------------------------------------------- #
    def to_markdown(self) -> str:
        lines = ["# HiveMind Session", ""]
        for msg in self.history:
            speaker = msg["name"]
            content = msg["content"].rstrip()
            lines.append(f"**{speaker}:**")
            lines.append("")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _display_markdown(self, md_text: str, handle=None):
        if not _HAS_IPY:
            print(md_text)
            return None
        if handle is None:
            return display(Markdown(md_text), display_id=True)
        handle.update(Markdown(md_text))
        return handle

    # --------------------------------------------------------------------- #
    # Core interaction
    # --------------------------------------------------------------------- #
    def ask(self, prompt: str, stream: bool = True) -> Optional[str]:
        target_match = re.search(r"@(\w+)", prompt)
        if not target_match:
            print("Host, please direct your message to a Drone using '@name'.")
            return None

        target_name = target_match.group(1)
        if target_name not in self.drones:
            print(f"Drone '{target_name}' not found in the swarm.")
            return None

        drone = self.drones[target_name]
        exec_result = self._execute_code_blocks(prompt) if self.execute else ""
        full_prompt = prompt + exec_result if exec_result else prompt
        self.history.append({"name": "Host", "content": full_prompt})

        messages: List[Dict[str, str]] = [{"role": "system", "content": drone.persona}]
        for msg in self.history:
            speaker = msg["name"]
            content = msg["content"]
            if speaker == target_name:
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": f"[{speaker}]: {content}"})

        reply = self._stream_and_display(
            name=target_name,
            model=drone.model,
            messages=messages,
            stream=stream,
            options=drone.options,
        )
        return reply

    def brainscan(self, drone_name: str, query: str, top_k: int = 5, stream: bool = True) -> Optional[str]:
        if drone_name not in self.drones:
            print(f"Drone '{drone_name}' not found in the swarm.")
            return None

        client = self._get_weaviate_client()
        try:
            collection = client.collections.get(self.weaviate_collection)
            hits = collection.query.near_text(query=query, limit=top_k).objects
        except Exception as exc:  # pragma: no cover - relies on external service
            print(f"Error querying TheBrain: {exc}. Did you run `make ingest`?")
            return None

        if not hits:
            print(f"TheBrain returned no context for query '{query}'.")
            return None

        context = "\n\n---\n\n".join([hit.properties.get("text", "") for hit in hits if hit])
        if not context:
            context = "[No context returned from TheBrain.]"

        host_prompt = f'Host: Using the following knowledge from TheBrain, answer this query: "{query}"'
        self.history.append({"name": "Host", "content": host_prompt})
        self.history.append({"name": "TheBrain", "content": f"CONTEXT:\n{context}"})

        drone = self.drones[drone_name]
        messages = [
            {"role": "system", "content": drone.persona},
            {
                "role": "user",
                "content": (
                    f'Using the following knowledge from TheBrain, answer this query: "{query}"\n\n'
                    f"CONTEXT:\n{context}"
                ),
            },
        ]

        reply = self._stream_and_display(
            name=drone_name,
            model=drone.model,
            messages=messages,
            stream=stream,
            options=drone.options,
        )
        return reply

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _get_weaviate_client(self) -> weaviate.WeaviateClient:
        if self._weaviate_client is None:
            try:
                self._weaviate_client = weaviate.connect_to_local(grpc_port=50051, http_host="localhost", http_port=8080)
            except Exception as exc:  # pragma: no cover - relies on external service
                raise ConnectionError("Failed to connect to Weaviate. Is it running? (`make awaken_hive`)") from exc
        return self._weaviate_client

    def _execute_code_blocks(self, prompt: str) -> str:
        fenced_block_pattern = re.compile(r"```(python|sh)\n(.*?)```", re.DOTALL)
        matches = list(fenced_block_pattern.finditer(prompt))
        if not matches:
            return ""

        results: List[str] = []
        for match in matches:
            lang, code = match.groups()
            header = f"--- EXECUTING {lang.upper()} ---"
            try:
                if lang == "python":
                    fname = f"script_{uuid.uuid4().hex[:8]}.py"
                    fpath = os.path.join(self.workspace_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as fh:
                        fh.write(code.strip())
                    try:
                        exit_code, output = lab.run_python_script_in_lab(fpath)
                    finally:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                else:
                    exit_code, output = lab.run_in_lab(code.strip())
                body = f"EXIT CODE: {exit_code}\n\nOUTPUT:\n{output}"
            except Exception as exc:  # pragma: no cover - theLab failures depend on environment
                body = f"EXECUTION FAILED:\n{exc}"
            results.append(f"{header}\n{body}\n--- END ---")
        return "\n\n" + "\n\n".join(results)

    def _stream_and_display(
        self,
        name: str,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        options: Optional[Dict],
    ) -> str:
        handle = self._display_markdown(self.to_markdown())
        reply = ""
        opts = dict(options or {})

        if stream:
            for part in ollama.chat(model=model, messages=messages, stream=True, options=opts):
                delta = part.get("message", {}).get("content", "")
                reply += delta
                if _HAS_IPY:
                    temp_md = self.to_markdown() + f"\n\n**{name}:**\n\n{reply} ..."
                    self._display_markdown(temp_md, handle)
        else:
            response = ollama.chat(model=model, messages=messages, options=opts)
            reply = response["message"]["content"]

        self.history.append({"name": name, "content": reply})
        self._display_markdown(self.to_markdown(), handle)
        return reply


__all__ = ["HiveMind", "Drone"]

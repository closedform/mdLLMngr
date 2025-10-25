import json, os, tempfile, subprocess, shutil, nbformat

def extract_ipynb_text(path: str) -> str:
    nb = nbformat.read(path, as_version=4)
    out = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            out.append(cell.source or "")
        elif cell.cell_type == "code":
            out.append("```code\n" + (cell.source or "") + "\n```")
    return "\n\n".join(out).strip()

def try_export_mathematica_nb_to_md(path: str) -> str | None:
    if shutil.which("wolframscript") is None:
        return None
    out_md = tempfile.mktemp(suffix=".md")
    code = f'Export["{out_md}", Import["{path}"], "Markdown"]'
    try:
        subprocess.run(["wolframscript", "-code", code], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_md if os.path.exists(out_md) else None
    except Exception:
        return None

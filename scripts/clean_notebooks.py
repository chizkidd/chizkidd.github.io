#!/usr/bin/env python3
"""Strip Colab/widget metadata that breaks nbconvert and GitHub rendering.

Edits notebooks in place in the working tree (the build copy); nothing is committed.
"""
import glob
import nbformat

WIDGET_MIMES = (
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
)
DROP_META = ("widgets", "colab", "accelerator")

changed = 0
for path in glob.glob("**/*.ipynb", recursive=True):
    if ".ipynb_checkpoints" in path or path.startswith((".site-tools/", "public/")):
        continue
    nb = nbformat.read(path, as_version=4)
    dirty = False
    for key in DROP_META:
        if key in nb.metadata:
            del nb.metadata[key]
            dirty = True
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.lower()
        if cell.get("outputs") and ("notebook_login" in src or "%%html" in src):
            cell["outputs"] = []
            dirty = True
        kept = [o for o in cell.get("outputs", [])
                if not any(m in o.get("data", {}) for m in WIDGET_MIMES)]
        if len(kept) != len(cell.get("outputs", [])):
            cell["outputs"] = kept
            dirty = True
    if dirty:
        nbformat.write(nb, path)
        changed += 1
        print(f"cleaned {path}")
print(f"{changed} notebook(s) cleaned")

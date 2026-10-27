"""Runs notebooks/network_performance_so_far.ipynb outside Jupyter and writes notebooks/network_performance_report.html.

Every code cell is executed in order (stdout kept as text, every figure embedded as a PNG), every markdown cell is kept, so the page is
the whole notebook with its results and opens in any browser, whatever the Jupyter / matplotlib backend. Run from anywhere:
    python notebooks/render_performance_report.py          (use the kernel environment of the notebook: needs numpy, matplotlib,
                                                            nibabel, scipy, torch and IPython)
"""

import base64
import contextlib
import html
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import IPython.display as ipd  # noqa: E402

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "network_performance_so_far.ipynb"
OUT = HERE / "network_performance_report.html"
os.chdir(HERE)

figures = []


def capture_display(obj, *args, **kwargs):
    if isinstance(obj, ipd.Image):
        figures.append(base64.b64encode(obj.data).decode())


ipd.display = capture_display


def markdown_to_html(text):
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables"])
    except ImportError:
        out = []
        for line in text.splitlines():
            m = re.match(r"^(#{1,4})\s+(.*)", line)
            out.append(f"<h{len(m.group(1)) + 1}>{html.escape(m.group(2))}</h{len(m.group(1)) + 1}>" if m else f"<p>{html.escape(line)}</p>")
        return "\n".join(out)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf8"))
namespace = {"__name__": "__main__"}
parts = []
failed = False
for number, cell in enumerate(notebook["cells"]):
    source = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        parts.append(f'<div class="md">{markdown_to_html(source)}</div>')
        continue
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    figures.clear()
    buffer = io.StringIO()
    error = ""
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, f"cell{number}", "exec"), namespace)
    except Exception:
        error = traceback.format_exc()
        failed = True
    text = buffer.getvalue()
    block = ""
    if text.strip():
        block += f"<pre>{html.escape(text)}</pre>"
    block += "".join(f'<img src="data:image/png;base64,{b}">' for b in figures)
    if error:
        block += f'<pre class="err">{html.escape(error)}</pre>'
    parts.append(f'<div class="out">{block}</div>')
    print(f"cell {number}: {len(figures)} figure(s)" + (" FAILED" if error else ""), flush=True)

style = """body{font-family:Segoe UI,Arial,sans-serif;max-width:1250px;margin:20px auto;padding:0 16px;background:#fff;color:#111}
pre{background:#f4f4f4;padding:8px;overflow-x:auto;font-size:12px}img{max-width:100%;display:block;margin:8px 0;background:#000}
.err{background:#fee;color:#900}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px}h2{margin-top:32px}"""
OUT.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Network performance so far</title><style>{style}</style></head>"
               f"<body>{''.join(parts)}</body></html>", encoding="utf8")
print("wrote", OUT, "(some cells FAILED, see the red blocks)" if failed else "")

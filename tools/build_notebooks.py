#!/usr/bin/env python
"""Turn the ``.py`` example sources into executed ``.ipynb`` notebooks.

Every example in ``docs/examples`` is authored as a jupytext *percent-format*
Python script. That file is the canonical source: it is a plain script you can
run with ``python``, it diffs cleanly in git, and it is what you edit.

This tool converts each one to a notebook and, for the notebooks listed in
``EXECUTE``, runs it and commits the outputs. The expensive N-body examples are
executed here rather than during the Sphinx build, so CI stays fast and does not
need a working galpy + compiler toolchain.

Usage
-----
    python tools/build_notebooks.py                 # convert all, execute the slow ones
    python tools/build_notebooks.py --no-execute    # convert only
    python tools/build_notebooks.py 04-tidal-stream # just one
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "docs" / "examples"

#: Notebooks whose outputs are committed. Anything not listed is executed by
#: myst-nb at build time instead (see ``nb_execution_excludepatterns`` in conf.py).
EXECUTE = {
    "04-tidal-stream",
    "05-tidal-stripping",
    "06-stream-track",
    "07-stream-gaps",
}

#: Cell timeout for the executed notebooks, in seconds.
TIMEOUT = 3600


def strip_progress_bars(path: Path) -> None:
    """Collapse tqdm's carriage-return spam to its final line.

    tqdm repaints by emitting ``\\r``-separated frames. nbconvert keeps every
    frame, which turned one 30 000-step run into a 15 MB notebook. Keeping only
    the last frame preserves what the reader sees (the finished bar) and drops
    the rest.
    """
    import nbformat

    nb = nbformat.read(path, as_version=4)
    saved = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "\r" in out.get("text", ""):
                before = len(out["text"])
                # Keep the last frame of each line-group.
                lines = [seg.split("\r")[-1] for seg in out["text"].split("\n")]
                out["text"] = "\n".join(l for l in lines if l.strip())
                saved += before - len(out["text"])
            kept.append(out)
        cell["outputs"] = kept
    if saved:
        nbformat.write(nb, path)
        print(f"      stripped {saved/1e6:.1f} MB of progress-bar frames")


def add_colab_badge(path: Path, user: str, repo: str) -> None:
    """Ensure the first markdown cell carries an 'Open in Colab' badge."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    if not nb.cells or nb.cells[0].cell_type != "markdown":
        return
    src = nb.cells[0].source
    url = (
        f"https://colab.research.google.com/github/{user}/{repo}/blob/main/"
        f"docs/examples/{path.name}"
    )
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"
    if "colab-badge.svg" in src:
        src = re.sub(r"\[!\[Open In Colab\][^\n]*", badge, src)
    else:
        lines = src.split("\n")
        at = 1 if lines and lines[0].startswith("#") else 0
        lines.insert(at, "\n" + badge)
        src = "\n".join(lines)
    nb.cells[0].source = src
    nbformat.write(nb, path)


def build(stem: str, execute: bool, user: str, repo: str) -> bool:
    src = EXAMPLES / f"{stem}.py"
    out = EXAMPLES / f"{stem}.ipynb"
    if not src.exists():
        print(f"  !! missing {src}")
        return False

    print(f"  -> {stem}")
    # A percent-format file must be valid Python: markdown lines that lose their
    # leading "# " parse as code and fail confusingly deep in the kernel.
    import ast
    try:
        ast.parse(src.read_text())
    except SyntaxError as exc:
        print(f"  !! {stem}.py is not valid Python (line {exc.lineno}): {exc.msg}")
        print(f"     {(exc.text or '').strip()[:90]}")
        return False

    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "notebook", "-o", str(out), str(src)],
        check=True,
        cwd=ROOT,
    )

    if execute and stem in EXECUTE:
        print("     executing (this is one of the slow ones) ...")
        r = subprocess.run(
            [
                sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", f"--ExecutePreprocessor.timeout={TIMEOUT}", str(out),
            ],
            cwd=EXAMPLES,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr[-3000:])
            print(f"  !! {stem} FAILED to execute")
            return False
        strip_progress_bars(out)

    add_colab_badge(out, user, repo)
    print(f"     {out.stat().st_size/1e6:.1f} MB")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems", nargs="*", help="example stems, e.g. 04-tidal-stream")
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--user", default="sgpfaff")
    ap.add_argument("--repo", default="tambora-docs")
    a = ap.parse_args()

    stems = a.stems or sorted(p.stem for p in EXAMPLES.glob("*.py"))
    if not stems:
        print("no .py examples found")
        return 1

    print(f"building {len(stems)} notebook(s)")
    failed = [s for s in stems if not build(s, not a.no_execute, a.user, a.repo)]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nall notebooks built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

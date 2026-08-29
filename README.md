# tambora documentation

Documentation for [tambora](https://pypi.org/project/tambora/) — a publicly maintained, modular N-body Python package designed for small galactic dynamics tasks with user-friendliness and extensibility in mind.

**Live site:** https://sgpfaff.github.io/tambora-docs/

tambora is GPL-2.0-or-later (it bundles falcON); see the
[licence and citing page](https://sgpfaff.github.io/tambora-docs/about.html).

Built with [Sphinx](https://www.sphinx-doc.org/) + [Furo](https://pradyunsg.me/furo/),
against the released `tambora==0.1.0a1`. The API reference is generated from the
installed package's docstrings, so it cannot drift from the code.

## Layout

```
docs/
  index.md              landing page
  installation.md       install, optional galpy, verification
  quickstart.md         first simulation in ten lines
  guide/                concepts, units, self-gravity, forces, hooks, ...
  examples/             notebooks -- *.py is the source, *.ipynb is generated
  api/                  autosummary-generated API reference
  _static/              figures, animations, shared matplotlib style
tools/
  build_notebooks.py    .py -> executed .ipynb
```

## Building locally

```bash
pip install --pre -r requirements.txt
make html
make serve          # http://localhost:8000
```

## Examples

Each example is authored as a **jupytext percent-format `.py`** — a plain,
runnable script — and the `.ipynb` is generated from it:

```bash
python docs/examples/04-tidal-stream.py     # run as a script
python tools/build_notebooks.py             # regenerate all notebooks
python tools/build_notebooks.py 04-tidal-stream
```

The N-body-heavy notebooks (04–07) ship with committed outputs because they take
minutes to run; the rest execute during the Sphinx build so they cannot go stale.
Keep `EXECUTE` in `tools/build_notebooks.py` in sync with
`nb_execution_excludepatterns` in `docs/conf.py`.

## Deployment

Pushing to `main` triggers `.github/workflows/docs.yml`, which builds the site
and publishes it to GitHub Pages. Enable Pages once, with **Source: GitHub
Actions**.

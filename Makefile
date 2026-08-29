# tambora docs

.PHONY: html clean notebooks serve linkcheck

html:                ## build the HTML docs into docs/_build/html
	sphinx-build -b html docs docs/_build/html

clean:               ## remove build products and the notebook execution cache
	rm -rf docs/_build docs/api/generated docs/jupyter_execute .jupyter_cache

notebooks:           ## regenerate .ipynb from the .py sources in docs/examples
	python tools/build_notebooks.py

serve: html          ## build, then serve locally at http://localhost:8000
	python -m http.server -d docs/_build/html 8000

linkcheck:
	sphinx-build -b linkcheck docs docs/_build/linkcheck

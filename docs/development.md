# Development

Python 3.10 through 3.13 is declared. The recorded environments are Python
3.11.2 with NumPy 1.24.2/SciPy 1.10.1 and Python 3.13.5 with NumPy 2.3.5/SciPy
1.16.3. Numerical changes must be compared in both environments when possible.

Install only the extras needed by the task:

```bash
python -m pip install -e '.[test,export,numba,cdp]'
```

The checkout used for the 2026-07-15 baseline has no `python` command in PATH;
use `python3` or an explicit virtual-environment interpreter. Do not commit
virtual environments, caches, generated outputs, or build trees.

Run the compact regression manifest with:

```bash
PYTHONPATH=src python3 scripts/regression_manifest.py
```

Update its golden file only after classifying every numerical difference.

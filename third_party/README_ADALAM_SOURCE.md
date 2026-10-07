# Bundled AdaLAM source provenance

The directory `third_party/adalam/` was copied from the user-supplied archive `adalam.zip`. The benchmark imports this directory directly under a private package name and does not fall back to Kornia or an installed `adalam` package.

Core files used by the benchmark:

- `__init__.py`
- `adalam.py`
- `core.py`
- `ransac.py`
- `utils.py`

The source files themselves were not modified. Integration changes are confined to `matching_comparison_all.py`.

# Verification sources

These editable Python sources were extracted from the repository's historical
`verification.zip`, then corrected. Use this directory for new runs; the ZIP is
retained unchanged as a historical snapshot and does not include these fixes.

Configure input/output paths in `main_verify.py`, then run
`python verification/main_verify.py` from the repository root. The DNS MATLAB
inputs are external. HDF5 MATLAB files require the optional `mat73` package;
older MATLAB files can be read by SciPy.

The statistics fix retains the old eligibility thresholds and equal weighting
of samples. It removes arbitrary file truncation and saves final statistics.
The plotting comparison selects each trace's zero-offset column from metadata.

This verification driver remains a separate implementation. Passing its checks
does not validate the top-level production pipeline or establish sampling
convergence. These sources and their tests run without the production
repair PR or Pyrcel. If the separate production fixes are installed, run their
regression suite as well.


## Install and test independently

From the repository root:

```bash
python -m pip install -r verification/requirements-dev.txt
python -m pytest -q verification/tests
```

These four regression tests use synthetic netCDF data. They do not need the
external DNS MATLAB inputs, the production modules, or Pyrcel. Running the full
driver still requires prepared DNS data; HDF5 MATLAB input also needs `mat73`.

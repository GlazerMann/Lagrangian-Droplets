# Particle-based model of turbulent microphysics in the MTU Pi Chamber

[![DOI](https://zenodo.org/badge/542801260.svg)](https://zenodo.org/badge/latestdoi/542801260)

The model follows LES-driven parcel histories and calculates aerosol activation
and droplet growth under fluctuating or uniform supersaturation.

Reference: Anderson, J.C., Beeler, P., Ovchinnikov, M., Cantrell, W., Krueger, S.,
Shaw, R.A., Yang, F. and Fierce, L., 2022. *Enhancements in cloud condensation
nuclei concentrations from turbulent fluctuations in supersaturation.*
arXiv:2210.15766.

## Install and test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The requirements constrain Pyrcel to the legacy 1.x thermodynamic API. Pin a
complete environment for scientific production runs. Focused tests use synthetic
netCDF fields and require no large LES/DNS download. On hosts where Pyrcel's
unrelated AOT compiler initializer fails, run
`LAGRANGIAN_TEST_LIGHT_IMPORTS=1 python -m pytest -q` to load the installed
constants/thermodynamics modules directly for isolated tests. That option does
not establish that normal application imports work on such a host and does not
substitute scientific equations.

## Run the stages

1. Obtain the external [LES/DNS inputs](https://portal.nersc.gov/project/m1657/LagrangianDroplets/).
   The parcel reader requires netCDF snapshots rather than raw SAM binary files.
   It expects `U`, `V`, `W`, `TABS`, `QV` shaped `(time,z,y,x)`, spatial coordinate
   vectors, time already in seconds, velocities in m/s, temperature in Kelvin,
   and `QV` in g/kg. Prepare external data to that contract; this repository
   does not include a SAM binary converter.
2. Configure and run `python main_parcel-traces.py`. A fresh ensemble starts at
   `num_start=0`; set it explicitly for restarts. Positive `stop_time` is a
   duration. Zero integrates until the last LES time. Parcel `time` stays
   absolute; `elapsed_time` records particle age.
3. Configure and run `python main_particle-traces.py` to generate Slurm scripts
   for both fluctuating and uniform runs. Set `submit_jobs=True` only when ready
   to submit. Account/module settings are inherited examples and must be adapted
   to the cluster. Alternatively call `run.main` directly. Batch jobs honor dry
   diameter, selected parcel count, and exposure duration.
4. Run `python main_make-figs.py --output-dir . --plot-dir figs`. Use
   `--parcel-ids 0` for a single trajectory. The corresponding supersaturation
   datasets must exist before plotting.

Output directories use decimal keys, e.g. `particle_traces/avgSS_0.03` for 3%.
Keys support a declared 1e-12 supersaturation grid, tolerate floating-point
roundoff, and reject unsupported finer precision rather than silently overwrite
another run. `supersaturation.json` records each directory's value.

## Trace units and migration

New schema version 2 stores `Dp` in meters and `Tp`/`T` in Kelvin, marked
`state_representation='physical'`, regardless of solver coordinates. Particle
time is elapsed seconds, `SS` is dimensionless (`0.01` means 1%), and mixing
ratio is kg vapor/kg dry air. `props` stores `Ddry`, `kappa`, `density`, `N`,
`Dp_crit`, and `SS_crit`.

Old pickles did not record their linear, logarithmic, or mixed representation.
They require an explicit format when read:

```python
import run
record = run.read_particle_trace(0, 'old_output/avgSS_000000',
                                 legacy_representation='lny')
```

The returned record is normalized to physical units. Select `linear` or `lnDp`
when those coordinates were used; do not guess from value signs. Old truncated
directory names are not searched automatically because they can be ambiguous
or already contain overwritten runs. Read an explicit old directory, migrate it
deliberately, or regenerate outputs. The plot command's
`--legacy-representation` handles old state coordinates after migration into the
requested decimal-key layout.

`unravel_particle_traces` rejects mismatched time grids. Perform any desired
resampling/population-selection policy explicitly before plotting.

## Scientific configuration and verification

See [SCIENTIFIC_NOTES.md](SCIENTIFIC_NOTES.md) for changed defaults, retained
choices, and validation work. Temperature noise is explicit and off by default.
`mean_definition='time'` uses elapsed-time weighting; `sample` permits deliberate
comparison with historical sample weighting. `run.main` accepts `P0` and `accom`
so pressure and accommodation can be configured explicitly.

`verification.zip` remains an unchanged historical snapshot. Editable DNS
verifier repairs are supplied as a separate, optional PR; this production
pipeline does not import or require that verifier. Neither its checks nor the
production regression suite replace full real-data validation.

## License

See [LICENSE.md](LICENSE.md) for GNU GPL v3 terms.


### Numerical and output validation

`run.main` and `run.one_particle` accept explicit `rtol`/`atol` solver controls.
The default relative tolerance is 1e-6. Default absolute tolerances reflect
solver coordinates: meters/Kelvin for linear states, logarithms for log states.
Successful solver status is insufficient: decoded physical states must also
be finite, positive, and at least as large as the dry core. Numerical tolerance
and sampling convergence should be checked for each scientific application.

Generated jobs require the requested parcel count to exist. A different stored
parcel selection requires a fresh output directory; stale files are rejected.
Failed runs leave an incomplete-run marker, and readers refuse that directory
until a successful rerun. Particle records are replaced atomically. Use one
writer per output directory and read only after it completes; concurrent runs
must use distinct directories.


Parcel regeneration also requires a consistent population. Use a new directory
when reducing it; old extra parcel IDs are rejected. Failed parcel runs retain
an incomplete marker and cannot be read. Recover by rerunning from num_start=0.
Parcel records are replaced atomically. As with particle output, use one writer
per directory and read only after completion.


Parcel flow solver controls are independent of output sampling:
`parcels.trajectories(..., rtol=1e-6, atol=1e-9)` uses relative tolerance and
absolute position tolerance in meters; atol may also be a three-component
vector. Check tolerance, source-grid, maximum-step, and sampling convergence
for each LES application. Fields must be collocated in (time,z,y,x) order;
coordinates/time/velocity/temperature/QV must already be meters, seconds, m/s,
Kelvin, and g/kg. Missing, masked, or nonphysical inputs are rejected.

# Scientific behavior and decisions for reviewers

These fixes change computed results and require new scientific validation.

| Area | Patch behavior | Further decision or validation |
| --- | --- | --- |
| Mean forcing | Time-weighted mean; centering remains per parcel | Decide whether a domain/ensemble mean is appropriate; `mean_definition='sample'` retains old sample weighting |
| Stochastic forcing | Off by default; explicit `temperature_noise_fraction` and optional seeded `rng`; paired solves share forcing | Establish any desired amplitude, spectrum, and covariance model |
| Positive initial supersaturation | Existing zero clamp for equilibrium initialization retained | Decide whether a different initial activation state is appropriate |
| Hygroscopicity | Existing example `kappa=1.0` retained | Decide whether a publication-reproduction configuration should use 1.2 for NaCl |
| Walls | Existing reflective coordinate lookup retained, made pure and robust; accepted coordinates are folded too | Validate the physical wall convention against the LES; this is not a switch to periodic walls |
| Environmental sampling | Includes LES snapshot times and a 0.1-second grid independently of velocity error control | Perform actual-field sampling and maximum-step convergence studies |
| Duration | Positive duration must fit the source history; zero in parcel/batch configuration means until end | A future truncation policy would need an explicit choice; current code raises |
| Pressure | Inverse LES helpers retain 100000 Pa; particle growth retains 101325 Pa, configurable as `P0` | Set consistent explicit pressure for each experiment; no chamber pressure was inferred |
| Thermal transfer | Mass interface, chain rules, arguments, and near-isothermal numerical limit repaired | Existing inner transfer law still does not use accommodation in mass transfer; changing that law requires a physical-model decision |

The focused tests do not replace a full real-data LES/DNS comparison or establish
which affected versions/code paths were used for the published study.


Solver defaults now use `rtol=1e-6` and state-coordinate-specific absolute
tolerances. `rtol` and `atol` remain explicitly configurable. This increases
numerical work relative to the old defaults; establish an accuracy/cost balance
through convergence testing rather than relying on the solver's success flag.


The chemistry/physics/software verification additionally enforces nonnegative
water vapor pressure below total pressure, rejects unmatched populations, and
requires positive kappa for the implemented equilibrium initializer. Kappa=0
is physically meaningful but requires an explicit dry/nonhygroscopic
initialization policy and is not implemented by these patches.

Settling uses the continuous Schiller-Naumann correction below and above Re=1,
instead of a discontinuous switch from pure Stokes drag. This changes finite-Re
settling speeds. Its established moderate-Re range (roughly Re<1000), spherical
particle assumptions, omitted buoyancy, and omitted slip correction still need
application-specific assessment; large-Re extrapolation is not validated.

The humid-air heat-capacity helper now uses a mass-fraction mixture with both
components in J/(kg*K), retaining the legacy dry-air fit and a room-temperature
constant vapor heat capacity (1860 J/(kg*K)). This helper enters the ventilation
correction when relative velocity is nonzero. The default thermal callback uses
zero relative velocity, so this repair does not change its ventilation factors.

The explicit-temperature and quasi-steady-temperature branches retain different
surface-tension and saturation-pressure parameterizations. They are not identical
physical models; representation-equivalence tests compare coordinates within
one branch. A constant-temperature explicit solve initialized with the Pyrcel
equilibrium can therefore have a small initial transient. Selecting a common
physical parameterization is a remaining model-owner decision.


Flow verification uses a coupled three-dimensional manufactured solution and
solid-body rotation in addition to the earlier one-axis motion checks. Parcel
integration now has configurable rtol=1e-6 and atol=1e-9 meters, independent of
the environmental sample interval and maximum step. In the rotation stress
case, the prior defaults produced about 0.43% radius drift. Tighter defaults
reduce that numerical error but do not establish a universal accuracy bound.

LES input requires collocated (time,z,y,x) fields on finite, strictly monotone
coordinate grids. Descending spatial axes are supported. Missing/masked values,
nonfinite data, negative vapor mixing ratio, and nonpositive Kelvin temperature
are rejected before output creation. No missing-data imputation, staggered-grid
reconstruction, coordinate-unit conversion, or new wall model is supplied.

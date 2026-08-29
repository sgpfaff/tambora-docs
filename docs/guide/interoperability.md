# Interoperability with galpy

tambora and galpy solve adjacent problems. galpy is the reference library for
Galactic potentials, orbit integration, distribution functions and coordinate
transforms; tambora runs the self-gravitating N-body piece that galpy does not.
Using them together is the intended workflow, and the seams are deliberately
narrow — a handful of conversion functions rather than a framework.

This page collects every touchpoint in one place.

## The unit boundary

Three unit systems are in play, and almost every interop bug is a confusion
between them.

| System | Length | Velocity |
| --- | --- | --- |
| tambora user | kpc | km/s |
| tambora internal | kpc | kpc/Gyr |
| galpy natural | `ro` = 8 kpc | `vo` = 220 km/s |

tambora's helpers hand you **user** units. galpy objects are in **natural** units
unless you turn physical output on. Do that, always:

```python
pot.turn_physical_on()
orb.turn_physical_on()
```

tambora checks and warns when a galpy object arrives without it, falling back to
galpy's defaults. See [Units](units.md).

## galpy → tambora

### Potentials become forces

```python
from galpy.potential import MWPotential2014

sim.add_external_pot(MWPotential2014)          # single, list, or composite
```

or explicitly, when you want to compose:

```python
from tambora.dynamics import ExternalGalpyPotential

sim.add_external_force(ExternalGalpyPotential(halo)
                       + ExternalGalpyPotential(disk))
```

Time-dependent wrapper potentials work; the current simulation time is passed
through at every step. See [External forces](external-forces.md) for the
supported list.

### Distribution functions become initial conditions

```python
from tambora.tools import galpydfsampler, galpysampler

pos, vel, mass = galpydfsampler(df, n=3000, m_total=1e10)
pos, vel, mass = galpysampler(pot, n=3000, m_total=1e10)
```

See [Initial conditions](initial-conditions.md).

### Orbits become particles

```python
from tambora.tools import galpy_orbit_to_tambora

pos, vel = galpy_orbit_to_tambora(orbit)     # kpc, km/s
```

### Tidal tensors

```python
from tambora.dynamics import TidalTensorGalpyForce

tt = TidalTensorGalpyForce(MWPotential2014, center=[12.0, 0.0, 0.0])
T = tt.tidal_tensor(pos)      # (3, 3), 1/Gyr^2
```

## tambora → galpy

There is no built-in exporter yet — `Sim.to_galpy_orbit()` raises
`NotImplementedError` — but the conversion is short, and worth having as a
snippet because several galpy tools want it.

galpy's phase-space layout is `(R, vR, vT, z, vz, phi)` in natural units:

```python
import numpy as np

RO, VO = 8.0, 220.0

def tambora_to_galpy_xv(pos_kpc, vel_kms, ro=RO, vo=VO):
    """(N,3) kpc + (N,3) km/s  ->  galpy (6,N) internal (R, vR, vT, z, vz, phi)."""
    x, y, z = pos_kpc.T
    vx, vy, vz = vel_kms.T
    R = np.hypot(x, y)
    vR = (x * vx + y * vy) / R
    vT = (x * vy - y * vx) / R
    return np.vstack([R / ro, vR / vo, vT / vo, z / ro, vz / vo, np.arctan2(y, x)])


xv = tambora_to_galpy_xv(sim.gc.pos(t=-1), sim.gc.vel(t=-1))
```

To build galpy `Orbit` objects instead — for action calculations, or to integrate
particles onwards in galpy — transpose into galpy's per-orbit ordering and attach
units:

```python
import astropy.units as u
from galpy.orbit import Orbit

p, v = sim.gc.pos(t=-1), sim.gc.vel(t=-1)
R = np.hypot(p[:, 0], p[:, 1])
vR = (p[:, 0]*v[:, 0] + p[:, 1]*v[:, 1]) / R
vT = (p[:, 0]*v[:, 1] - p[:, 1]*v[:, 0]) / R

orbits = Orbit(
    np.c_[R, vR, vT, p[:, 2], v[:, 2], np.degrees(np.arctan2(p[:, 1], p[:, 0]))],
    ro=8.0, vo=220.0,
)
orbits.turn_physical_on()
```

From there you get sky coordinates (`orbits.ra()`, `orbits.dec()`,
`orbits.pmra()`, `orbits.vlos()`), actions, and everything else galpy computes
for orbits.

## Fitting a stream track

The most substantial join. galpy's {class}`galpy.df.StreamTrack` fits a smooth
one-dimensional track through stream particles and exposes observables and a
covariance along it — exactly what you need to compare an N-body stream with a
survey.

```python
from galpy.df import StreamTrack

trk = StreamTrack.from_particles(
    xv[:, arm_indices],   # (6, N) galpy internal, ONE arm only
    cart_internal,        # (M, 6) dense progenitor track
    t_grid / TU,          # (M,) times in galpy units
    arm_sign=+1,          # +1 leading, -1 trailing
    order=2,              # fit mean AND covariance
    prog_orbit=prog, ro=8.0, vo=220.0, solarmotion=[-11.1, 12.24, 7.25],
)
```

Three things will bite you, all covered with worked code in the
[stream track notebook](../examples/06-stream-track.ipynb):

1. **`Orbit.integrate` treats `ts[0]` as the initial-condition time.** Passing
   `linspace(-T, T, n)` puts the progenitor at the *start* of the grid, not the
   middle. Integrate backwards and forwards separately and stitch.
2. **The time span must match the stream.** Too short and particles pile up at
   the grid edges; too long and the orbit wraps in azimuth, the projection goes
   degenerate, and the fitted track diverges.
3. **Fit each arm to its own particles.** Passing the whole stream to both calls
   makes the two tracks converge on the same half of it.

### A galpy bug worth knowing

`StreamTrack` accepts the named solar-motion presets in its docstring, but
`_get_vsun_kms` does `np.asarray(self._solarmotion, dtype=float)`, so a string
raises:

```text
ValueError: could not convert string to float: 'schoenrich'
```

Pass the numbers instead — `solarmotion=[-11.1, 12.24, 7.25]` is Schönrich et al.

## Analytic perturbation theory

galpy's impulse-approximation functions take stream particles directly, so you
can check an N-body result against theory:

```python
from galpy.df import impulse_deltav_plummer_curvedstream

dv = impulse_deltav_plummer_curvedstream(
    v / VO, x / RO, b / RO, w / VO, x0 / RO, v0 / VO, GM, rs / RO
) * VO       # -> km/s
```

`GM` is in natural units: `M / mass_in_msol(vo, ro)`.

:::{important}
The analytic formulae assume a **Plummer** perturber. To match, run tambora with
`kernel=0` *and* remember that `eps` is twice the Plummer scale radius — pass
`eps = 2 * rs`. Getting either wrong changes the peak kick by tens of percent.
See [Reliable N-body simulations](reliable-nbody.md#what-eps-actually-means) and
the [stream gaps notebook](../examples/07-stream-gaps.ipynb).
:::

## Running without galpy

tambora imports and runs fine without it. You lose external potentials, the
sampling helpers and the tidal-tensor force; you keep self-gravity, integration,
hooks and all the accessors. The galpy-backed names stay importable and raise a
pointed `ImportError` only when called.

## Related

- [External forces](external-forces.md)
- [Initial conditions](initial-conditions.md)
- [Stream tracks and observables](../examples/06-stream-track.ipynb)

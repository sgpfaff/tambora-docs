# Units

Galactic dynamics has no natural unit system, so every code invents one. The
tension is always the same: the units that make the equations of motion cheap to
integrate are not the units anyone wants to think in. Nobody reports a velocity
dispersion in kpc/Gyr, and no integrator wants a factor of 1.0227 in its inner
loop.

tambora resolves this by keeping both, and converting at a single, well-defined
boundary. You speak kpc, km/s, M☉ and Gyr — the units you would use in a paper.
The integrator works in a system where $G$ is a plain number and no conversion
appears in the hot loop. Every accessor translates on the way out, so the
internal representation never reaches your analysis.

That works so reliably that you can usually forget it exists. The exception is
hooks, which run *inside* the loop and therefore see internal units. That single
exception is the only unit bug tambora can hand you, and it is worth knowing
about before you hit it.

## The two systems

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} User units — what you pass in and get back
| Quantity | Unit |
| --- | --- |
| length | kpc |
| velocity | **km/s** |
| mass | M☉ |
| time | Gyr |
| acceleration | km/s/Gyr |
| potential (specific) | (km/s)² |
| angular velocity | km/s/kpc |
| angle | rad |
:::

:::{grid-item-card} Internal units — what the integrator uses
| Quantity | Unit |
| --- | --- |
| length | kpc |
| velocity | **kpc/Gyr** |
| mass | M☉ |
| time | Gyr |
| acceleration | kpc/Gyr² |
| potential (specific) | (kpc/Gyr)² |
| angular velocity | 1/Gyr |
| angle | rad |
:::

::::

Only **velocity** and things derived from it differ. The conversion is a single
constant:

$$ 1\ \mathrm{km/s} = 1.022712\ \mathrm{kpc/Gyr} $$

Internal units are what they are because with lengths in kpc, masses in M☉ and
times in Gyr, $G = 4.4985\times10^{-6}$ and the equations of motion need no
conversion factors in the inner loop.

## Where the conversion happens

Every accessor on `Sim` and `Component` is wrapped by
{func}`~tambora.tools.util.units.unit_handler`, which multiplies the internal
result by the right factor on the way out. So:

```python
sim.vel(t=-1)        # km/s   -- converted for you
sim.KE(t=-1)         # Msun (km/s)^2
sim.self_gravity(t=-1)   # km/s/Gyr
```

Likewise `add_particles` converts your km/s velocities *in* at the boundary.

### Escaping the conversion

Every wrapped accessor takes a hidden `return_internal` keyword:

```python
sim.vel(t=-1)                        # km/s
sim.vel(t=-1, return_internal=True)  # kpc/Gyr
```

You rarely want this. The one place it is genuinely useful is when you are
feeding a value straight back into something that expects internal units — for
example constructing a force by hand.

## The one place it will bite you: hooks

:::{warning}
{class}`~tambora.dynamics.integration.StepState` — the object your hooks
receive — returns **internal units**, deliberately and without conversion.
:::

Hooks run inside the integrator's hot loop, potentially every single step.
Converting arrays there would be pure waste for the majority of hooks that only
compare a quantity against itself. So `StepState` hands you the raw arrays.

This means a velocity dispersion accumulated by
{class}`~tambora.dynamics.hooks.BoundednessHook` is in kpc/Gyr:

```python
from tambora.tools.util.units import KPCGYR_TO_KMS

sigma_kms = np.asarray(bh.dispersion) * KPCGYR_TO_KMS   # kpc/Gyr -> km/s
```

Similarly, positions captured by `capture_transitions=('vel',)` are internal.
Positions, masses and times need no conversion — they are the same in both
systems.

Quantities that are *dimensionless ratios* — `ConservationMonitor.drift`, the
bound fraction from `BoundednessHook.fraction()` — are unit-free and need
nothing.

## Constants

Import them from `tambora.tools.util.units` (also re-exported from
`tambora.tools`):

```python
from tambora.tools.util.units import (
    G_INTERNAL,      # 4.4985e-06  kpc^3 Msun^-1 Gyr^-2
    G_KPC_KMS,       # 4.3009e-06  kpc (km/s)^2 Msun^-1
    KMS_TO_KPCGYR,   # 1.022712
    KPCGYR_TO_KMS,   # 0.977792
)
```

`G_KPC_KMS` is the one to use for back-of-the-envelope checks in user units —
for example a tidal radius:

```python
r_t = r_peri * (m_sat / (3 * M_enc))**(1/3)     # no G needed
v_circ = np.sqrt(G_KPC_KMS * M_enc / r)         # km/s
```

## galpy's units are a third system

This catches everyone once. A bare galpy object is in **natural units**, where
lengths are in units of `ro` (8 kpc) and velocities in units of `vo`
(220 km/s):

```python
from galpy.orbit import Orbit

Orbit([12.0, 0.0, 1.05, 0.0, 40.0, 0.0])   # R = 12 * 8 kpc = 96 kpc!
```

Always attach astropy units, and call `turn_physical_on()`:

```python
import astropy.units as u

o = Orbit([12.0*u.kpc, 0.0*u.km/u.s, 140.0*u.km/u.s,
           0.0*u.kpc, 20.0*u.km/u.s, 0.0*u.deg])
o.turn_physical_on()
```

tambora's galpy bridge checks this for you and warns when a potential arrives
with physical output disabled:

```text
UserWarning: The provided galpy object does not have physical units explicitly
set. Using galpy defaults (ro=8.0 kpc, vo=220.0 km/s). Set them explicitly with
turn_physical_on(ro=..., vo=...)
```

That warning is benign if you *intended* the defaults, and a real bug otherwise.
The sampling helpers ({func}`~tambora.tools.mkPlummer_galpy` and friends) handle
this internally and always hand you kpc and km/s.

## Quick reference

| Where you are | Velocity unit |
| --- | --- |
| Arguments to `add_particles` | km/s |
| Return of `sim.vel()`, `sim.cluster.vr()`, … | km/s |
| Return of `..., return_internal=True` | kpc/Gyr |
| Inside a hook (`StepState.vel()`) | kpc/Gyr |
| `BoundednessHook.dispersion`, `com_vel` | kpc/Gyr |
| Return of `mkPlummer_galpy` etc. | km/s |
| A bare galpy object | `vo` = 220 km/s |

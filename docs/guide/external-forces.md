# External forces

An external force is anything that pushes on your particles without being pushed
back: a galaxy the satellite orbits in, a bar, a tidal field. tambora keeps these
separate from self-gravity because they are separate physics with different
costs, and because you frequently want one without the other.

The design has two ideas worth knowing. First, **forces are objects that compose
with `+`**, so building a galaxy out of a bulge, a disk and a halo is addition
rather than configuration. Second, the type of the result carries meaning:
adding conservative forces gives you something that still has a potential, and
mixing in a non-conservative one gives you something that deliberately does not,
so a half-defined energy is an error rather than a plausible wrong number.

## galpy potentials

The common case is one call:

```python
from galpy.potential import MWPotential2014

sim.add_external_pot(MWPotential2014)
```

That wraps the potential in an {class}`~tambora.dynamics.ExternalGalpyPotential`
and adds it.

:::{important}
`add_external_pot` requires a galpy **`Potential` instance**. On galpy ≥ 1.11,
combining potentials with `+` gives a `CompositePotential`, which is one — and
that is why `MWPotential2014` works, since it is itself a composite. A plain
Python **list** is not, and raises:

```text
TypeError: External potential must be a galpy Potential object.
```

So combine with `+`:

```python
sim.add_external_pot(bulge + disk + halo)      # works
sim.add_external_pot([bulge, disk, halo])      # TypeError
```
:::

The potential must have physical units enabled. If it does not, tambora warns and
falls back to galpy's defaults (`ro=8` kpc, `vo=220` km/s):

```text
UserWarning: The provided galpy object does not have physical units explicitly
set. Using galpy defaults (ro=8.0 kpc, vo=220.0 km/s).
```

That is benign if you meant the defaults and a real bug otherwise. Set them
explicitly:

```python
from galpy.potential import NFWPotential
import astropy.units as u

halo = NFWPotential(amp=8e11 * u.Msun, a=16 * u.kpc, ro=8.0, vo=220.0)
halo.turn_physical_on()
```

### Which potentials work

tambora vectorises the potential evaluation where galpy supports it and loops
per-particle where it does not. Both give correct answers; the vectorised path is
much faster.

```python
from tambora.tools.util._galpy_bridge import (
    VECTORIZED_POTENTIALS, UNVECTORIZED_POTENTIALS, ALL_SUPPORTED_WRAPPERS,
)
```

Vectorised: most spherical profiles (Hernquist, NFW, Plummer, Jaffe, King,
Isochrone, Burkert, Dehnen, Einasto, power laws), the common axisymmetric ones
(Miyamoto–Nagai, Kuzmin, logarithmic halo, MN3 exponential disk, ring), and
triaxial cases including `DehnenBarPotential` and `SpiralArmsPotential`. On
galpy > 1.11.2 the `EllipsoidalPotential` family is vectorised as well.

Looped: `DoubleExponentialDiskPotential`, `RazorThinExponentialDiskPotential`,
`FerrersPotential`, `HomogeneousSpherePotential`, `SphericalShellPotential`,
`SoftenedNeedleBarPotential`, `NullPotential`.

Wrappers — `DehnenSmoothWrapperPotential`, `SolidBodyRotationWrapperPotential`,
`CorotatingRotationWrapperPotential`, `GaussianAmplitudeWrapperPotential` and
friends — are supported, so growing bars and rotating patterns work. Anything
unsupported raises at the point you add it, not partway through a long run.

### Time dependence

tambora passes the current simulation time to the potential at every step, so
time-dependent wrappers behave correctly:

```python
from galpy.potential import DehnenBarPotential, DehnenSmoothWrapperPotential

bar = DehnenBarPotential(omegab=1.3, rb=0.5, Af=0.01)
grown = DehnenSmoothWrapperPotential(pot=bar, tform=-4.0, tsteady=2.0)
sim.add_external_pot(MWPotential2014 + grown)
```

Expect the energy monitor to show real drift here. A time-dependent potential
does work on the system, so total energy genuinely is not conserved; the number
is measuring physics, not integration error.

## Composing forces

```python
from tambora.dynamics import ExternalGalpyPotential

force = (ExternalGalpyPotential(bulge)
         + ExternalGalpyPotential(disk)
         + ExternalGalpyPotential(halo))
sim.add_external_force(force)
```

`+` returns a composite that satisfies the same interface, and nesting flattens,
so `(a + b) + c` and `a + (b + c)` are the same object.

The type of the result is the point:

- all members conservative → a conservative composite, with `.potential()`
- any member non-conservative → a plain composite, **without** `.potential()`

So `(halo + dynamical_friction).potential()` is an `AttributeError` rather than a
number that silently ignores the friction term.

## The tidal-tensor force

{class}`~tambora.dynamics.TidalTensorGalpyForce` expands the host potential to
second order about a chosen centre and keeps only the tidal part, discarding the
constant and linear terms that merely accelerate the system as a whole.

```python
from tambora.dynamics import TidalTensorGalpyForce

tt = TidalTensorGalpyForce(MWPotential2014, center=[12.0, 0.0, 0.0])
sim.add_external_force(tt)

T = tt.tidal_tensor(pos)      # (3, 3), internal units [1/Gyr^2]
```

If `center` is `None` the median particle position is used, so the tensor tracks
the system.

This is useful when you want the tidal deformation isolated from the bulk orbital
motion — the cluster stays in the middle of your box and you watch it strain —
and when you want to attribute mass loss to the smooth tide rather than to
higher-order terms. It is a *local* approximation, accurate while the system is
small compared with the scale over which the tidal field varies, so use it for a
bound progenitor and not for an extended stream. The
[stripping notebook](../examples/05-tidal-stripping.ipynb) measures where it
breaks down.

## Writing your own force

Subclass one of three abstract bases depending on what you can provide.

**Conservative and mass-independent** (a potential every particle feels the same
way) — subclass {class}`~tambora.dynamics.ExternalConservativeForce` and
implement `acc(pos, t)` and `potential(pos, t)`:

```python
import numpy as np
from tambora.dynamics import ExternalConservativeForce
from tambora.tools.util.units import G_INTERNAL


class PointMass(ExternalConservativeForce):
    """A fixed point mass at the origin, softened."""

    def __init__(self, mass, eps=0.0):
        self.mass, self.eps = mass, eps

    def acc(self, pos, t):
        r2 = np.sum(pos**2, axis=1) + self.eps**2
        return -G_INTERNAL * self.mass * pos / r2[:, None] ** 1.5

    def potential(self, pos, t):
        r2 = np.sum(pos**2, axis=1) + self.eps**2
        return -G_INTERNAL * self.mass / np.sqrt(r2)
```

**Velocity-dependent or otherwise non-conservative** — subclass
{class}`~tambora.dynamics.Force` and implement
`acc(pos, vel, mass, t)`. There is no `potential`, and composing it with a
conservative force correctly produces something without one:

```python
from tambora.dynamics import Force


class LinearDrag(Force):
    """A crude drag term. Not conservative, deliberately has no potential."""

    def __init__(self, rate):
        self.rate = rate

    def acc(self, pos, vel, mass, t):
        return -self.rate * vel
```

:::{warning}
Custom forces work in **internal** units: positions in kpc, velocities in
kpc/Gyr, accelerations in kpc/Gyr², potentials in (kpc/Gyr)². Use `G_INTERNAL`,
not `G_KPC_KMS`. Nothing converts for you at this level — see
[Units](units.md).
:::

Both then go in the same way:

```python
sim.add_external_force(PointMass(1e10, eps=0.1))
sim.add_external_force(ExternalGalpyPotential(halo) + LinearDrag(0.05))
```

`add_external_force` rejects self-gravity solvers with a pointed `TypeError` —
those belong in `run(method=...)`, not here.

## Reading the field back

After a run:

```python
sim.external_acc(t=-1)          # (N, 3)  km/s/Gyr
sim.external_ax(t=-1)           # components
sim.compute_external_pot(t=-1)  # (N,)    Msun (km/s)^2
```

`compute_external_pot` evaluates on demand rather than from a cache, so asking
for all snapshots at once loops and warns. Ask for the snapshots you need.

## Related

- [Self-gravity](self-gravity.md) — the other half of the force budget
- [Interoperability](interoperability.md) — the full galpy bridge
- [External potentials](../examples/03-external-potentials.ipynb) — worked example

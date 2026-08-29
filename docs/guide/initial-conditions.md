# Initial conditions

A tambora simulation begins with three arrays — positions in kpc, velocities in
km/s, masses in M☉ — and it does not care where they came from. That is a
deliberate boundary: sampling a distribution function is galpy's job, or your
job, and tambora's job starts once you have particles.

In practice you have four routes, in increasing order of effort: the built-in
profile helpers, any galpy distribution function, any galpy potential via an
Eddington inversion, or arrays you build yourself. All of them coexist in one
simulation.

## The built-in helpers

```python
from tambora.tools import mkPlummer_galpy, mkKing_galpy, mkNFW_galpy

pos, vel, mass = mkPlummer_galpy(m=1e6, b=0.1, n=3000)
pos, vel, mass = mkKing_galpy(m=1e6, n=3000, W0=6.0)
pos, vel, mass = mkNFW_galpy(m=1e9, n=3000)
```

| Function | Shape parameter | Notes |
| --- | --- | --- |
| {func}`~tambora.tools.mkPlummer_galpy` | `b`, scale radius [kpc] | isotropic; $r_{1/2}=1.305\,b$ |
| {func}`~tambora.tools.mkKing_galpy` | `W0`, central potential | also `rt`, `npts`; keep `W0` ≲ 200 |
| {func}`~tambora.tools.mkNFW_galpy` | via `nfw_kwargs` | also `nfw_df_kwargs` for the sampler |

All take `n`, `m_total`, `rmin`, `center_pos` and `center_vel`, and all require
galpy.

## Any galpy distribution function

```python
from galpy.df import isotropicHernquistdf
from galpy.potential import HernquistPotential
from tambora.tools import galpydfsampler
import astropy.units as u

pot = HernquistPotential(amp=2e10 * u.Msun, a=1.5 * u.kpc, ro=8.0, vo=220.0)
pot.turn_physical_on()
df = isotropicHernquistdf(pot=pot, ro=8.0, vo=220.0)

pos, vel, mass = galpydfsampler(df, n=3000, m_total=1e10)
```

Supported DF types are `isotropicHernquistdf`, `isotropicPlummerdf`,
`isotropicNFWdf`, `kingdf` and `eddingtondf` — the spherical ones. Anything else
raises with the list of what is accepted.

## Any spherical galpy potential

{func}`~tambora.tools.galpysampler` picks the DF for you: the matching isotropic
DF for Plummer, Hernquist and NFW, and an Eddington inversion otherwise.

```python
from tambora.tools import galpysampler

pos, vel, mass = galpysampler(pot, n=2000, m_total=1e10)
```

## From orbits

{func}`~tambora.tools.galpy_orbit_to_tambora` converts one orbit or many into
tambora arrays — the usual way to set up test particles.

```python
from galpy.orbit import Orbit
from tambora.tools import galpy_orbit_to_tambora

orbits = Orbit([[10.0, 0.0, 200.0, 0.0, 0.0, 0.0],
                [12.0, 0.0, 190.0, 0.5, 10.0, 30.0]], ro=8.0, vo=220.0)
orbits.turn_physical_on()
pos, vel = galpy_orbit_to_tambora(orbits)
```

It returns only positions and velocities; supply masses yourself.

## Placing a system on an orbit

`center_pos` and `center_vel` shift the sampled centre of mass **exactly**, after
sampling, so the realisation's Poisson noise does not perturb the orbit you
asked for.

```python
import astropy.units as u
from galpy.orbit import Orbit

o = Orbit([12.0*u.kpc, 0.0*u.km/u.s, 140.0*u.km/u.s,
           0.0*u.kpc, 20.0*u.km/u.s, 0.0*u.deg])
o.turn_physical_on()

pos, vel, mass = mkPlummer_galpy(
    m=3e4, b=0.008, n=5000,
    center_pos=[o.x(), o.y(), o.z()],
    center_vel=[o.vx(), o.vy(), o.vz()],
)
```

:::{warning}
**Attach astropy units to the orbit.** Without them galpy reads natural units, so
`Orbit([12.0, 0.0, 1.05, 0.0, 40.0, 0.0])` means $R = 12 \times 8 = 96$ kpc. The
error is silent and large.
:::

## Your own arrays

Nothing is special about the built-ins. Any `(N,3)`, `(N,3)`, `(N,)` triple in
kpc, km/s and M☉ works:

```python
import numpy as np

rng = np.random.default_rng(0)
n = 2000
u_ = rng.random(n) ** (1/3)                       # uniform density
cost = rng.uniform(-1, 1, n)
phi = rng.uniform(0, 2*np.pi, n)
sint = np.sqrt(1 - cost**2)
pos = 0.2 * u_[:, None] * np.c_[sint*np.cos(phi), sint*np.sin(phi), cost]
vel = np.zeros_like(pos)
mass = np.full(n, 1e6 / n)

sim.add_particles("blob", pos, vel, mass)
```

`add_particles` validates shapes and raises a message naming the mismatch, so
mistakes surface immediately rather than as a broken run.

## Several components

```python
sim.add_particles("host", host_pos, host_vel, host_mass)
sim.add_particles("satellite", sat_pos, sat_vel, sat_mass)
```

They are integrated together and feel each other's gravity, while staying
individually addressable as `sim.host` and `sim.satellite`. Give them different
softening if their densities differ:

```python
sim.run(..., eps={"host": 0.1, "satellite": 0.02})
```

## Check equilibrium before trusting it

A sampler produces equilibrium for the potential it assumed. Add softening,
truncate the profile, or embed the system in an external field, and it is no
longer exactly in equilibrium.

```python
T = sim.KE(t=0).sum()
U = 0.5 * sim.self_potential(t=0).sum()
print(f"-2T/U = {-2*T/U:.3f}")        # ~1 for an isolated system in equilibrium
```

A useful independent check is an analytic velocity scale. For a Plummer sphere
the central 1D dispersion is $\sigma^2 = GM/(6b)$:

```python
from tambora.tools.util.units import G_KPC_KMS
sigma = np.sqrt(G_KPC_KMS * m / (6 * b))     # km/s
```

If the measured dispersion is far off, suspect a unit problem in the sampling.

## Not yet available

`Sim.add_subhalos()` raises `NotImplementedError`. Sample subhalo particles and
add them as a normal component — or, if you only need the perturber's field and
not its internal structure, use a single massive particle with
`eps = 2 * r_s`, which is exactly a Plummer sphere under `kernel=0`. See the
[stream gaps notebook](../examples/07-stream-gaps.ipynb).

`Sim.tag()`, which would define a component from a boolean mask, is also not yet
implemented.

## Related

- [Initial conditions notebook](../examples/02-initial-conditions.ipynb)
- [Interoperability](interoperability.md) — the full galpy bridge
- [Units](units.md)

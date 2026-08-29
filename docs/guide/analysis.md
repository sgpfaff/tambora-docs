# Analysing results

After `run()`, the `Sim` holds every snapshot and answers questions about them.
The design goal is that the code you write to ask a question looks like the
question: `sim.sat.vr(t=2.0)` rather than a slice into a snapshot array with a
remembered index range and a unit conversion.

Everything on this page is available on both `Sim` (whole system) and
`Component` (one named set of particles), with the same names and signatures.

## The `t` argument

Every accessor takes `t`, and it means three different things depending on type:

| You pass | Meaning | Returns |
| --- | --- | --- |
| `int` — `t=0`, `t=-1` | snapshot **index** | one snapshot |
| `float` — `t=1.5` | snapshot **nearest** 1.5 Gyr | one snapshot |
| omitted / `...` | **all** snapshots | leading time axis |

```python
sim.pos(t=-1)     # (N, 3)          last snapshot
sim.pos(t=1.5)    # (N, 3)          nearest 1.5 Gyr
sim.pos()         # (n_snap, N, 3)  everything
```

`t=0` and `t=0.0` differ: the first is the first snapshot, the second is the
snapshot nearest zero. They coincide unless you started the run at `t0 != 0`.

Out-of-range integers raise `IndexError`; out-of-range floats raise
`ValueError`. Both name the valid range.

`sim.times` is the array of snapshot times, and `sim.mass` the particle masses.

## Positions and velocities

| Positions | Velocities |
| --- | --- |
| `pos` — (N, 3) kpc | `vel` — (N, 3) km/s |
| `x`, `y`, `z` — kpc | `vx`, `vy`, `vz` — km/s |
| `r` — spherical radius, kpc | `vr` — spherical radial velocity, km/s |
| `cylR` — cylindrical radius, kpc | `cylvR` — cylindrical radial velocity, km/s |
| `phi` — azimuth, rad | `vphi` — **angular** velocity, km/s/kpc |
| `theta` — polar angle, rad | `vtheta` — polar velocity, km/s |

Note that `vphi` is an angular velocity, $\omega_\phi = (x v_y - y v_x)/R^2$, not
a linear one. Multiply by `cylR` if you want $v_\phi$ in km/s.

```python
r = sim.gc.r(t=2.0)                       # kpc
vr = sim.gc.vr(t=2.0)                     # km/s
v_phi = sim.gc.vphi(t=2.0) * sim.gc.cylR(t=2.0)   # km/s
```

## Momentum and angular momentum

`p`, `px`, `py`, `pz` in M☉ km/s. `L`, `Lx`, `Ly`, `Lz` in M☉ kpc km/s, and they
take an optional origin:

```python
sim.sat.Lz(t=1.0)                                    # about the Galactic centre
sim.sat.L(t=1.0, center_pos=com, center_vel=com_v)   # about the satellite
```

`center_vel` is in **internal** units (kpc/Gyr) — the one place a raw internal
value appears in an accessor argument. Multiply a km/s value by
`KMS_TO_KPCGYR`.

For *specific* quantities, divide by mass — these are per-particle totals, not
per unit mass:

```python
Lz_specific = sim.gc.Lz(t=-1) / sim.gc.mass       # kpc km/s
```

## Energies

| Call | Is |
| --- | --- |
| `KE(t)` | kinetic energy per particle |
| `self_potential(t)` | self-gravitational potential *energy* per particle |
| `compute_external_pot(t)` | external potential energy per particle |
| `PE(t)` | `self_potential + compute_external_pot` |
| `energy(t)` | `KE + PE` |
| `system_energy(t)` | one number for the whole system |
| `dE(t)` | $\lvert(E - E_0)/E_0\rvert$, the relative drift |

All in M☉ (km/s)², i.e. energies rather than specific energies.

`system_energy` is not the sum of `energy` over particles. Pairwise potential
energy is shared between two particles, so summing per-particle values
double-counts it:

```python
sim.system_energy(t=-1)
# == KE.sum() + 0.5 * self_potential.sum() + external_pot.sum()
```

The factor of one half on the self term is the difference, and it is why
`sim.PE(t=-1).sum()` was `-2.4e7` while `system_energy` was `-6.2e6` in the
[quickstart](../quickstart.md).

(cached-versus-on-the-fly)=
## Cached versus on-the-fly

During a run tambora caches the self-gravity acceleration and potential at every
snapshot. Accessors read that cache, which is why `sim.PE()` over all snapshots
is instant.

You can override it — to recompute with different softening, or a different
solver:

```python
sim.PE(t=-1)                                   # cached (default after a run)
sim.PE(t=-1, method="direct", eps=0.005)       # recompute
```

The rules, enforced by a decorator so the failure modes are explicit:

- No `method`, run finished → use the cache.
- `method` given → recompute, and `use_cached` must not be `True`.
- `use_cached=True` before `run()` → error, there is no cache.
- `use_cached=False` with no `method` → error, nothing to compute with.

Recomputation is **one snapshot at a time**. `t=...` is rejected:

```text
TypeError: Cannot compute on-the-fly for all times. Please provide an integer
index or a float time for t. You will have to manually loop over snapshots.
```

If you disabled the caches with `cache_self_gravity_acc=False` /
`cache_self_gravity_pot=False` to save memory, every such call must specify a
method.

## Accelerations

```python
sim.self_gravity(t=-1)     # (N, 3) km/s/Gyr, from self-gravity
sim.self_ax(t=-1)          # components
sim.external_acc(t=-1)     # (N, 3) km/s/Gyr, from external forces
sim.external_ax(t=-1)
```

The self-gravity accessors take the same `method` / `use_cached` arguments as the
energies.

## Component views

`sim.<name>` returns a {class}`~tambora.simulation.Component` — a view, not a
copy. It carries `name` and `mass`, and every accessor above.

```python
for c in sim.components:
    print(c.name, len(c.mass), f"{c.mass.sum():.2e} Msun")
```

### The one argument components add

On a component, "self-gravity" is ambiguous, so the energy and acceleration
accessors take `include_all_components`:

```python
sim.sat.self_potential(t=1.0, include_all_components=True)   # in the total field
sim.sat.self_potential(t=1.0, include_all_components=False)  # isolated
```

`True` answers "how deep in the whole system's potential are these particles?".
`False` answers "are these particles bound to each other?" — the question you
want for a satellite's survival, and what
{class}`~tambora.dynamics.hooks.BoundednessHook` uses.

## Built-in diagnostic plots

Two quick-look plots, meant for checking rather than publishing:

```python
sim.plot_energy_diagnostic()                 # |dE/E0| vs t, log scale
sim.plot_energy_diagnostic(nsnap=50, filename="energy.png")
sim.plot_momentum_diagnostic()               # momentum conservation
```

Pass `filename` to save instead of show.

## Worked patterns

**Half-mass radius over time**

```python
r_half = np.array([np.median(sim.gc.r(t=i)) for i in range(len(sim.times))])
```

**Bound mass without a hook** — reusable after the fact:

```python
from tambora.dynamics.diagnostics import bound_mask

mask = bound_mask(sim.gc.pos(t=-1),
                  sim.gc.vel(t=-1) * 1.022712,   # km/s -> kpc/Gyr
                  sim.gc.mass, eps=0.002)
print(f"bound: {mask.sum()} / {len(mask)}")
```

`bound_mask` is the same function `BoundednessHook` calls, so results agree. Note
that it wants **internal** velocity units.

**Radial density profile**

```python
bins = np.logspace(-2.5, 0.5, 30)
counts, _ = np.histogram(sim.gc.r(t=-1), bins=bins)
shell = 4/3 * np.pi * np.diff(bins**3)
rho = counts * sim.gc.mass[0] / shell        # Msun / kpc^3
```

**Centre of mass track**

```python
com = np.array([sim.gc.pos(t=i).mean(0) for i in range(len(sim.times))])
```

Or get it for free during the run with
`BoundednessHook(..., track=("com",))`, which uses the *bound* particles only —
usually what you actually want for a disrupting system.

## Not yet implemented

`save()`, `load()`, `to_galpy_orbit()`, `tag()` and `add_subhalos()` raise
`NotImplementedError` in {{ tambora_version }}. To persist a run, pull the arrays
out yourself:

```python
np.savez_compressed("run.npz", pos=sim.pos(), vel=sim.vel(),
                    times=sim.times, mass=sim.mass)
```

To restart, feed the final state to a fresh `Sim` — see the
[stream gaps notebook](../examples/07-stream-gaps.ipynb).

## Related

- [Units](units.md) — what comes back in what
- [Hooks](hooks.md) — measuring during the run instead
- [Core concepts](concepts.md) — why the API is shaped this way

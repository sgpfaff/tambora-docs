# Core concepts

Most N-body codes hand you back arrays. You get positions with shape
`(n_snapshots, n_particles, 3)`, you keep a note somewhere of which index range
was the satellite, and every analysis script begins by reconstructing that
bookkeeping. The physics is the easy part; remembering that particles 5000–8000
were the disk is what actually costs you an afternoon.

tambora is built around the opposite premise: the simulation should remember
its own structure, so that asking a question of it reads like the question.
You name a component when you create it, and from then on `sim.sat.vr(t=2.0)`
means *the radial velocities of the satellite at 2 Gyr, in km/s* — no indices, no
unit conversion, no note to self.

Four ideas carry the whole API, and they are all in service of that. Once they
click, most of tambora is guessable without looking anything up.

## 1. One `Sim` per simulation

{class}`~tambora.simulation.Sim` holds everything: the particles, the forces, and
after {meth}`~tambora.simulation.Sim.run` the snapshots too. There is no separate
"snapshot" or "output" object to manage.

The lifecycle is strictly ordered:

```text
   Sim()
     |
     |  add_particles()      name your components
     |  add_external_pot()   attach forces
     |  add_hook()           attach measurements
     v
   run()
     |
     +--> accessors:     sim.pos(t), sim.gc.r(t), sim.energy(t), ...
     +--> hook results:  sim.monitor, and whatever your hooks accumulated
```

Setup must finish before `run()`. Calling `add_particles` or `add_hook`
afterwards raises `RuntimeError` — the particle arrays are already concatenated
and the snapshot arrays already allocated, so a late addition would silently
mean something different from what you wrote.

## 2. Particles live in named components

You never hand tambora one anonymous blob of particles. Every set gets a name:

```python
sim.add_particles("host", host_pos, host_vel, host_mass)
sim.add_particles("sat",  sat_pos,  sat_vel,  sat_mass)
```

Internally these are concatenated into flat `(N, 3)` arrays, and each name maps
to a `slice`. That is an implementation detail you can mostly forget, because
the name becomes an attribute:

```python
sim.sat.r(t=1.0)        # radii of satellite particles only
sim.host.KE(t=1.0)      # kinetic energy of host particles only
sim.r(t=1.0)            # radii of *everything*
```

`sim.sat` returns a {class}`~tambora.simulation.Component` — a lightweight view,
not a copy. It carries the same accessors as `Sim`, restricted to its slice.

You can also iterate:

```python
for c in sim.components:
    print(c.name, len(c.mass), f"{c.mass.sum():.2e} Msun")
```

:::{note}
Component names become attributes, so avoid names that collide with existing
methods (`pos`, `mass`, `run`, `times`, …). `sim.add_particles("mass", ...)`
will not error at creation, but `sim.mass` will keep returning the mass array.
:::

### Components are views, and that matters for gravity

Because a component is a window onto the whole system, there are two different
questions you can ask about its self-gravity, and tambora makes you choose:

```python
# The satellite's particles, in the gravity field of EVERYTHING:
sim.sat.self_potential(t=1.0, include_all_components=True)

# The satellite's particles, in their OWN gravity only, as if isolated:
sim.sat.self_potential(t=1.0, include_all_components=False)
```

The second is what you want for "is this satellite still self-bound?"; the first
is what you want for "how deep in the total potential is it?". There is no
default on `Component` for exactly this reason.

## 3. Accessors take `t`, and return physical units

Every quantity is a *method* taking a time argument, not a stored attribute.
That is deliberate: it means `r`, `vr`, `KE` and friends are computed from the
snapshot you ask for, and `t` has one consistent meaning everywhere.

```python
sim.pos(t=-1)      # last snapshot            -> (N, 3)
sim.pos(t=1.5)     # snapshot nearest 1.5 Gyr -> (N, 3)
sim.pos()          # every snapshot           -> (n_snap, N, 3)
```

### The three kinds of `t`

| Type | Example | Meaning |
| --- | --- | --- |
| `int` | `t=0`, `t=-1` | Snapshot **index**. Negative counts from the end. |
| `float` | `t=1.5`, `t=0.0` | The snapshot **nearest that time** in Gyr. |
| omitted / `...` | `sim.pos()` | **All** snapshots, with a leading time axis. |

`t=0` and `t=0.0` are different questions that usually have the same answer. If
you ran with `t0=2.0`, `t=0` is still the first snapshot but `t=0.0` is out of
range and raises `ValueError`.

Out-of-range integers raise `IndexError`; out-of-range floats raise
`ValueError`. Both messages tell you the valid range.

### Full list of accessors

Available on **both** `Sim` and `Component`:

| Group | Methods |
| --- | --- |
| Position | `pos`, `x`, `y`, `z`, `r`, `cylR`, `phi`, `theta` |
| Velocity | `vel`, `vx`, `vy`, `vz`, `vr`, `vtheta`, `vphi`, `cylvR` |
| Momentum | `p`, `px`, `py`, `pz`, `L`, `Lx`, `Ly`, `Lz` |
| Energy | `KE`, `PE`, `energy`, `self_potential`, `compute_external_pot` |
| Acceleration | `self_gravity`, `self_ax`, `self_ay`, `self_az`, `external_acc`, `external_ax/ay/az` |

`Sim` additionally has `system_energy`, `dE`, `plot_energy_diagnostic` and
`plot_momentum_diagnostic`, which are whole-system questions.

See [Analysing results](analysis.md) for what each one actually computes.

## 4. Forces are objects, and they compose

Everything that pushes on a particle is a {class}`~tambora.dynamics.Force`.
There are two families:

**Self-gravity** — the particles' mutual attraction.
{class}`~tambora.dynamics.FalcONGravity` (default) or
{class}`~tambora.dynamics.DirectSummationGravity`. You normally select this by
string through `run(method=...)` rather than constructing it yourself.

**External forces** — everything else, most often a galpy potential wrapped in
{class}`~tambora.dynamics.ExternalGalpyPotential`.

They add:

```python
from tambora.dynamics import ExternalGalpyPotential
from galpy.potential import NFWPotential, MiyamotoNagaiPotential

halo = ExternalGalpyPotential(NFWPotential(amp=8e11, a=16.0, ro=8., vo=220.))
disk = ExternalGalpyPotential(MiyamotoNagaiPotential(amp=7e10, a=3.5, b=0.28, ro=8., vo=220.))
sim.add_external_force(halo + disk)
```

The sum of two conservative forces is still conservative, so it keeps its
`.potential()`. Mix in a non-conservative force and the result deliberately
*loses* `.potential()` — so a half-defined energy is an `AttributeError` rather
than a plausible wrong number. See [External forces](external-forces.md).

## Putting it together

```python
import numpy as np
from galpy.potential import MWPotential2014
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy
from tambora.dynamics.hooks import BoundednessHook

pos, vel, mass = mkPlummer_galpy(m=3e4, b=0.008, n=5000,
                                 center_pos=[12., 0., 0.],
                                 center_vel=[0., 140., 20.])

sim = Sim()                                     # 1. one Sim
sim.add_particles("gc", pos, vel, mass)         # 2. named component
sim.add_external_pot(MWPotential2014)           # 4. forces
sim.add_hook(BoundednessHook("gc", eps=0.002))  #    + a measurement

sim.run(t_end=3.0, dt=1e-4, dt_out=1e-2, eps=0.002)

sim.gc.r(t=3.0)                                 # 3. accessors, in kpc
```

That is the complete mental model. Everything else is detail.

## Next

- [Units](units.md) — the one place the abstraction is deliberately leaky.
- [Initial conditions](initial-conditions.md) — how to get `pos, vel, mass`.

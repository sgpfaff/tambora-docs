# Quickstart

Five minutes, one Plummer sphere, and every idea you need to read the rest of
these docs. Nothing here takes more than a couple of seconds to run.

If you would rather run it than read it, the same material is
[notebook 01](examples/01-first-simulation.ipynb), which opens in Colab.

## 1. Build a simulation

Every tambora run is one {class}`~tambora.simulation.Sim` object. You add named
particle sets to it, optionally add external forces, then call
{meth}`~tambora.simulation.Sim.run`.

```python
import numpy as np
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy

# 10^6 Msun Plummer sphere, scale radius 0.1 kpc, 2000 particles.
# Returns pos [kpc], vel [km/s], mass [Msun] — tambora's user units.
pos, vel, mass = mkPlummer_galpy(m=1e6, b=0.1, n=2000)

sim = Sim()
sim.add_particles("cluster", pos, vel, mass)
```

The string `"cluster"` is the important part. It is the name you will use to ask
questions about this set of particles for the rest of the session.

## 2. Run it

```python
sim.run(t_end=0.5, dt=1e-3, dt_out=1e-2, eps=0.02)
```

Four numbers, all in Gyr except the last:

`t_end`
: When to stop. Here, 0.5 Gyr.

`dt`
: The integration timestep. This sets your accuracy.

`dt_out`
: How often to store a snapshot. **Must be an exact multiple of `dt`** — tambora
  raises `ValueError: dt_out must be a multiple of dt.` otherwise, and it is
  strict about floating-point remainders.

`eps`
: Gravitational softening, in **kpc**. Not a time at all. It stops close pairs
  from producing infinite accelerations.

That run takes about a second and produces 51 snapshots (t = 0, 0.01, …, 0.5).

:::{tip}
`dt_out` controls memory, not accuracy. Snapshots are dense arrays: 2000
particles × 51 snapshots × 3 components × 8 bytes is only ~2.4 MB, but
100 000 particles at 1000 snapshots is 2.4 GB. Store what you will actually look
at.
:::

## 3. Ask it questions

This is the part that makes tambora tambora. Every accessor takes a time `t`,
and returns physical units.

```python
sim.cluster.r(t=0.5)      # (2000,)     radii at 0.5 Gyr          [kpc]
sim.cluster.vr(t=0.5)     # (2000,)     radial velocities         [km/s]
sim.cluster.KE(t=0.5)     # (2000,)     kinetic energy per particle
sim.pos()                 # (51, 2000, 3)  every snapshot         [kpc]
sim.times                 # (51,)       snapshot times            [Gyr]
```

`t` accepts three things, and the distinction matters:

| You pass | It means |
| --- | --- |
| a **float**, `t=0.25` | the snapshot *nearest* 0.25 Gyr |
| an **int**, `t=-1` | snapshot by *index* — `-1` is the last one |
| **omitted** (`...`) | *all* snapshots, with a leading time axis |

So `t=0` is the first snapshot and `t=0.0` is the snapshot nearest zero — which
happen to coincide here, but will not if you start a run at `t0=2.0`.

## 4. Check that you can believe it

tambora attaches a {class}`~tambora.dynamics.hooks.ConservationMonitor` to every
run automatically. After the run it is sitting on `sim.monitor`:

```python
sim.monitor.drift["energy"][-1]     # relative energy drift |ΔE/E₀|
```

For the run above that is about `3.7e-04`. That is fine for a demonstration and
too loose for science. Energy drift in a leapfrog scheme is dominated by two
things:

- **`dt` too large** relative to the shortest orbital time in your system.
- **`eps` too small**, which creates violently accelerated close pairs.

Halve `dt` and it improves roughly fourfold — leapfrog is second-order:

```python
sim2 = Sim()
sim2.add_particles("cluster", pos, vel, mass)
sim2.run(t_end=0.5, dt=2.5e-4, dt_out=1e-2, eps=0.02)
sim2.monitor.drift["energy"][-1]    # ~2.3e-05
```

Always report the drift you achieved. [Reliable N-body simulations](guide/reliable-nbody.md) explains how to pick them
properly, and how to show your choice converged.

## 5. Put it in a galaxy

One call turns any supported galpy potential into an external force:

```python
from galpy.potential import MWPotential2014

pos, vel, mass = mkPlummer_galpy(
    m=1e6, b=0.1, n=2000,
    center_pos=[12.0, 0.0, 0.0],     # kpc
    center_vel=[0.0, 140.0, 20.0],   # km/s
)

sim = Sim()
sim.add_particles("cluster", pos, vel, mass)
sim.add_external_pot(MWPotential2014)
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, eps=0.02)
```

`center_pos` and `center_vel` place the cluster's centre of mass exactly, after
sampling — so you get the orbit you asked for, not one offset by sampling noise.

:::{warning}
Bare galpy `Orbit` and potential objects use **natural units** unless you say
otherwise. `Orbit([12.0, 0.0, 1.05, 0.0, 40.0, 0.0])` means R = 12 × 8 kpc =
**96 kpc**, not 12 kpc. Always pass astropy quantities:

```python
import astropy.units as u
from galpy.orbit import Orbit

o = Orbit([12.0*u.kpc, 0.0*u.km/u.s, 140.0*u.km/u.s,
           0.0*u.kpc, 20.0*u.km/u.s, 0.0*u.deg])
o.turn_physical_on()
center_pos = [o.x(), o.y(), o.z()]
center_vel = [o.vx(), o.vy(), o.vz()]
```
:::

## 6. Plot something

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(sim.cluster.x(t=1.0), sim.cluster.y(t=1.0), s=1, c="k")
ax.set_xlabel("$x$ [kpc]")
ax.set_ylabel("$y$ [kpc]")
ax.set_aspect("equal")
```

## The whole thing

```python
import numpy as np
import matplotlib.pyplot as plt
from galpy.potential import MWPotential2014
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy

pos, vel, mass = mkPlummer_galpy(
    m=1e6, b=0.1, n=2000,
    center_pos=[12.0, 0.0, 0.0],
    center_vel=[0.0, 140.0, 20.0],
)

sim = Sim()
sim.add_particles("cluster", pos, vel, mass)
sim.add_external_pot(MWPotential2014)
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, eps=0.02)

print(f"energy drift: {sim.monitor.drift['energy'][-1]:.2e}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(sim.cluster.x(t=1.0), sim.cluster.y(t=1.0), s=1, c="k")
ax.set_xlabel("$x$ [kpc]"); ax.set_ylabel("$y$ [kpc]"); ax.set_aspect("equal")
plt.show()
```

## Where to go next

- **The concepts behind all of this** → [User guide](guide/index.md)
- **Why kpc/Gyr shows up in hook code** → [Units](guide/units.md)
- **Watching a cluster fall apart** → [Tidal streams](examples/04-tidal-stream.ipynb)
- **Measuring things mid-run** → [Hooks](guide/hooks.md)
- **Every argument of every method** → [API reference](api/index.md)

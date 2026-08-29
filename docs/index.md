---
myst:
  html_meta:
    description: "tambora — a modular N-body Python package for small galactic dynamics tasks. Documentation, tutorials and API reference."
---

# tambora <img src="_static/tambora-drum.svg" class="brand-drum" alt="A Dominican tambora, the drum the package is named after">

*Keeping your N-body simulations in the pocket.*

```{image} _static/stream_progenitor.gif
:alt: A globular cluster shedding tidal tails over 3 Gyr, with its mass-loss curve
:width: 100%
:align: center
```

The animation above is the [tidal stream example](examples/04-tidal-stream.ipynb):
a $3\times10^{4}\,M_\odot$ cluster on an eccentric orbit in `MWPotential2014`,
losing 41% of its mass over 3 Gyr. It is about forty lines of code, and it runs
in under four minutes on a laptop.

:::{admonition} This documents the released alpha
:class: warning

These pages are built against **tambora {{ tambora_version }}**, the version you
get from PyPI. tambora is pre-1.0 and backwards compatibility is *not* yet
guaranteed — if something here does not match your install, check your version
first with `python -c "import tambora; print(tambora.__version__)"`.
:::

## Start here

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Quickstart
:link: quickstart
:link-type: doc

Your first simulation in ten lines. Five minutes, start to finish.
:::

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

pip, the optional galpy dependency, and how to check it worked.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: guide/index
:link-type: doc

The concepts: units, components, self-gravity, forces, hooks. Read this once
and the rest of the API explains itself.
:::

:::{grid-item-card} {octicon}`beaker` Examples
:link: examples/index
:link-type: doc

Runnable notebooks, from a first Plummer sphere to subhalo impacts on
tidal streams. Every one opens in Colab.
:::

:::{grid-item-card} {octicon}`list-unordered` API reference
:link: api/index
:link-type: doc

Every class, method and function with its arguments, generated from the
source.
:::

:::{grid-item-card} {octicon}`question` Troubleshooting
:link: guide/troubleshooting
:link-type: doc

The errors you are most likely to hit, and what they actually mean.
:::

:::{grid-item-card} {octicon}`law` Licence & citing
:link: about
:link-type: doc

How to cite tambora, the BSD licence, and the work it builds on.
:::

::::

## Gallery

Every one of these is a notebook in [Examples](examples/index.md) that runs on a
laptop, and every frame came out of tambora.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Ring galaxy
:link: examples/08-ring-galaxy
:link-type: doc

```{raw} html
<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:4px">
  <source src="_static/gallery/ring_galaxy.mp4" type="video/mp4">
</video>
```
+++
A companion punched straight through a disk. The ring is a kinematic caustic,
not swept-up matter.
:::

:::{grid-item-card} Disk–disk merger
:link: examples/11-disk-merger
:link-type: doc

```{raw} html
<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:4px">
  <source src="_static/gallery/disk_merger.mp4" type="video/mp4">
</video>
```
+++
A tidal bridge spanning the gap, then two tails past 80 kpc, then coalescence.
:::

:::{grid-item-card} Spiral arms from a flyby
:link: examples/10-spiral-arms
:link-type: doc

```{raw} html
<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:4px">
  <source src="_static/gallery/spiral_arms.mp4" type="video/mp4">
</video>
```
+++
Prograde on the left, retrograde on the right. Same satellite; only the disk's
spin is flipped.
:::

:::{grid-item-card} Disk instability
:link: examples/09-disk-instability
:link-type: doc

```{raw} html
<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:4px">
  <source src="_static/gallery/disk_instability.mp4" type="video/mp4">
</video>
```
+++
A bare disk goes bar-unstable; the same disk inside a halo does not.
:::

::::

## What it looks like

```python
import numpy as np
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy
from galpy.potential import MWPotential2014

# A 10^6 Msun Plummer sphere, 2000 particles, 12 kpc out and orbiting.
pos, vel, mass = mkPlummer_galpy(
    m=1e6, b=0.1, n=2000,
    center_pos=[12.0, 0.0, 0.0],
    center_vel=[0.0, 140.0, 20.0],
)

sim = Sim()
sim.add_particles("cluster", pos, vel, mass)
sim.add_external_pot(MWPotential2014)
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, eps=0.02)

# Ask questions in physical units, at any time.
sim.cluster.r(t=1.0)          # (2000,) spherical radii  [kpc]
sim.cluster.vr(t=0.5)         # (2000,) radial velocities [km/s]
sim.pos()                     # (101, 2000, 3) every snapshot [kpc]
sim.monitor.drift["energy"]   # relative energy drift, per snapshot
```

## Why tambora

**Named accessors, not array indices.** Components are named when you add them
and are reachable as attributes forever after. `sim.sat.vr(t=2.0)` is
unambiguous in a way that `positions[200, 5000:8000, :]` never is.

**Units you can trust at the boundary.** You give kpc, km/s and M☉; you get kpc,
km/s and M☉ back. The internal kpc/Gyr representation never leaks into your
analysis unless you ask for it. See [Units](guide/units.md).

**Fast self-gravity.** Solvers are pluggable and selected by name. The default
is a fast-multipole tree with $O(N)$ scaling, so 50 000 particles is routine
rather than an overnight job; exact direct summation is there to check it
against.

**galpy is a first-class citizen.** Any supported galpy potential becomes an
external force with one call, and galpy's distribution functions become your
initial conditions.

**Measurement during the run, not after.** [Hooks](guide/hooks.md) see every
step. `BoundednessHook` gives you a full stripping history — which particle left
the progenitor and when — that no amount of post-processing on saved snapshots
can reconstruct as cheaply.

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

guide/index
```

```{toctree}
:hidden:
:caption: Examples

examples/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
guide/troubleshooting
about
```

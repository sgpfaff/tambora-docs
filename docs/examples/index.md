# Examples

Every example here is a Jupyter notebook that runs top to bottom against the
released tambora, and every one opens in Colab with a single click — no local
install needed.

Each notebook is also a **plain Python script**. The `.py` file beside it is the
canonical source (jupytext percent format); the `.ipynb` is generated from it.
So if you would rather run an example as a script, or diff it, or paste chunks of
it into your own code, use the `.py`:

```bash
python docs/examples/04-tidal-stream.py
```

:::{admonition} Runtimes
:class: note

Times quoted are for a laptop. The stream notebooks each run a real N-body
simulation, so they take minutes rather than seconds. Most have a `QUICK = True`
switch near the top that drops the resolution for a fast pass.
:::

## Getting started

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 01 · Your first simulation
:link: 01-first-simulation
:link-type: doc

A Plummer sphere in isolation. Building a `Sim`, running it, and asking it
questions. **~30 s**
:::

:::{grid-item-card} 02 · Initial conditions
:link: 02-initial-conditions
:link-type: doc

Sampling Plummer, King and NFW spheres, placing them on orbits, and rolling
your own. **~1 min**
:::

:::{grid-item-card} 03 · External potentials
:link: 03-external-potentials
:link-type: doc

Putting a system in a galaxy: galpy potentials, composing forces, and
test-particle runs. **~1 min**
:::

::::

## Streams

Tidal streams are tambora's home turf, and they exercise nearly every feature:
self-gravity, external potentials, hooks, per-component softening, and the galpy
bridge. These build on each other, so read them in order the first time.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 04 · Tidal streams
:link: 04-tidal-stream
:link-type: doc

The flagship example. A globular cluster disrupting in `MWPotential2014`:
morphology, stripping history, integrals of motion, and an animation.
**~4 min**
:::

:::{grid-item-card} 05 · Tidal stripping
:link: 05-tidal-stripping
:link-type: doc

*Where* and *when* stars escape. The Roche effective potential, $L_1$/$L_2$,
and the tidal-tensor approximation. **~2 min**
:::

:::{grid-item-card} 06 · Stream tracks and observables
:link: 06-stream-track
:link-type: doc

Hand the particles to galpy's `StreamTrack`: on-sky track, distances,
proper motions, width and linear density. **~2 min**
:::

:::{grid-item-card} 07 · Subhalo impacts and gaps
:link: 07-stream-gaps
:link-type: doc

Fly a dark subhalo through the stream and check the velocity kick against
the analytic impulse approximation. **~5 min**
:::

::::

:::{admonition} Planned
:class: seealso

Two more stream notebooks are in progress: an **action–angle** view of stream
debris using galpy's action finders, and **remnant velocity anisotropy**, which
tracks how $\beta(r)$ evolves as a progenitor is stripped using a custom hook.
:::

## Galaxies

Disk galaxies, and what happens when you hit one or leave it alone. Neither of
these uses a built-in sampler — every IC helper that ships with tambora is
spherical, so both notebooks build an exponential disk by hand. Read 08 first if
you have not made a disk before; 09 reuses the same construction.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 08 · Ring galaxy formation
:link: 08-ring-galaxy
:link-type: doc

A companion punched straight through a disk. Building a disk from scratch, and
why the ring is a kinematic caustic rather than swept-up matter. **~1 min**
:::

:::{grid-item-card} 09 · Disk instability
:link: 09-disk-instability
:link-type: doc

A bare disk goes bar-unstable; the same disk inside a halo does not. Bar
strength $A_2$, the Ostriker–Peebles criterion, and pattern speed. **~5 min**
:::

::::

## Running them yourself

**In Colab** — click the badge at the top of any notebook. The first cell
installs tambora and galpy, which takes a minute.

**Locally** — clone this repository and run them in Jupyter:

```bash
git clone https://github.com/sgpfaff/tambora-docs.git
cd tambora-docs
pip install --pre -r requirements.txt
jupyter lab docs/examples
```

**Rebuilding the committed outputs** — the stream notebooks ship pre-executed so
the documentation builds quickly. To regenerate them:

```bash
python tools/build_notebooks.py
```

```{toctree}
:hidden:
:caption: Getting started

01-first-simulation
02-initial-conditions
03-external-potentials
```

```{toctree}
:hidden:
:caption: Streams

04-tidal-stream
05-tidal-stripping
06-stream-track
07-stream-gaps
```

```{toctree}
:hidden:
:caption: Galaxies

08-ring-galaxy
09-disk-instability
```


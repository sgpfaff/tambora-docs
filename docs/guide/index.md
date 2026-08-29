# User guide

tambora is a publicly maintained, modular N-body Python package for small
galactic dynamics tasks, built for user-friendliness and extensibility. It runs
self-gravitating simulations with external galactic potentials, and is built so
that **asking a question of the result is as easy as asking it in words**. You
do not index into arrays of snapshots; you write `sim.cluster.r(t=2.0)` and get
spherical radii in kpc at 2 Gyr.

The [quickstart](../quickstart.md) shows you the shape of a tambora session.
This guide explains why it has that shape, so that when you hit something the
quickstart did not cover, you can guess right.

Read [Core concepts](concepts.md) and [Units](units.md) first — between them
they explain most of the surprises. The rest can be read in any order.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Core concepts
:link: concepts
:link-type: doc

`Sim`, components, snapshots, and the accessor pattern that runs through
the whole API.
:::

:::{grid-item-card} Units
:link: units
:link-type: doc

kpc, km/s, M☉, Gyr at the boundary; kpc/Gyr inside. When the difference
can bite you.
:::

:::{grid-item-card} Initial conditions
:link: initial-conditions
:link-type: doc

Plummer, King and NFW spheres, sampling any galpy DF, placing things on
orbits, and rolling your own.
:::

:::{grid-item-card} Self-gravity
:link: self-gravity
:link-type: doc

falcON vs direct summation, softening, `theta`, and per-component
softening.
:::

:::{grid-item-card} External forces
:link: external-forces
:link-type: doc

galpy potentials, composing forces with `+`, tidal tensors, and writing
your own force.
:::

:::{grid-item-card} Running a simulation
:link: running
:link-type: doc

The arguments to `run()`, the rules tambora enforces on them, snapshot
cadence, backwards integration, and memory.
:::

:::{grid-item-card} Reliable N-body simulations
:link: reliable-nbody
:link-type: doc

Choosing `dt` and `eps`, what softening really means, and how to show a
run converged. Read before you publish.
:::

:::{grid-item-card} Analysing results
:link: analysis
:link-type: doc

Every accessor, energies and angular momenta, and the on-the-fly
recomputation machinery.
:::

:::{grid-item-card} Hooks
:link: hooks
:link-type: doc

Measuring during the run: conservation monitors, boundedness tracking,
cadences, and custom hooks.
:::

:::{grid-item-card} Interoperability
:link: interoperability
:link-type: doc

Every place tambora and galpy meet: potentials, samplers, orbits, stream
tracks, and the unit boundary between them.
:::

:::{grid-item-card} Performance
:link: performance
:link-type: doc

What scales how, where the time actually goes, and how big you can go.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc

The errors you will actually hit, and what they mean.
:::

::::

```{toctree}
:hidden:

concepts
units
initial-conditions
self-gravity
external-forces
running
reliable-nbody
analysis
hooks
interoperability
performance
troubleshooting
```

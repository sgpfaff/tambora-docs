# Running a simulation

Once the particles and forces are in place, one call does the rest. tambora
keeps the numerical knobs deliberately few and explicit: a single global
timestep rather than per-particle adaptive stepping, one symplectic integrator,
and a softening length you state rather than have guessed for you. Three numbers
therefore describe the accuracy of any tambora run, which makes it something you
can report in a paper rather than reconstruct from a config file.

The other deliberate choice is that `dt` and `dt_out` are separate. Accuracy and
output cadence are different concerns, and coupling them — as codes that write a
snapshot every step effectively do — makes you trade one for the other.

```python
sim.run(t_end, dt, dt_out, t0=0.0, method='falcON',
        integration_method='leapfrog', progress=True, monitors='auto', **kwargs)
```

The full signature is in {meth}`~tambora.simulation.Sim.run`. This page covers
what each argument does and the rules tambora enforces. For *how to choose*
`dt` and `eps` well, and how to convince yourself a run is converged, see
[Reliable N-body simulations](reliable-nbody.md).

## The time arguments

`t_end`, `dt`, `dt_out` and `t0` are all in **Gyr**.

`dt` is the integration timestep and sets your accuracy. `dt_out` is how often a
snapshot is stored and sets your memory use and time resolution for analysis.
They are independent knobs — that separation is the point.

### Two rules tambora enforces

**`dt_out` must be an exact multiple of `dt`.**

```python
sim.run(t_end=1.0, dt=1.5e-4, dt_out=1e-2)
# ValueError: dt_out must be a multiple of dt.
```

`1e-2 / 1.5e-4 = 66.67`, not an integer. The tolerance is `1e-9` on the
remainder, so pick `dt` values that divide `dt_out` exactly: with
`dt_out = 1e-2`, use `dt` of `1e-4`, `1.25e-4`, `2e-4`, `2.5e-4`, `5e-4`.

**`t_end - t0` should be a whole number of `dt_out` steps.** If it is not, you
get a warning rather than an error, and the run stops at the last whole snapshot:

```text
UserWarning: Simulation duration (1.05 Gyr) is not an exact multiple of
dt_out=0.1 Gyr. Last output will be at t=1 Gyr instead of t=1.05 Gyr.
```

### Backwards integration

Set `t0` after `t_end` and make `dt` negative:

```python
sim.run(t0=0.0, t_end=-2.0, dt=-1e-3, dt_out=-1e-2)
```

`dt` and `dt_out` must share the sign of the integration direction, or you get a
`ValueError` explaining which one is wrong. Leapfrog is time-symmetric, so
integrating back and forward returns you close to where you started — a good
sanity check on your `dt`.

## Softening and timestep

`dt` is in Gyr; `eps` is a length in **kpc**, not a time. Both are required
inputs — tambora will not guess them for you.

`eps` may be a scalar, or a dict giving a different softening per component:

```python
sim.run(t_end=0.3, dt=1e-4, dt_out=1e-2,
        eps={"stream": 0.002, "subhalo": 0.25})
```

Every component must appear in the dict; a missing one raises a `ValueError`
naming it. Each value is a scalar, or an array matching that component's
particle count.

Per-component softening is what lets one `Sim` hold a dense cluster and a
diffuse perturber at once. A single particle with softening $\epsilon$ under the
Plummer kernel behaves as a Plummer sphere of scale radius $\epsilon/2$ — which
is how the [subhalo example](../examples/07-stream-gaps.ipynb) models a
perturber without resolving it. See
[Reliable N-body simulations](reliable-nbody.md#what-eps-actually-means) for
that factor of two and the other kernels.

:::{seealso}
Choosing `dt` and `eps` well, and demonstrating that your choice converged, is
covered in [Reliable N-body simulations](reliable-nbody.md).
:::

## Self-gravity method

```python
sim.run(..., method='falcON')     # default
sim.run(..., method='direct')     # O(N^2) reference
sim.run(..., method=None)         # no self-gravity
```

See [Self-gravity](self-gravity.md). `'falcON'` accepts `theta` and `kernel` as
extra kwargs; `'direct'` accepts `use_C`. Passing a kwarg the chosen method does
not understand is an error naming the offender, rather than a silent no-op.

To run with external forces only, turn self-gravity off explicitly:

```python
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, method=None)
```

Drop `eps` as well — it is a self-gravity parameter and is rejected alongside
`method=None`. This is the right way to make a test-particle run, and it is much
faster.

## Integrator

`integration_method='leapfrog'` is currently the only option, and is the
sensible default regardless: kick-drift-kick leapfrog is symplectic and
time-reversible, so energy error stays bounded and oscillates with the orbital
phase instead of growing secularly. That bounded oscillation is exactly what you
see in the energy-conservation plots throughout these docs.

The registry is public — {data}`~tambora.dynamics.INTEGRATORS` — so new
integrators can be added by name.

## Progress and monitors

```python
sim.run(..., progress=True, monitors='auto')
```

`progress` shows a tqdm bar. Hooks can push fields onto it, so by default you get
a live energy drift and, if you attached a
{class}`~tambora.dynamics.hooks.BoundednessHook`, a live bound count:

```text
54%|█████▍    | 16302/30000 [01:58<01:39, 137.9it/s, n_bound(gc)=3891, |dE/E0|=1.66e-07]
```

Set `progress=False` in scripts and notebooks whose output you intend to commit —
tqdm writes a frame per update, which bloats saved notebooks substantially.

`monitors='auto'` attaches a {class}`~tambora.dynamics.hooks.ConservationMonitor`
tracking energy. Pass `monitors=False` to attach none, or a tuple of names to
choose. Read the result from `sim.monitor` afterwards.

## Memory

Snapshot arrays are dense `float64`:

$$ \text{bytes} \approx n_{\rm snap} \times N \times 3 \times 8 \times 2 $$

(positions and velocities), plus the same again for cached self-gravity
acceleration and $n_{\rm snap} \times N \times 8$ for the potential.

| N | snapshots | pos+vel | with caches |
| --- | --- | --- | --- |
| 5 000 | 301 | 72 MB | ~150 MB |
| 20 000 | 301 | 289 MB | ~600 MB |
| 100 000 | 1 001 | 4.8 GB | ~10 GB |

If you are tight, coarsen `dt_out` first — it is free accuracy-wise. Then turn
off the caches:

```python
sim.run(..., cache_self_gravity_acc=False, cache_self_gravity_pot=False)
```

The cost is that `sim.self_gravity()` and `sim.PE()` can no longer read a cached
value and must recompute on the fly, one snapshot at a time. See
[Analysing results](analysis.md#cached-versus-on-the-fly).

## What `run()` does, in order

1. Resolves `eps` (expanding a dict into a flat per-particle array).
2. Picks and constructs the self-gravity solver, validating its kwargs.
3. Attaches default monitors, unless you supplied your own.
4. Builds the time arrays and checks the `dt` / `dt_out` rules.
5. Allocates snapshot arrays and records the state at `t0`.
6. Steps the integrator, storing a snapshot every `dt_out` and firing each hook
   on its cadence.
7. Marks the simulation as run, so setup methods now raise.

## Next

- [Self-gravity](self-gravity.md)
- [Hooks](hooks.md)
- [Analysing results](analysis.md)

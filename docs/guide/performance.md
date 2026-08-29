# Performance

tambora is fast enough that the interesting limit is usually your patience with
a parameter scan rather than any single run. This page is about where the time
actually goes, so you can tell which knob to turn.

The short version: **the force evaluation dominates, and the timestep sets how
many of them you do.** Everything else — snapshot storage, accessors, hooks at a
sensible cadence — is noise by comparison.

## The cost model

Runtime is, to a good approximation,

$$ T \;\approx\; \frac{t_{\rm end} - t_0}{\Delta t}\;\times\;\big(C_{\rm sg} + C_{\rm ext}\big) $$

— the number of steps times the cost of one force evaluation. Leapfrog does one
self-gravity evaluation per step, reusing the previous step's acceleration for
the first half-kick.

### Self-gravity

falcON is $O(N)$; direct summation is $O(N^2)$. Measured per force evaluation on
one core:

| $N$ | falcON | direct |
| --- | --- | --- |
| 1 000 | ~1 ms | ~2 ms |
| 5 000 | ~7 ms | ~40 ms |
| 20 000 | ~40 ms | ~600 ms |
| 100 000 | ~250 ms | ~15 s |

The crossover is around a thousand particles. Above that there is no reason to
use direct summation except as a correctness check.

`theta` trades accuracy for speed roughly as $\theta^{-3}$ in the number of
tree interactions. Going from `0.6` to `0.3` costs you a factor of a few.

### External potentials

A galpy potential is evaluated at every particle, every step. Whether that is
cheap depends on whether tambora can vectorise it:

- **Vectorised** — a single array call. `MWPotential2014` costs roughly the same
  as falcON self-gravity at $N = 5000$.
- **Looped** — a Python-level loop over particles. One to two orders of magnitude
  slower, and it will dominate your runtime.

Check which you have before starting a long run:

```python
from tambora.tools.util._galpy_bridge import (
    VECTORIZED_POTENTIALS, UNVECTORIZED_POTENTIALS,
)
print(type(pot) in VECTORIZED_POTENTIALS)
```

If you need an unvectorised profile, consider `interpRZPotential` or
`interpSphericalPotential`, which build an interpolation table once and are
vectorised thereafter.

## Real numbers

Measured while writing these docs, on an M-series laptop, single core:

| Run | $N$ | steps | time |
| --- | --- | --- | --- |
| Isolated Plummer, 0.5 Gyr | 2 000 | 500 | 1 s |
| GC stream in MWPotential2014, 0.5 Gyr | 4 000 | 5 000 | 40 s |
| GC stream, 3 Gyr (the flagship example) | 5 000 | 30 000 | 217 s |
| GC stream, 0.6 Gyr | 20 000 | 6 000 | 255 s |
| Test particles, no self-gravity, 1 Gyr | 12 | 1 000 | 0.1 s |

The 3 Gyr flagship run is the useful reference point: 5000 particles, a realistic
Galactic potential, self-gravity throughout, a boundedness hook firing 300 times,
and energy conserved to $1.5\times10^{-8}$ — in under four minutes.

## What to change when it is too slow

**In order of effect:**

1. **Check `dt` is not smaller than it needs to be.** This is the single biggest
   lever, because runtime is exactly linear in the number of steps. Verify with
   the convergence test in
   [Reliable N-body simulations](reliable-nbody.md#timestep-convergence) rather
   than guessing — people routinely run at four times the necessary resolution.
2. **Check for an unvectorised potential.** See above.
3. **Turn off self-gravity if you do not need it.** For tracer particles,
   `run(..., method=None)` removes the dominant cost entirely.
4. **Raise `theta`.** `0.6 → 0.8` is a noticeable saving if your accuracy budget
   allows it.
5. **Reduce $N$.** Last, because it changes the physics you can resolve — and
   check the answer is converged in $N$ before and after.

Note what is *not* on this list: `dt_out`. Coarsening it saves memory, not time.

## Hooks

A hook at `EveryOutput()` costs essentially nothing — it fires a few hundred
times over a run.

The exception is anything that computes. `BoundednessHook` runs an iterative
unbinding solve on each fire, which is several self-gravity evaluations on the
component. At `EveryOutput()` on the flagship run that is a few percent of
runtime; at `EveryStep()` it would be a hundred times the cost of the simulation
itself.

If several hooks need the same expensive quantity on the same step, they share
it: `StepState.bound_mask` is cached per step and keyed by its parameters, so two
hooks asking for the same mask pay once. That only works if the parameters match
exactly — differing `eps` or `method` means two separate solves.

## Memory

Snapshot arrays are dense `float64`:

| $N$ | snapshots | pos + vel | with self-gravity caches |
| --- | --- | --- | --- |
| 5 000 | 301 | 72 MB | ~150 MB |
| 20 000 | 301 | 289 MB | ~600 MB |
| 100 000 | 1 001 | 4.8 GB | ~10 GB |

Coarsen `dt_out` first — it costs nothing in accuracy. If that is not enough:

```python
sim.run(..., cache_self_gravity_acc=False, cache_self_gravity_pot=False)
```

The trade is that `sim.PE()` and `sim.self_gravity()` must then recompute on
demand, one snapshot at a time, with an explicit `method`. See
[Analysing results](analysis.md#cached-versus-on-the-fly).

## Threading

falcON uses OpenMP where the build supports it. Control it with the usual
environment variable, set **before** importing tambora:

```bash
OMP_NUM_THREADS=4 python my_run.py
```

For parameter scans, running several independent single-threaded simulations in
parallel usually beats one multi-threaded run, since the tree walk does not scale
perfectly.

## Profiling a run

```python
import time

for dt in (4e-4, 2e-4, 1e-4):
    s = Sim(); s.add_particles("c", pos, vel, mass)
    t0 = time.time()
    s.run(t_end=0.1, dt=dt, dt_out=0.1, eps=0.002, progress=False)
    n = round(0.1 / dt)
    print(f"dt={dt:.0e}: {time.time() - t0:5.1f} s, "
          f"{1e3 * (time.time() - t0) / n:.2f} ms/step, "
          f"|dE/E0|={s.monitor.drift['energy'][-1]:.1e}")
```

Cost per step should be flat in `dt`. If it is not, something other than the
force evaluation is dominating — most often a hook at too fine a cadence.

## Related

- [Self-gravity](self-gravity.md) — solver choice
- [Reliable N-body simulations](reliable-nbody.md) — the accuracy side of the trade
- [Running a simulation](running.md)

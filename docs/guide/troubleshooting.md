# Troubleshooting

tambora tries to fail loudly and specifically rather than quietly and plausibly —
most of its error messages name the offending argument and the valid range. This
page collects the ones you are most likely to meet, and the small number of
failures that are *silent* and therefore worth knowing in advance.

## Installation

**`No matching distribution found for tambora`**

tambora is a pre-release, and pip skips those unless asked:

```bash
pip install --pre "tambora==0.1.0a1"
```

**`ImportError: galpy is required to use tambora's galpy tools`**

Exactly what it says. `pip install galpy`, or use tambora's non-galpy features.
The galpy-backed names stay importable without it and raise only when called, so
this appears at the call site rather than at import.

**`AttributeError` for something these docs say exists**

Almost always the wrong tambora. If you have a git clone, running Python from
inside it shadows the installed package, because the working directory is first
on `sys.path`. Check:

```python
import tambora
print(tambora.__version__, tambora.__file__)
```

If that prints a path inside your clone when you meant the installed release,
`cd` somewhere else.

## Setting up

**`RuntimeError: Cannot add components after run()`**
**`RuntimeError: Cannot add hooks after run()`**

Setup must finish before `run()`. The particle arrays are already concatenated
and the snapshot arrays allocated, so a late addition would not mean what you
wrote. Build a new `Sim` — restarting from a finished one is just pulling the
final state out and feeding it in:

```python
s2 = Sim()
s2.add_particles("stream", s1.gc.pos(t=-1), s1.gc.vel(t=-1), s1.gc.mass)
```

**`ValueError: Component 'x' already exists`**

Names are unique. Note also that component names become attributes, so avoid
`mass`, `times`, `pos`, `run` and friends — `sim.add_particles("mass", ...)` will
not error, but `sim.mass` keeps returning the mass array.

**`ValueError: pos must be shape (N, 3), received (3, N)`**

tambora wants particle-major arrays. Transpose.

## Running

**`ValueError: dt_out must be a multiple of dt`**

Exactly, to within `1e-9` on the remainder. With `dt_out = 1e-2`, the usable
timesteps are its exact divisors: `1e-4`, `1.25e-4`, `2e-4`, `2.5e-4`, `5e-4`.
`1.5e-4` gives `66.67` and is rejected.

**`ValueError: The end time (t_end) is less than the start time (t0) …`**

Backwards integration needs `dt` *and* `dt_out` negative:

```python
sim.run(t0=0.0, t_end=-2.0, dt=-1e-3, dt_out=-1e-2)
```

**`UserWarning: Simulation duration … is not an exact multiple of dt_out`**

Not fatal. The run stops at the last whole snapshot, and the warning tells you
which time that is.

**`ValueError: eps dict is missing components: {...}`**

Every component must appear in the dict. There is no default for the ones you
leave out, deliberately.

**`ValueError: {'theta'} is (are) invalid kwarg(s) for 'direct' …`**

Each solver validates its own arguments. `theta` and `kernel` belong to
`'falcON'`; `'direct'` takes only `eps` and `use_C`.

**`ValueError: {'eps'} is (are) invalid kwarg(s) for None self-gravity method`**

You asked for `method=None` (no self-gravity) but still passed `eps`. Softening
is a self-gravity parameter; drop it.

:::{note}
There is no `turn_self_gravity_off()` in {{ tambora_version }} — that method
exists only in the development tree. Use `run(..., method=None)`.
:::

## Analysis

**`IndexError: Time index 500 is out of bounds …`** (integer `t`)
**`ValueError: t=5.0 Gyr is out of bounds for simulation time range …`** (float `t`)

The message gives the valid range. Remember `t=0` is an *index* and `t=0.0` is a
*time*.

**`ValueError: Cannot use cached results before run()`**

`use_cached=True` needs a cache, which only exists after a run. Supply a `method`
to compute on the fly instead.

**`ValueError: 'use_cached' is False but no 'method' was provided`**

Recomputation needs to know what with: `method='direct'` or `method='falcON'`.

**`TypeError: Cannot compute on-the-fly for all times`**

On-the-fly recomputation is one snapshot at a time. Loop:

```python
pe = np.array([sim.PE(t=i, method="direct", eps=0.002).sum()
               for i in range(len(sim.times))])
```

**`UserWarning: Computing external potential on-the-fly for multiple snapshots may be slow`**

`compute_external_pot` is not cached, so `t=...` loops. Ask for the snapshots you
need, or evaluate the potential yourself in one vectorised galpy call.

## Hooks

**`ValueError: An equivalently-configured BoundednessHook is already registered`**

Two hooks with the same `_dedup_key` are rejected as duplicates. Change the
configuration, or reuse the one you have. Different components are not
duplicates; the same component with the same settings is.

**`TypeError: cadence must be a Cadence instance`**

Pass `EveryNSteps(10)`, not `10`.

**Hook results look wrong, or all identical**

`StepState` hands out **views into live arrays**, not copies. If you stored
`state.pos()` you stored a reference that keeps changing. Copy:

```python
self.snapshots.append(state.pos().copy())
```

## Results that are wrong but do not raise

These are the ones worth internalising, because nothing tells you.

**A galpy object in natural units.** `Orbit([12.0, 0.0, 1.05, 0.0, 40.0, 0.0])`
means $R = 96$ kpc, not 12. Always attach astropy units and call
`turn_physical_on()`. tambora warns for potentials, not for orbits.

**Hook output in internal units.** `BoundednessHook.dispersion` and `com_vel`,
and captured transition velocities, are kpc/Gyr. Multiply by `KPCGYR_TO_KMS`
(0.9778). Positions, masses and times are the same in both systems. See
[Units](units.md).

**`eps` is twice the Plummer scale radius** under `kernel=0`. If you are using a
softened particle to represent an extended body, or comparing with an analytic
Plummer result, pass `eps = 2 * r_s`. See
[Reliable N-body simulations](reliable-nbody.md#what-eps-actually-means).

**Summed potential energy double-counts.** `sim.PE(t).sum()` is not the system's
potential energy — pairwise terms are shared between two particles. Use
`sim.system_energy(t)`, which applies the factor of one half.

**`vphi` is an angular velocity**, in km/s/kpc. Multiply by `cylR` for a linear
velocity.

**A too-large timestep does not crash.** It quietly heats your system. Check
`sim.monitor.drift["energy"][-1]` after every run, and converge it.

## Not implemented yet

These raise `NotImplementedError` in {{ tambora_version }}:

| Call | Instead |
| --- | --- |
| `Sim.save()` / `Sim.load()` | `np.savez_compressed` on the accessor output |
| `Sim.to_galpy_orbit()` | Build the `Orbit` by hand — see [Interoperability](interoperability.md) |
| `Sim.add_subhalos()` | Add a component, or one particle with `eps = 2*r_s` |
| `Sim.tag()` | Keep your own boolean masks |
| `compute_bound` / `compute_tidal_radius` | {func}`~tambora.dynamics.diagnostics.bound_mask` |

## Still stuck

Report the output of

```python
import tambora, galpy, numpy, sys
print(tambora.__version__, tambora.__file__)
print("galpy", galpy.__version__, "numpy", numpy.__version__)
print(sys.version)
```

along with a minimal script, at
[github.com/sgpfaff/tambora/issues](https://github.com/sgpfaff/tambora/issues).

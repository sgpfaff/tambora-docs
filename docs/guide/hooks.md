# Hooks

Most analysis happens after a run: you save snapshots, load them, and measure
things. That works until the question you want to ask is about something that
happened *between* snapshots — the exact moment a star left its cluster, say —
or until the measurement is cheap to make during the run and ruinously expensive
to reconstruct afterwards.

A hook is a callable that the integrator invokes as it goes, handing it a live
view of the simulation. It sees every step if you want it to. It accumulates
whatever you tell it to on itself, and you read the results off after `run()`
returns.

The design point is that hooks turn a simulation from something you *store* into
something you *observe*. tambora ships two, and writing your own takes about ten
lines.

## The shape of it

```python
from tambora.dynamics.hooks import BoundednessHook

bh = BoundednessHook("gc", eps=0.002, track=("com", "dispersion"))
sim.add_hook(bh)

sim.run(t_end=3.0, dt=1e-4, dt_out=1e-2, eps=0.002)

bh.fraction()          # bound fraction at every fire
bh.release_time()      # when each particle was released
```

Hooks must be registered before `run()`; adding one afterwards raises
`RuntimeError`, because the run they would have observed is already over.

## What ships

### ConservationMonitor

Attached to every run automatically, so you rarely construct it yourself. It
tracks the drift of conserved quantities and pushes the current value onto the
progress bar.

```python
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, eps=0.01)
sim.monitor.drift["energy"][-1]     # |ΔE/E₀| at the end
sim.monitor.t                       # the times it fired
sim.monitor.values["energy"]        # raw energies, not just the drift
```

The drift is `||value − value₀|| / scale`, with the reference captured at the
first fire. For energy the scale is `|E₀|`, so it reduces to the familiar
relative drift.

To turn it off, or choose what it tracks:

```python
sim.run(..., monitors=False)            # none
sim.run(..., monitors=("energy",))      # explicit
```

Or construct your own and register it, in which case `run` leaves it alone:

```python
from tambora.dynamics.hooks import ConservationMonitor, EveryNSteps

mon = ConservationMonitor(track=("energy",))
sim.add_hook(mon, cadence=EveryNSteps(50))   # finer than the default
```

### BoundednessHook

The interesting one. At every fire it solves for which particles of a component
are still self-bound — iterative unbinding in the component's own centre-of-mass
frame — and records the *changes*.

Recording changes rather than states is what makes it cheap. A full boolean
history for 5000 particles over 300 snapshots is 1.5 million values; the
transition log for the same run is a few thousand entries, and every boolean
quantity is reconstructed from it on demand.

```python
bh = BoundednessHook(
    "gc",
    eps=0.002,                          # softening for the unbinding solve
    track=("com", "com_vel", "dispersion"),
    capture_transitions=("pos", "vel"), # phase space at the moment of escape
)
```

What you can ask afterwards:

| Call | Gives you |
| --- | --- |
| `bh.fraction()` | bound fraction at each fire |
| `bh.n_bound()` / `bh.n_unbound()` | counts |
| `bh.mask_at(t)` | boolean mask at any time |
| `bh.history(times)` | masks at several times |
| `bh.release_time()` | per particle, when it was last unbound (`nan` if bound) |
| `bh.transition_times("unbound")` | every unbinding time |
| `bh.transitions("unbound")` | full events, with captured payload |
| `bh.com`, `bh.com_vel`, `bh.dispersion` | the tracked reductions, per fire |
| `bh.initial_mask` | the mask at the first fire |

`mask_at` is indexed **component-locally**, so it lines up with the component's
own accessors:

```python
bound = bh.mask_at(2.0)
sim.gc.pos(t=2.0)[bound]        # the bound particles at 2 Gyr
```

:::{warning}
Tracked reductions are in tambora's **internal** units — `com_vel` and
`dispersion` are kpc/Gyr, not km/s, as are captured transition velocities.
Multiply by `KPCGYR_TO_KMS`. Positions, masses and times are the same in both
systems. See [Units](units.md).
:::

:::{note}
**Rebinding is real and common.** A star hovering near the tidal boundary can
cross it many times before leaving for good — in the
[stripping example](../examples/05-tidal-stripping.ipynb) one star flips 47
times, and there are 4900 unbinding events for 1700 net escapes. The hook
faithfully records every crossing. `release_time()` already collapses this to
the *most recent* unbinding per particle; if you want only escapes that stuck,
also require the particle to be unbound at the end.
:::

## Cadence: when a hook fires

```python
from tambora.dynamics.hooks import EveryStep, EveryNSteps, EveryOutput, EveryNOutputs

sim.add_hook(hook, cadence=EveryOutput())    # default: aligned with snapshots
sim.add_hook(hook, cadence=EveryStep())      # every integration step
sim.add_hook(hook, cadence=EveryNSteps(10))
sim.add_hook(hook, cadence=EveryNOutputs(5))
```

If you do not pass a cadence, the hook's own `default_cadence` is used, and if it
has none, `EveryOutput()`.

Cadence is a real cost decision. `BoundednessHook` runs an iterative unbinding
solve on each fire, which is several self-gravity evaluations; at `EveryStep()`
on a 30 000-step run that will dominate your runtime. `EveryOutput()` is almost
always what you want, and it is the default for that reason.

The flip side is resolution: a transition is dated to the fire that *saw* it, so
release times are late by up to one fire interval. If you need timing precision,
pay for a finer cadence deliberately.

## Writing your own

Subclass {class}`~tambora.dynamics.hooks.Hook` and implement `__call__`. You
receive a {class}`~tambora.dynamics.integration.StepState` — a live, read-only
view with the same accessor names as `Sim`, minus the `t` argument, in internal
units.

Here is a hook that records the velocity anisotropy profile

$$ \beta(r) = 1 - \frac{\sigma_t^2}{2\sigma_r^2} $$

of a component, which is exactly the kind of thing that is awkward to
reconstruct from snapshots and trivial to accumulate live:

```python
import numpy as np
from tambora.dynamics.hooks import Hook, EveryOutput


class AnisotropyHook(Hook):
    """Radial profile of the velocity anisotropy beta(r), in the component's
    centre-of-mass frame, at every fire."""

    default_cadence = EveryOutput()

    def __init__(self, component, bins):
        self.component = component
        self.bins = np.asarray(bins)      # radial bin edges [kpc]
        self.t = []
        self.beta = []

    def _dedup_key(self):
        # Two hooks on the same component with the same bins are duplicates.
        return (type(self), self.component, self.bins.tobytes())

    def __call__(self, state):
        c = state.component(self.component)
        pos, vel, m = c.pos(), c.vel(), c.mass

        com = (m[:, None] * pos).sum(0) / m.sum()
        com_v = (m[:, None] * vel).sum(0) / m.sum()
        dx, dv = pos - com, vel - com_v

        r = np.linalg.norm(dx, axis=1)
        r_hat = dx / r[:, None]
        v_r = np.sum(dv * r_hat, axis=1)
        v_t2 = np.sum(dv**2, axis=1) - v_r**2      # both tangential components

        out = np.full(len(self.bins) - 1, np.nan)
        idx = np.digitize(r, self.bins) - 1
        for k in range(len(out)):
            sel = idx == k
            if sel.sum() > 10:
                out[k] = 1.0 - v_t2[sel].mean() / (2.0 * v_r[sel].var())

        self.t.append(state.t)
        self.beta.append(out)
        state.report(**{f"beta({self.component})": f"{np.nanmedian(out):+.2f}"})
```

Use it like anything else:

```python
bins = np.logspace(-2.5, -0.5, 12)
ah = AnisotropyHook("gc", bins)
sim.add_hook(ah)
sim.run(...)

beta = np.array(ah.beta)        # (n_fires, n_bins)
```

A bare function works too, if you do not need configuration:

```python
radii = []
sim.add_hook(lambda state: radii.append(np.median(state.gc.r())))
```

### What `StepState` gives you

Kinematics, mirroring `Sim`, with no `t` argument: `pos()`, `vel()`, `x()`,
`y()`, `z()`, `vx()`, `vy()`, `vz()`, `r()`, `KE()`, and `mass` as a property.
Components via `state.gc` or `state.component("gc")`. Metadata via `state.t` and
`state.step`.

Energies need an explicit choice, because on a component view the two meanings
differ and there is deliberately no default:

```python
state.gc.self_potential(include_all_components=False)   # the component alone
state.gc.self_potential(include_all_components=True)    # in the whole field
state.gc.energy(include_all_components=False)
state.system_energy()                                   # whole system
state.external_pot()
state.external_acc()
```

Boundedness, cached and shared between hooks that ask for the same thing on the
same step:

```python
mask = state.bound_mask("gc", eps=0.002)
```

And `state.report(**fields)` pushes named values onto the progress bar.

### Two things to know

**Arrays are borrowed, not copied.** `state.pos()` is a view into the
integrator's live array. Copy anything you intend to keep:

```python
self.snapshots.append(state.pos().copy())     # not state.pos()
```

**The fast path is free; everything else costs.** Asking for whole-system
self-gravity with the run's own solver reuses what the integrator already
computed. Asking for a component in isolation, or with a different solver,
triggers a fresh evaluation — cached for the step, but paid once per fire.

## Registration rules

`add_hook` refuses duplicates. Registering the same *instance* twice raises, as
does registering a second hook whose `_dedup_key()` matches an existing one.
That is why `_dedup_key` should be built from the hook's meaningful
configuration and never from accumulated results — two `BoundednessHook`s on
different components are not duplicates; two on the same component with the same
settings are.

Return `None` from `_dedup_key` to opt out entirely.

Read back what is registered with `sim.hooks`, which returns `(hook, cadence)`
pairs in registration order.

## Related

- [Analysing results](analysis.md) — the after-the-fact accessors
- [Units](units.md) — why hook output is in kpc/Gyr
- [Tidal stripping](../examples/05-tidal-stripping.ipynb) — `BoundednessHook` in
  anger

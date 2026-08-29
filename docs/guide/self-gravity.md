# Self-gravity

Computing the mutual gravity of $N$ particles is the expensive part of any N-body
code, and the choice of how to do it is the main accuracy/speed dial you have.
tambora gives you two solvers and one honest default.

The default is **falcON**, a fast-multipole tree that scales as $O(N)$ rather
than the $O(N^2)$ of direct summation. That scaling is why 50 000 particles is a
routine laptop run in tambora rather than an overnight job. Direct summation is
kept as the reference implementation: it makes no approximation, so it is what
you check the tree against.

Which one you are using is never implicit — `run()` takes a `method` string, and
the mapping from string to class is a public registry you can read.

## Choosing a solver

```python
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, eps=0.02, method="falcON")  # default
sim.run(..., method="direct")   # O(N^2), exact
sim.run(..., method=None)       # no self-gravity at all
```

| `method` | Class | Cost | Accuracy |
| --- | --- | --- | --- |
| `'falcON'` | {class}`~tambora.dynamics.FalcONGravity` | $O(N)$ | set by `theta` |
| `'direct'` | {class}`~tambora.dynamics.DirectSummationGravity` | $O(N^2)$ | exact |
| `None` | {class}`~tambora.dynamics.NullSelfGravity` | free | — |

The registry is {data}`~tambora.dynamics.SELF_GRAVITY_METHODS`.

Each solver validates its own keyword arguments. Passing `theta` to `'direct'`
is an error naming the offender, not a silent no-op:

```text
ValueError: {'theta'} is (are) invalid kwarg(s) for 'direct' self-gravity method.
```

## Softening

`eps` is required, in **kpc**. It replaces the $1/r^2$ divergence at small
separations with something finite, which is what stops a chance close pair from
destroying your energy conservation.

```python
sim.run(..., eps=0.02)                                  # same for everyone
sim.run(..., eps={"cluster": 0.002, "halo": 0.5})       # per component
```

Per-component softening is not a nicety. A simulation containing a dense cluster
and a diffuse halo has no single good value: whatever you pick over-softens one
or under-softens the other. Each dict value may be a scalar or an array matching
that component's particle count, and every component must appear.

For falcON you also choose a kernel:

| `kernel` | Profile |
| --- | --- |
| `0` | Plummer |
| `1` | default, falls off faster (~$r^{-7}$) |
| `2`, `3` | faster still |

:::{important}
Under `kernel=0`, `eps` is **twice** the Plummer scale radius — a nominal
`eps = 0.25` kpc behaves as a Plummer sphere of scale 0.125 kpc. This matters
whenever you use a softened particle to represent an extended body, or compare
against an analytic Plummer result. The measurement and its consequences are in
[Reliable N-body simulations](reliable-nbody.md#what-eps-actually-means).
:::

Choosing a value is covered in
[Reliable N-body simulations](reliable-nbody.md#bracketing-a-value).

## The falcON opening angle

`theta` controls when the tree may treat a distant group of particles as a
single multipole expansion rather than descending into it. Smaller is more
accurate and slower.

```python
sim.run(..., method="falcON", eps=0.002, theta=0.4)
```

`0.6` is the default and a reasonable compromise. `0.3`–`0.4` is what to use when
you care about force accuracy — the tidal field of a satellite, say. Above about
`0.8` the errors become visible in the energy budget.

The honest way to pick it is to measure, on a system small enough to afford the
exact answer:

```python
import numpy as np
from tambora.dynamics.forces.self_gravity.self_gravity import self_gravity

acc_exact, _ = self_gravity(pos, mass, method="direct", eps=0.002)

for theta in (0.3, 0.5, 0.6, 0.8, 1.0):
    acc, _ = self_gravity(pos, mass, method="falcON", eps=0.002, theta=theta)
    err = np.linalg.norm(acc - acc_exact, axis=1) / np.linalg.norm(acc_exact, axis=1)
    print(f"theta={theta}: median {np.median(err):.2e}, 99th {np.percentile(err, 99):.2e}")
```

## Turning it off

For test particles — tracers with negligible mass — self-gravity is wasted work:

```python
sim.run(t_end=1.0, dt=1e-3, dt_out=1e-2, method=None)
```

`method=None` selects {class}`~tambora.dynamics.NullSelfGravity`, which returns
zeros. This is the right way to do orbit integration in an external potential,
and it is much faster.

Note that you must **also drop `eps`**. Softening is a self-gravity parameter, so
passing it alongside `method=None` is rejected:

```text
ValueError: {'eps'} is (are) invalid kwarg(s) for None self-gravity method.
```

## Using a solver directly

Sometimes you want accelerations for a set of positions without building a
`Sim` — recomputing energies with different softening, checking a force law,
writing your own integrator. The procedural entry point does that:

```python
from tambora.dynamics.forces.self_gravity.self_gravity import self_gravity

acc, pot = self_gravity(pos, mass, method="falcON", eps=0.002, theta=0.6)
acc      = self_gravity(pos, mass, method="direct", eps=0.002,
                        return_potential=False)
```

Returns are in **internal** units: acceleration in kpc/Gyr², specific potential
in (kpc/Gyr)². This is below the accessor layer, so nothing converts for you.

It also accepts `'direct_C'` to force the C implementation of direct summation;
`'direct'` uses it by default and falls back to pure Python only if you construct
{class}`~tambora.dynamics.DirectSummationGravity` with `use_C=False`.

You can also construct solver objects and pass them around:

```python
from tambora.dynamics import DirectSummationGravity, FalcONGravity

solver = FalcONGravity(eps=0.002, theta=0.4, kernel=0)
acc, pot = solver.acc_and_potential(pos, mass)
```

`acc_and_potential` is the one to prefer when you need both: falcON and direct
summation each produce them in a single sweep, so asking separately does the work
twice.

## Component self-gravity after a run

On a `Component` there are two different questions, and tambora makes you pick:

```python
sim.sat.self_potential(t=1.0, include_all_components=True)   # in the whole field
sim.sat.self_potential(t=1.0, include_all_components=False)  # isolated
```

The first is how deep the satellite's particles sit in the total potential; the
second is whether the satellite is bound *to itself*. Since they differ, there is
no default. See [Analysing results](analysis.md).

## Cost

falcON's advantage grows with $N$. Rough scaling on one core, per force
evaluation:

| $N$ | falcON | direct |
| --- | --- | --- |
| 1 000 | ~1 ms | ~2 ms |
| 5 000 | ~7 ms | ~40 ms |
| 20 000 | ~40 ms | ~600 ms |
| 100 000 | ~250 ms | ~15 s |

Use direct summation for validation and for systems of a few thousand particles
where you want the exact answer. Use falcON for everything else.

## Related

- [Reliable N-body simulations](reliable-nbody.md) — choosing `eps`, and what it means
- [Running a simulation](running.md) — where these arguments go
- [Performance](performance.md) — where the time actually goes

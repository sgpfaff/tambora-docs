# Reliable N-body simulations

This page is not about tambora's API. It is about the handful of numerical
choices that decide whether an N-body result means anything, and how to
demonstrate — to a referee, a supervisor, or yourself in six months — that yours
were adequate.

The first half is general: it applies to any self-gravitating simulation you run
in tambora, whether that is a disk, a halo, a merger, or a dissolving cluster.
The second half, [Parameters by regime](#parameters-by-regime), gives concrete
starting points for the specific kinds of system tambora is used for.

If you already know this material, the one section worth reading anyway is
[What `eps` actually means](#what-eps-actually-means), because tambora's
softening convention carries a factor of two that will otherwise bite you.

## The failure modes are silent

Neither of the two ways an N-body run goes wrong announces itself.

A timestep that is too long does not crash. It produces a system that looks
plausible and has quietly gained energy, because the integrator injected a little
on every orbit — a cluster that expanded, a disk that thickened, a halo that
developed a core it should not have. Softening that is too small does not raise.
It produces close pairs with enormous accelerations that corrupt the energy
budget and scatter particles onto wrong orbits.

Both show up in the same diagnostic: energy conservation. That is why tambora
attaches a {class}`~tambora.dynamics.hooks.ConservationMonitor` to every run
whether you ask for it or not, and why `sim.monitor.drift["energy"][-1]` is the
first number to look at after any run.

## The timestep

The constraint is the shortest dynamical time anywhere in your system:

$$ t_{\rm dyn} \sim \sqrt{\frac{b^{3}}{G\,m}} \sim \frac{1}{\sqrt{G\rho}} $$

for a structure of mass $m$ and scale radius $b$, or equivalently in terms of its
characteristic density $\rho$. Aim for **20–50 steps per $t_{\rm dyn}$**:

```python
import numpy as np
from tambora.tools.util.units import G_INTERNAL

t_dyn = np.sqrt(b**3 / (G_INTERNAL * m))     # Gyr
dt = t_dyn / 30
```

The density form is the more useful one in practice, because it makes clear that
the *densest* region sets the timestep, not the most massive. tambora uses one
global timestep, so a dense subsystem embedded in a diffuse one is expensive
for exactly this reason: a globular cluster in a galactic halo forces the whole
simulation onto the cluster's timestep.

:::{note}
`dt_out` must be an exact multiple of `dt`, so pick `dt` from the divisors of
your intended output cadence. With `dt_out = 1e-2` Gyr, the usable timesteps
include `1e-4`, `1.25e-4`, `2e-4`, `2.5e-4` and `5e-4`.
:::

(what-eps-actually-means)=
## What `eps` actually means

Softening replaces the $1/r^2$ divergence at small separation with something
finite. tambora exposes this through `eps` (kpc) and, for falcON, a `kernel`
index. **The relationship between `eps` and the physical profile is not the
identity, and it depends on the kernel.**

You can measure it directly:

```python
import numpy as np
from tambora.dynamics.forces.self_gravity.self_gravity import self_gravity
from tambora.tools.util.units import G_INTERNAL

M, EPS = 1e10, 0.25
r = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 1.0, 2.0])

acc = []
for rr in r:
    pos = np.array([[0., 0., 0.], [rr, 0., 0.]])
    a = self_gravity(pos, np.array([M, 1e-6]), method="falcON",
                     eps=np.array([EPS, 1e-6]), theta=0.3, kernel=0,
                     return_potential=False)
    acc.append(abs(a[1, 0]))

plummer = G_INTERNAL * M * r / (r**2 + EPS**2)**1.5
print(np.array(acc) / plummer)
```

Fitting the effective Plummer scale radius for `kernel=0` gives a consistent
answer at every radius:

| nominal `eps` | effective Plummer scale radius |
| --- | --- |
| 0.25 kpc | **0.125 kpc** |

So under falcON's Plummer kernel, **`eps` is twice the Plummer scale radius**.

This matters whenever you compare against an analytic result, or when you use a
single softened particle to *represent* an extended mass. To model a Plummer
body of scale radius $r_s$, pass `eps = 2 * r_s`. Getting it wrong makes the body
twice as concentrated as intended — in the
[subhalo example](../examples/07-stream-gaps.ipynb) that inflated the peak
velocity kick by about 80% relative to the analytic prediction, a discrepancy
that looked like physics and was in fact a units convention.

The other kernels (`kernel=1`, the default, and `2`, `3`) fall off faster than
Plummer at large radius and have no simple closed-form equivalent. Use
`kernel=0` when you need to compare against analytic Plummer expressions;
otherwise the default is a better-behaved force law.

### Bracketing a value

Two considerations bound it from either side:

- **Below** the scale you want to resolve: $\epsilon \lesssim b/4$.
- **Above** the mean interparticle separation, so discreteness noise does not
  dominate: $\epsilon \gtrsim b\,N^{-1/3}$.

If that bracket is empty, you do not have enough particles to resolve what you
are asking about, and no choice of $\epsilon$ will fix it.

## Two-body relaxation: the limit you cannot soften away

Softening controls the *force law*; it does not make your $N$ particles into
real stars. A simulation of $N$ particles relaxes on roughly

$$ t_{\rm relax} \sim \frac{0.1\,N}{\ln N}\; t_{\rm dyn} $$

which for $N = 10^{4}$ is a few hundred dynamical times. Beyond that, your system
is evolving because of numerical two-body scattering rather than the physics you
intended.

Whether this matters depends entirely on what you are simulating, and it is the
main reason the regimes below want different things:

- **Collisionless systems** — disks, halos, dark-matter substructure — are
  *supposed* to have $t_{\rm relax} \gg t_{\rm Hubble}$. Here relaxation is pure
  numerical error, so you want $N$ as large as you can afford and $\epsilon$
  large enough to suppress close encounters.
- **Collisional systems** — star clusters over many relaxation times — have real
  relaxation you may be trying to capture. Then $N$ should ideally match the real
  particle number, and softening should be small.

Most tambora work sits in the first category. If your run lasts longer than
$\sim 0.1\,t_{\rm relax}$ and you are treating the system as collisionless, say
so and justify it.

(parameters-by-regime)=
## Parameters by regime

The numbers below are **starting points**, not answers. Verify them with the
convergence tests in the next section.

### Dense star clusters and globular clusters

The stiffest case: small scale radius, high density, short dynamical time.

| | typical |
| --- | --- |
| $M$ | $10^{4}$–$10^{6}\,M_\odot$ |
| scale radius $b$ | 1–20 pc |
| $t_{\rm dyn}$ | 1–10 Myr |
| `dt` | $10^{-4}$ Gyr or smaller |
| `eps` | 0.5–2 pc (`0.0005`–`0.002`) |

Worked example, the cluster in the
[tidal stream notebook](../examples/04-tidal-stream.ipynb): $M = 3\times10^{4}$,
$b = 8$ pc gives $t_{\rm dyn} = 2.7$ Myr, so `dt = 1e-4` (27 steps per
$t_{\rm dyn}$). With $N = 5000$, $bN^{-1/3} \approx 0.5$ pc and $b/4 = 2$ pc, so
`eps = 0.002` sits inside the bracket.

If the cluster is orbiting in an external potential, the cluster still sets
`dt` — the orbital time is far longer and is never the binding constraint.

### Disk galaxies

The **vertical** structure is the constraint, not the radial. A thin disk has a
much shorter vertical oscillation time than its orbital period, and under-resolving
it is what makes simulated disks spuriously thicken.

| | typical |
| --- | --- |
| scale height $h_z$ | 0.2–0.5 kpc |
| vertical time $\sim h_z/\sigma_z$ | 10–20 Myr |
| `dt` | $5\times10^{-4}$–$10^{-3}$ Gyr |
| `eps` | 0.05–0.15 kpc |

Disks are also the regime where particle number matters most for a reason other
than relaxation: too few particles and the disk develops spurious bar and spiral
modes from Poisson noise in the density field. If you are studying disk
instabilities you need enough particles that the modes you see are physical, and
you should demonstrate that by re-running at higher $N$.

### Dark matter halos and subhalos

Diffuse and long-timescale, so the cheapest regime per particle.

| | typical |
| --- | --- |
| $M$ | $10^{9}$–$10^{12}\,M_\odot$ |
| scale radius $r_s$ | 5–30 kpc |
| $t_{\rm dyn}$ | 50–200 Myr |
| `dt` | $10^{-3}$–$5\times10^{-3}$ Gyr |
| `eps` | 0.2–1 kpc |

A common convention is $\epsilon \approx r_{\rm vir}/(50\sqrt{N})$ or simply a
fixed fraction of $r_s$. Halos are firmly collisionless, so err on the side of
larger $\epsilon$ and larger $N$.

If a subhalo is only a perturber and you do not care about its internal
structure, do not resolve it at all: one massive particle with `eps = 2 * r_s`
under `kernel=0` *is* a Plummer sphere, costs one particle, and imposes no
timestep constraint of its own.

### Mixed systems

When a simulation contains components with very different densities — a cluster
in a halo, a satellite falling into a disk — remember that

- the **densest** component sets `dt` for everything, and
- `eps` should be set **per component**, via the dict form:

```python
sim.run(t_end=0.3, dt=1e-4, dt_out=1e-2,
        eps={"cluster": 0.002, "halo": 0.5, "subhalo": 0.5})
```

Using one global `eps` in a mixed simulation means either over-softening the
dense component or under-softening the diffuse one. Neither is acceptable, and
the dict form costs nothing.

## Demonstrating convergence

Do not trust any of the above. Measure.

### Timestep convergence

Leapfrog is second order, so halving `dt` should reduce the energy drift by
about a factor of four. If it does not, `dt` is not what limits you — suspect
`eps`.

```python
for dt in (4e-4, 2e-4, 1e-4):
    s = Sim()
    s.add_particles("c", pos, vel, mass)
    s.run(t_end=0.5, dt=dt, dt_out=1e-2, eps=0.002, progress=False)
    print(f"dt={dt:.1e}  |dE/E0| = {s.monitor.drift['energy'][-1]:.2e}")
```

### Solver convergence

falcON's opening angle `theta` trades accuracy for speed; `direct` summation has
no approximation at all. For a system small enough to afford it, check one
against the other. See [Self-gravity](self-gravity.md).

### Time reversibility

Leapfrog is time-symmetric, so integrating forward then backward should return
close to the initial state. Any large residual is integration error:

```python
fwd = Sim(); fwd.add_particles("c", pos, vel, mass)
fwd.run(t_end=1.0, dt=1e-4, dt_out=1e-1, eps=0.002, progress=False)

back = Sim()
back.add_particles("c", fwd.pos(t=-1), fwd.vel(t=-1), mass)
back.run(t0=1.0, t_end=0.0, dt=-1e-4, dt_out=-1e-1, eps=0.002, progress=False)

print(np.abs(back.pos(t=-1) - pos).max(), "kpc")
```

### Resolution convergence

The result that matters — a bound fraction, a bar amplitude, a stream width, a
density profile — should not change when you add particles. Run at $N$ and $2N$
and overplot. If it moves, you are measuring your resolution, not your physics.

This is the test people skip, and the one referees ask for.

## What good looks like

Measured for the runs in these docs, on a laptop:

| Setup | `dt` | `eps` | final $\lvert\Delta E/E_0\rvert$ |
| --- | --- | --- | --- |
| Isolated Plummer, $N$=2000, 0.5 Gyr | 1e-3 | 0.02 | 3.7e-04 |
| Same, quarter timestep | 2.5e-4 | 0.02 | 2.3e-05 |
| GC stream, $N$=5000, 3 Gyr | 1e-4 | 0.002 | 1.5e-08 |
| GC stream, $N$=20000, 0.6 Gyr | 1e-4 | 0.002 | ~1e-08 |

A drift below $10^{-6}$ over many orbits is comfortable for structural work.
Above $10^{-3}$, the run is not converged and quantitative claims from it are
unsafe.

The drift oscillates with orbital phase rather than growing steadily. That
bounded behaviour is the signature of a symplectic integrator, and it is the
*envelope*, not the instantaneous value, that you should quote.

## Reporting

State in the paper: the number of particles, `dt`, `eps` and the softening
kernel, the self-gravity method and `theta`, and the achieved energy
conservation. Those five numbers let someone reproduce your result, and their
absence is the most common reason an N-body figure cannot be checked.

## Related

- [Running a simulation](running.md) — the arguments themselves
- [Self-gravity](self-gravity.md) — solver accuracy and cost
- [Hooks](hooks.md) — measuring conservation during the run

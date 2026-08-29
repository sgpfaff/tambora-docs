# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Your first tambora simulation
#
# A Plummer sphere, left alone for half a gigayear. Nothing dramatic happens —
# that is the point. A system set up in equilibrium should *stay* in equilibrium,
# and checking that it does is how you learn to trust a code and your settings.
#
# Along the way this covers the four things every tambora session uses: building a
# `Sim`, naming components, running, and querying the result.
#
# **Runtime.** A few seconds.

# %%
# Colab / fresh environment only:
# %pip install --pre --quiet "tambora==0.1.0a1" galpy astropy matplotlib

# %%
import matplotlib.pyplot as plt
import numpy as np

import tambora
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy

print("tambora", tambora.__version__)

try:
    plt.style.use("../_static/tambora.mplstyle")
except OSError:
    pass

np.random.seed(42)

# %% [markdown]
# ## 1. Initial conditions
#
# `mkPlummer_galpy` samples an isotropic Plummer sphere from galpy's distribution
# function. It returns three arrays in tambora's user units — positions in kpc,
# velocities in km/s, masses in M☉ — which is exactly what `add_particles` wants.

# %%
M, B, N = 1e6, 0.1, 2000  # Msun, kpc, particles

pos, vel, mass = mkPlummer_galpy(m=M, b=B, n=N)

print(f"pos  {pos.shape}  kpc")
print(f"vel  {vel.shape}  km/s")
print(f"mass {mass.shape} Msun, {mass[0]:.1f} each")
print(f"half-mass radius {np.median(np.linalg.norm(pos, axis=1)) * 1000:.0f} pc")

# %% [markdown]
# ## 2. Build the simulation
#
# The string `"cluster"` is the component name. It is how you will refer to this
# set of particles for the rest of the session, so pick something you will
# recognise.

# %%
sim = Sim()
sim.add_particles("cluster", pos, vel, mass)

# %% [markdown]
# ### Look at what you built
#
# Evaluate the `Sim` itself and it prints an overview of everything it holds:
# components with their particle counts and masses, any external forces, and any
# registered hooks. It is the quickest way to confirm a setup is what you meant,
# and worth a glance before every `run()`.

# %%
sim

# %% [markdown]
# ## 3. Run
#
# Four numbers. `t_end`, `dt` and `dt_out` are in Gyr; `eps` is a softening
# *length* in kpc.
#
# `dt_out` must be an exact multiple of `dt`, and `eps` is not a time — those are
# the two things people get wrong first.

# %%
sim.run(t_end=0.5, dt=1e-3, dt_out=1e-2, eps=0.02, progress=False)

print(f"{len(sim.times)} snapshots from {sim.times[0]} to {sim.times[-1]} Gyr")

# %% [markdown]
# ## 4. Ask it questions
#
# Every accessor takes a time argument and returns physical units. The time can be
# an **integer index**, a **float time in Gyr**, or omitted for *all* snapshots.

# %%
print("radii at 0.5 Gyr      ", sim.cluster.r(t=0.5).shape, "kpc")
print("radial velocities     ", sim.cluster.vr(t=-1).shape, "km/s")
print("positions, all times  ", sim.pos().shape)
print()
print(f"KE at t=0   {sim.KE(t=0).sum():.4e} Msun (km/s)^2")
print(f"KE at t=0.5 {sim.KE(t=-1).sum():.4e} Msun (km/s)^2")
print(f"total energy {sim.system_energy(t=-1):.4e}")

# %% [markdown]
# `t=0` and `t=0.0` mean different things — the first is "snapshot index zero",
# the second is "the snapshot nearest time zero". Here they coincide; they would
# not if you had started the run at `t0=2.0`.

# %% [markdown]
# ## 5. Did it stay in equilibrium?
#
# A Plummer sphere sampled from its own distribution function is in equilibrium,
# so the half-mass radius should be flat and the virial ratio $-2T/U$ should sit
# near 1. Some initial adjustment is normal — the sampling is a finite realisation
# of a smooth model, and softening slightly changes the potential it lives in.

# %%
r_half = np.array([np.median(sim.cluster.r(t=i)) for i in range(len(sim.times))])

KE = np.array([sim.KE(t=i).sum() for i in range(len(sim.times))])
PE = np.array([0.5 * sim.self_potential(t=i).sum() for i in range(len(sim.times))])
virial = -2 * KE / PE

fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4))

ax[0].plot(sim.times, r_half * 1000, c="k")
ax[0].set_xlabel("$t$ [Gyr]")
ax[0].set_ylabel(r"$r_{1/2}$ [pc]")
ax[0].set_ylim(0, None)
ax[0].set_title("Half-mass radius")

ax[1].plot(sim.times, virial, c="k")
ax[1].axhline(1.0, c="#d94801", lw=1.0, ls="--")
ax[1].set_xlabel("$t$ [Gyr]")
ax[1].set_ylabel(r"$-2T/U$")
ax[1].set_title("Virial ratio")

ax[2].plot(sim.times, sim.monitor.drift["energy"], c="k")
ax[2].set_yscale("log")
ax[2].set_xlabel("$t$ [Gyr]")
ax[2].set_ylabel(r"$|\Delta E/E_0|$")
ax[2].set_title("Energy conservation")

fig.tight_layout()
plt.show()

print(f"r_half: {r_half[0] * 1000:.1f} -> {r_half[-1] * 1000:.1f} pc")
print(f"virial: {virial[0]:.3f} -> {virial[-1]:.3f}")
print(f"final |dE/E0| = {sim.monitor.drift['energy'][-1]:.2e}")

# %% [markdown]
# The cluster holds together, but that energy drift — a few times $10^{-4}$ — is
# loose. tambora attaches a `ConservationMonitor` to every run precisely so this
# number is always in front of you.
#
# Leapfrog is second order, so quartering the timestep should improve it by about
# sixteen.

# %%
sim2 = Sim()
sim2.add_particles("cluster", pos, vel, mass)
sim2.run(t_end=0.5, dt=2.5e-4, dt_out=1e-2, eps=0.02, progress=False)

d1 = sim.monitor.drift["energy"][-1]
d2 = sim2.monitor.drift["energy"][-1]
print(f"dt = 1.0e-3 : |dE/E0| = {d1:.2e}")
print(f"dt = 2.5e-4 : |dE/E0| = {d2:.2e}")
print(f"improvement factor {d1 / d2:.1f}  (expect ~16 for a 4x smaller dt)")

# %% [markdown]
# See [Reliable N-body simulations](../guide/reliable-nbody.md) for how to choose
# `dt` and `eps` properly, and how to show that your choice converged.

# %% [markdown]
# ## 6. Look at it

# %%
fig, ax = plt.subplots(1, 3, figsize=(12.0, 4.0))

for a, t, lab in zip(ax[:2], (0, -1), ("$t = 0$", f"$t = {sim.times[-1]}$ Gyr")):
    a.scatter(sim.cluster.x(t=t), sim.cluster.y(t=t), s=2, lw=0, c="k",
              rasterized=True)
    a.set_xlim(-0.6, 0.6)
    a.set_ylim(-0.6, 0.6)
    a.set_aspect("equal")
    a.set_xlabel("$x$ [kpc]")
    a.set_ylabel("$y$ [kpc]")
    a.set_title(lab)

bins = np.logspace(-2.3, 0.2, 30)
for t, c, lab in ((0, "0.6", "$t=0$"), (-1, "k", "final")):
    counts, _ = np.histogram(sim.cluster.r(t=t), bins=bins)
    shell = 4 / 3 * np.pi * np.diff(bins**3)
    ax[2].loglog(np.sqrt(bins[1:] * bins[:-1]), counts / shell, c=c, label=lab)
ax[2].set_xlabel("$r$ [kpc]")
ax[2].set_ylabel(r"$n(r)$ [kpc$^{-3}$]")
ax[2].set_title("Density profile")
ax[2].legend()

fig.tight_layout()
plt.show()

# %% [markdown]
# ## What you now know
#
# - A `Sim` holds particles, forces and (after `run()`) every snapshot, and
#   evaluating it prints an overview of all three.
# - Components are **named**, and the name becomes an attribute: `sim.cluster`.
# - Accessors take `t` and return **physical units** — kpc, km/s, M☉.
# - `sim.monitor.drift["energy"]` is your first check on any result.
#
# ## Next
#
# - [02 · Initial conditions](02-initial-conditions.ipynb) — King and NFW spheres,
#   placing systems on orbits, and writing your own sampler.
# - [03 · External potentials](03-external-potentials.ipynb) — putting this
#   cluster in a galaxy.
# - [Core concepts](../guide/concepts.md) — the ideas behind the API.

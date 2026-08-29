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
# # Initial conditions
#
# Every simulation starts as three arrays: positions in kpc, velocities in km/s,
# masses in M☉. tambora does not care where they came from, which means you can
# use its samplers, borrow galpy's, or write your own — and mix all three in one
# simulation.
#
# This notebook covers each route, and the two things that most often go wrong:
# placing a system on the orbit you actually meant, and checking that what you
# sampled is in equilibrium before you trust what it does.
#
# **Runtime.** Under a minute.

# %%
# Colab / fresh environment only:
# %pip install --pre --quiet "tambora==0.1.0a1" galpy astropy matplotlib

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from galpy.df import isotropicHernquistdf, kingdf
from galpy.orbit import Orbit
from galpy.potential import HernquistPotential, MWPotential2014

import tambora
from tambora.simulation import Sim
from tambora.tools import (
    galpy_orbit_to_tambora,
    galpydfsampler,
    galpysampler,
    mkKing_galpy,
    mkNFW_galpy,
    mkPlummer_galpy,
)
from tambora.tools.util.units import G_KPC_KMS

print("tambora", tambora.__version__)

try:
    plt.style.use("../_static/tambora.mplstyle")
except OSError:
    pass

np.random.seed(11)

# %% [markdown]
# ## 1. The built-in samplers
#
# Three ready-made profiles. Each returns `(pos, vel, mass)` and takes
# `center_pos` / `center_vel` to place the system.

# %%
plummer = mkPlummer_galpy(m=1e6, b=0.1, n=3000)
king = mkKing_galpy(m=1e6, n=3000, W0=6.0)
nfw = mkNFW_galpy(m=1e9, n=3000)

for name, (p, v, m) in [("Plummer", plummer), ("King W0=6", king), ("NFW", nfw)]:
    r = np.linalg.norm(p, axis=1)
    print(f"{name:11s} N={len(m):5d}  M={m.sum():.2e}  "
          f"r_half={np.median(r) * 1000:7.1f} pc  "
          f"sigma={np.std(v):5.1f} km/s")

# %% [markdown]
# `W0` is the King model's dimensionless central potential: larger means more
# concentrated. `mkNFW_galpy` also takes `nfw_kwargs` and `nfw_df_kwargs` to
# control the profile and the sampler.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9))

bins = np.logspace(-2.5, 1.2, 40)
mid = np.sqrt(bins[1:] * bins[:-1])
shell = 4 / 3 * np.pi * np.diff(bins**3)
for (name, (p, v, m)), c in zip(
    [("Plummer", plummer), ("King W0=6", king), ("NFW", nfw)],
    ["k", "#d94801", "#2166ac"],
):
    counts, _ = np.histogram(np.linalg.norm(p, axis=1), bins=bins)
    ax[0].loglog(mid, counts / shell / len(m), c=c, label=name)
    ax[1].hist(np.linalg.norm(v, axis=1), bins=40, histtype="step", color=c,
               density=True, label=name)

ax[0].set_xlabel("$r$ [kpc]")
ax[0].set_ylabel("normalised $n(r)$")
ax[0].set_title("Density profiles")
ax[0].legend()

ax[1].set_xlabel("$|v|$ [km s$^{-1}$]")
ax[1].set_ylabel("PDF")
ax[1].set_xscale("log")
ax[1].set_title("Speed distributions")
ax[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# The King model's truncation is visible as the sharp cutoff in density; the NFW
# profile's cusp and long tail dominate the left panel.

# %% [markdown]
# ## 2. Sampling any galpy distribution function
#
# The built-ins are thin wrappers. For anything else, build the galpy DF yourself
# and hand it to `galpydfsampler`, which handles the sampling and the unit
# conversion.

# %%
pot = HernquistPotential(amp=2e10 * u.Msun, a=1.5 * u.kpc, ro=8.0, vo=220.0)
pot.turn_physical_on()
df = isotropicHernquistdf(pot=pot, ro=8.0, vo=220.0)

p_h, v_h, m_h = galpydfsampler(df, n=3000, m_total=1e10)
print(f"Hernquist: r_half = {np.median(np.linalg.norm(p_h, axis=1)):.3f} kpc, "
      f"sigma = {np.std(v_h):.1f} km/s")

# %% [markdown]
# `galpysampler` goes one step further: give it a *potential* and it picks a
# sensible DF for you — the matching isotropic DF for Plummer, Hernquist and NFW,
# and an Eddington inversion for anything else spherical.

# %%
p_g, v_g, m_g = galpysampler(pot, n=2000, m_total=1e10)
print(f"via galpysampler: r_half = {np.median(np.linalg.norm(p_g, axis=1)):.3f} kpc")

# %% [markdown]
# :::{note}
# Both require the galpy object to have physical units turned on. If you forget,
# tambora's bridge warns you and falls back to galpy's defaults
# (`ro=8` kpc, `vo=220` km/s) — which may well not be what you meant.
# :::

# %% [markdown]
# ## 3. Putting a system on an orbit
#
# This is the step that catches people. `center_pos` and `center_vel` shift the
# sampled centre of mass *exactly*, after sampling, so Poisson noise in the
# realisation does not perturb the orbit you asked for.
#
# The orbit itself should come from galpy, with **astropy units attached**.

# %%
o = Orbit(
    [
        15.0 * u.kpc,  # R
        0.0 * u.km / u.s,  # vR
        160.0 * u.km / u.s,  # vT
        2.0 * u.kpc,  # z
        30.0 * u.km / u.s,  # vz
        45.0 * u.deg,  # phi
    ]
)
o.turn_physical_on()

pos, vel, mass = mkPlummer_galpy(
    m=1e7, b=0.05, n=2000,
    center_pos=[o.x(), o.y(), o.z()],
    center_vel=[o.vx(), o.vy(), o.vz()],
)

print(f"requested centre  {np.array([o.x(), o.y(), o.z()]).round(4)}")
print(f"sampled  centre   {pos.mean(0).round(4)}")
print(f"offset            {np.linalg.norm(pos.mean(0) - [o.x(), o.y(), o.z()]):.2e} kpc")

# %% [markdown]
# :::{warning}
# Without units, galpy's `Orbit` reads its arguments in **natural units**:
# `Orbit([15.0, 0.0, 0.73, ...])` means $R = 15 \times 8 = 120$ kpc. The
# difference is silent and enormous. Always attach `u.kpc` and `u.km/u.s`.
# :::
#
# `galpy_orbit_to_tambora` converts an orbit (or a set of orbits) directly into
# tambora arrays, which is handy for test particles:

# %%
orbits = Orbit(
    [
        [10.0, 0.0, 200.0, 0.0, 0.0, 0.0],
        [12.0, 0.0, 190.0, 0.5, 10.0, 30.0],
        [14.0, 0.0, 180.0, -0.5, -10.0, 60.0],
    ],
    ro=8.0,
    vo=220.0,
)
orbits.turn_physical_on()
p_orb, v_orb = galpy_orbit_to_tambora(orbits)
print("positions from orbits:\n", p_orb.round(3))

# %% [markdown]
# ## 4. Rolling your own
#
# Nothing special is required — just three arrays of the right shape and units. A
# uniform-density sphere in solid-body rotation, for instance:

# %%
def uniform_sphere(n, radius, m_total, omega=0.0, rng=None):
    """Uniform-density sphere, optionally rotating about z. omega in km/s/kpc."""
    rng = rng or np.random.default_rng()
    # rejection-free: r ~ U^(1/3) gives uniform density
    u_ = rng.random(n) ** (1 / 3)
    cost = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sint = np.sqrt(1 - cost**2)
    pos = radius * u_[:, None] * np.c_[sint * np.cos(phi), sint * np.sin(phi), cost]
    vel = np.zeros_like(pos)
    vel[:, 0] = -omega * pos[:, 1]
    vel[:, 1] = omega * pos[:, 0]
    return pos, vel, np.full(n, m_total / n)


p_u, v_u, m_u = uniform_sphere(2000, 0.2, 1e6, omega=50.0,
                               rng=np.random.default_rng(0))
print(f"uniform sphere: r_max = {np.linalg.norm(p_u, axis=1).max():.3f} kpc, "
      f"v_max = {np.linalg.norm(v_u, axis=1).max():.1f} km/s")

# %% [markdown]
# ## 5. Combining components
#
# Add as many named components as you like. They are integrated together and feel
# each other's gravity, but stay individually addressable.

# %%
host_p, host_v, host_m = mkPlummer_galpy(m=1e10, b=2.0, n=3000)
sat_p, sat_v, sat_m = mkPlummer_galpy(
    m=1e8, b=0.3, n=1500, center_pos=[5.0, 0.0, 0.0], center_vel=[0.0, 90.0, 0.0]
)

sim = Sim()
sim.add_particles("host", host_p, host_v, host_m)
sim.add_particles("satellite", sat_p, sat_v, sat_m)

for c in sim.components:
    print(f"{c.name:10s} N={len(c.mass):5d}  M={c.mass.sum():.2e} Msun")

# %% [markdown]
# Because the two have very different densities, they want different softening.
# `eps` takes a dict keyed by component name:

# %%
sim.run(t_end=0.1, dt=1e-3, dt_out=2e-2,
        eps={"host": 0.1, "satellite": 0.02}, progress=False)

print(f"|dE/E0| = {sim.monitor.drift['energy'][-1]:.2e}")

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.scatter(sim.host.x(t=-1), sim.host.y(t=-1), s=1.5, lw=0, c="0.6", label="host")
ax.scatter(sim.satellite.x(t=-1), sim.satellite.y(t=-1), s=1.5, lw=0, c="k",
           label="satellite")
ax.set_xlabel("$x$ [kpc]")
ax.set_ylabel("$y$ [kpc]")
ax.set_aspect("equal")
ax.legend()
ax.set_title(f"Two components at $t = {sim.times[-1]:.2f}$ Gyr")
plt.show()

# %% [markdown]
# ## 6. Check equilibrium before you trust it
#
# A sampler can only give you equilibrium for the potential it assumed. Add
# softening, truncate the profile, or embed the system in an external field, and
# it is no longer exactly in equilibrium. The cheap check is the virial ratio
# $-2T/U$, which should be close to 1 for an isolated self-gravitating system.

# %%
for name, (p, v, m) in [("Plummer", plummer), ("King W0=6", king)]:
    s = Sim()
    s.add_particles("c", p, v, m)
    s.run(t_end=0.05, dt=1e-3, dt_out=5e-2, eps=0.02, progress=False)
    T = s.KE(t=0).sum()
    U = 0.5 * s.self_potential(t=0).sum()
    print(f"{name:11s} -2T/U = {-2 * T / U:.3f}")

# %% [markdown]
# You can also sanity-check the velocity scale by hand. For a Plummer sphere the
# 1D velocity dispersion at the centre is $\sigma^2 = GM/(6b)$:

# %%
p, v, m = plummer
sigma_pred = np.sqrt(G_KPC_KMS * 1e6 / (6 * 0.1))
core = np.linalg.norm(p, axis=1) < 0.05
sigma_meas = np.std(v[core], axis=0).mean()
print(f"predicted central sigma_1D {sigma_pred:.2f} km/s")
print(f"measured                   {sigma_meas:.2f} km/s")

# %% [markdown]
# ## Summary
#
# | You want | Use |
# | --- | --- |
# | Plummer / King / NFW | `mkPlummer_galpy`, `mkKing_galpy`, `mkNFW_galpy` |
# | Any spherical galpy DF | `galpydfsampler(df, n, m_total)` |
# | Any spherical galpy potential | `galpysampler(pot, n, m_total)` |
# | Test particles on known orbits | `galpy_orbit_to_tambora(orbit)` |
# | Anything else | Build the arrays yourself |
#
# All of them take `center_pos` and `center_vel`, and all return kpc, km/s, M☉.
#
# ## Next
#
# - [03 · External potentials](03-external-potentials.ipynb) — put these systems
#   in a galaxy.
# - [Initial conditions](../guide/initial-conditions.md) — the reference version
#   of this page.

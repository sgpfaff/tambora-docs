#!/usr/bin/env python
"""High-resolution reruns of the galaxy examples, for a machine with time.

The notebooks in ``docs/examples`` run at a resolution that finishes while you
watch. That is the right trade for a tutorial and the wrong one for a picture:
scatter plots stop being informative somewhere north of a hundred thousand
particles, and what you actually want by then is a *surface density map*.

This script reruns the same three set-ups at whatever resolution you can afford
and writes density maps rather than particle dumps.

Why maps instead of snapshots
-----------------------------
Storing every snapshot costs ``N * 3 * 8 * 2 * n_snapshots`` bytes -- at 500k
particles and 200 outputs that is nearly 5 GB of RAM before you have plotted
anything, and far more on disk. A density map is a fixed ``bins x bins`` array
no matter how many particles went into it, so the memory cost of the *movie* is
decoupled from the resolution of the *simulation*.

``DensityMapHook`` therefore bins particles during the run, via tambora's hook
mechanism, and can fire on its own cadence -- so you can have finely sampled
frames while keeping ``dt_out`` coarse and the stored snapshots few.

Usage
-----
    python hires_galaxies.py --case ring    --n-disk 200000
    python hires_galaxies.py --case spiral  --n-disk 200000
    python hires_galaxies.py --case merger  --n-disk 150000

    # See the cost before committing to it
    python hires_galaxies.py --case merger --n-disk 400000 --dry-run

    # On a server
    nohup python hires_galaxies.py --case merger --n-disk 400000 \
          --out runs/merger_400k > merger.log 2>&1 &

Each run writes ``<out>/<case>_maps.npz`` (the maps plus metadata) and
``<out>/<case>_density.png``. Rendering is separate from running, so a long job
is never lost to a plotting bug: re-render any time with ``--render-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Physical set-ups. These mirror the notebooks; kept standalone deliberately so
# this file can be copied to a machine on its own.
# --------------------------------------------------------------------------

G_KPC_KMS = 4.3009e-6  # kpc (km/s)^2 / Msun

CASES = {
    "ring": dict(
        m_disk=5.0e10, h_r=3.5, h_z=0.30, q=1.2,
        m_halo=3.0e11, a_halo=16.0,
        t_end=0.8, dt=5e-4, dt_out=5e-3, eps_disk=0.10, extent=30.0,
    ),
    "spiral": dict(
        m_disk=2.5e10, h_r=3.5, h_z=0.30, q=1.5,
        m_halo=4.0e11, a_halo=14.0,
        t_end=1.0, dt=5e-4, dt_out=5e-3, eps_disk=0.10, extent=28.0,
    ),
    "merger": dict(
        m_disk=2.5e10, h_r=3.0, h_z=0.30, q=1.5,
        m_halo=2.5e11, a_halo=12.0,
        t_end=1.8, dt=5e-4, dt_out=1e-2, eps_disk=0.12, extent=90.0,
    ),
}


def build_potential(cfg, hernquist=False):
    """galpy model for the disk + halo, and grid interpolants for v_c."""
    import astropy.units as u
    from galpy import potential
    from galpy.potential import (
        DoubleExponentialDiskPotential,
        HernquistPotential,
        NFWPotential,
    )

    rho0 = cfg["m_disk"] / (4.0 * np.pi * cfg["h_r"] ** 2 * cfg["h_z"])
    disk = DoubleExponentialDiskPotential(
        amp=rho0 * u.Msun / u.kpc**3,
        hr=cfg["h_r"] * u.kpc, hz=cfg["h_z"] * u.kpc, ro=8.0, vo=220.0,
    )
    if hernquist:
        halo = HernquistPotential(
            amp=2 * cfg["m_halo"] * u.Msun, a=cfg["a_halo"] * u.kpc, ro=8.0, vo=220.0
        )
    else:
        halo = NFWPotential(
            amp=cfg["m_halo"] * u.Msun, a=cfg["a_halo"] * u.kpc, ro=8.0, vo=220.0
        )
    for q in (disk, halo):
        q.turn_physical_on()

    parts = [disk, halo]
    r_max = 5.0 * cfg["h_r"]
    Rg = np.linspace(0.05, r_max * 1.4, 140)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vcg = np.array(
            [float(potential.vcirc(parts, float(r) * u.kpc, ro=8.0, vo=220.0))
             for r in Rg]
        )
    Og = vcg / Rg
    kg = np.sqrt(np.clip(Rg * np.gradient(Og**2, Rg) + 4.0 * Og**2, 1e-12, None))
    return disk, halo, (Rg, vcg, Og, kg)


def make_disk(n, cfg, grids, spin=+1, seed=0):
    """Double-exponential disk in centrifugal + epicyclic equilibrium."""
    Rg, vcg, Og, kg = grids
    r = np.random.default_rng(seed)
    h_r, h_z, r_max = cfg["h_r"], cfg["h_z"], 5.0 * cfg["h_r"]

    RR = np.linspace(0.0, r_max, 4000)
    cdf = 1.0 - (1.0 + RR / h_r) * np.exp(-RR / h_r)
    cdf /= cdf[-1]
    R = np.interp(r.random(n), cdf, RR)

    phi = r.uniform(0.0, 2.0 * np.pi, n)
    z = np.where(r.random(n) < 0.5, -1.0, 1.0) * (-h_z * np.log(r.random(n)))
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), z])

    vc = np.interp(R, Rg, vcg)
    Om = np.interp(R, Rg, Og)
    ka = np.interp(R, Rg, kg)

    Sigma = cfg["m_disk"] / (2.0 * np.pi * h_r**2) * np.exp(-R / h_r)
    sigma_R = cfg["q"] * 3.36 * G_KPC_KMS * Sigma / ka
    sigma_phi = sigma_R * ka / (2.0 * Om)
    sigma_z = np.sqrt(2.0 * np.pi * G_KPC_KMS * Sigma * h_z)

    drift = sigma_R**2 * (1.0 - ka**2 / (4.0 * Om**2) - 2.0 * R / h_r)
    v_phi = np.sqrt(np.clip(vc**2 + drift, 0.0, None)) + r.normal(0.0, sigma_phi)
    v_R = r.normal(0.0, sigma_R)

    vel = np.column_stack(
        [
            v_R * np.cos(phi) - spin * v_phi * np.sin(phi),
            v_R * np.sin(phi) + spin * v_phi * np.cos(phi),
            r.normal(0.0, sigma_z),
        ]
    )
    # Subtract the sampling momentum, or the disk walks off the plane.
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
    return pos, vel, np.full(n, cfg["m_disk"] / n)


def make_hernquist_halo(n, cfg, seed=0, r_cut=80.0):
    """Hernquist halo from its exact isotropic DF, truncated at r_cut."""
    import astropy.units as u
    from galpy.df import isotropicHernquistdf
    from galpy.potential import HernquistPotential
    from tambora.tools import galpydfsampler

    pot = HernquistPotential(
        amp=2 * cfg["m_halo"] * u.Msun, a=cfg["a_halo"] * u.kpc, ro=8.0, vo=220.0
    )
    pot.turn_physical_on()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = isotropicHernquistdf(pot=pot, ro=8.0, vo=220.0)
        p, v, _ = galpydfsampler(df, n=int(n * 1.6), m_total=cfg["m_halo"])
    keep = np.linalg.norm(p, axis=1) < r_cut
    p, v = p[keep][:n], v[keep][:n]
    p -= p.mean(axis=0)
    v -= v.mean(axis=0)
    m_in = cfg["m_halo"] * r_cut**2 / (r_cut + cfg["a_halo"]) ** 2
    return p, v, np.full(len(p), m_in / len(p))


# --------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------

def make_density_hook(components, extent, bins, every_n_steps=None):
    """Build a hook that bins surface density during the run.

    Importing tambora lazily keeps ``--help`` fast on a login node.
    """
    from tambora.dynamics.hooks import EveryNSteps, EveryOutput, Hook

    class DensityMapHook(Hook):
        """Accumulate face-on and edge-on surface density maps as the run goes.

        Memory is ``n_frames * bins * bins * 4`` bytes per component per
        projection, independent of the particle count -- which is the entire
        point of doing this in a hook rather than post-processing snapshots.
        """

        default_cadence = (
            EveryNSteps(every_n_steps) if every_n_steps else EveryOutput()
        )

        def __init__(self, components, extent, bins):
            self.components = list(components)
            self.extent = float(extent)
            self.bins = int(bins)
            self.t = []
            self.xy = {c: [] for c in self.components}
            self.xz = {c: [] for c in self.components}

        def _bin(self, a, b, mass):
            rng = [[-self.extent, self.extent], [-self.extent, self.extent]]
            H, _, _ = np.histogram2d(a, b, bins=self.bins, range=rng, weights=mass)
            return H.astype(np.float32)

        def __call__(self, state):
            self.t.append(float(state.t))
            for name in self.components:
                s = state.component(name)
                m = s.mass
                self.xy[name].append(self._bin(s.x(), s.y(), m))
                self.xz[name].append(self._bin(s.x(), s.z(), m))

        def as_arrays(self):
            return (
                np.asarray(self.t),
                {c: np.asarray(v) for c, v in self.xy.items()},
                {c: np.asarray(v) for c, v in self.xz.items()},
            )

    return DensityMapHook(components, extent, bins)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def setup(case, n_disk, seed=0):
    """Return (Sim, eps, cfg, component_names) ready to run."""
    from tambora.simulation import Sim
    from tambora.tools import mkPlummer_galpy

    cfg = CASES[case]

    if case == "merger":
        disk_pot, halo_pot, grids = build_potential(cfg, hernquist=True)
        n_halo = max(n_disk // 2, 5000)
        r_cut = 80.0
        m_halo_in = cfg["m_halo"] * r_cut**2 / (r_cut + cfg["a_halo"]) ** 2
        m_gal = cfg["m_disk"] + m_halo_in

        d_start, r_peri = 90.0, 14.0
        v_rel = np.sqrt(2 * G_KPC_KMS * 2 * m_gal / d_start)
        v_t = np.sqrt(2 * G_KPC_KMS * 2 * m_gal * r_peri) / d_start
        v_r = -np.sqrt(max(v_rel**2 - v_t**2, 0.0))
        pos1 = np.array([-d_start / 2, 0.0, 0.0])
        pos2 = -pos1
        vel1 = np.array([-v_r / 2, +v_t / 2, 0.0])
        vel2 = -vel1
        spin = int(np.sign(pos1[0] * vel1[1] - pos1[1] * vel1[0]))

        s = Sim()
        for i, (p0, v0, sd) in enumerate([(pos1, vel1, seed + 1), (pos2, vel2, seed + 2)], 1):
            dp, dv, dm = make_disk(n_disk, cfg, grids, spin=spin, seed=sd)
            hp, hv, hm = make_halo_for(n_halo, cfg, seed=sd + 100, r_cut=r_cut)
            s.add_particles(f"disk{i}", dp + p0, dv + v0, dm)
            s.add_particles(f"halo{i}", hp + p0, hv + v0, hm)
        eps = {"disk1": cfg["eps_disk"], "halo1": 0.6,
               "disk2": cfg["eps_disk"], "halo2": 0.6}
        return s, eps, cfg, ["disk1", "disk2"]

    # ring and spiral: live disk (+ perturber), rigid halo
    disk_pot, halo_pot, grids = build_potential(cfg, hernquist=False)
    dp, dv, dm = make_disk(n_disk, cfg, grids, spin=+1, seed=seed + 1)
    s = Sim()
    s.add_particles("disk", dp, dv, dm)

    if case == "ring":
        m_int, b_int = 0.4 * cfg["m_disk"], 1.0
        sp, sv, sm = mkPlummer_galpy(
            m=m_int, b=b_int, n=max(n_disk // 5, 2000),
            center_pos=[0.0, 0.0, -25.0], center_vel=[0.0, 0.0, 300.0],
        )
        s.add_particles("sat", sp, sv, sm)
        eps = {"disk": cfg["eps_disk"], "sat": 0.25}
    else:  # spiral
        import astropy.units as u
        from galpy.orbit import Orbit

        o = Orbit([38.0 * u.kpc, -140.0 * u.km / u.s, 95.0 * u.km / u.s,
                   0.0 * u.kpc, 0.0 * u.km / u.s, 0.0 * u.deg])
        o.turn_physical_on()
        sp, sv, sm = mkPlummer_galpy(
            m=0.4 * cfg["m_disk"], b=1.5, n=max(n_disk // 8, 2000),
            center_pos=[o.x(), o.y(), o.z()], center_vel=[o.vx(), o.vy(), o.vz()],
        )
        s.add_particles("sat", sp, sv, sm)
        eps = {"disk": cfg["eps_disk"], "sat": 0.40}

    s.add_external_pot(halo_pot)
    return s, eps, cfg, ["disk"]


def make_halo_for(n, cfg, seed, r_cut):
    return make_hernquist_halo(n, cfg, seed=seed, r_cut=r_cut)


# --------------------------------------------------------------------------
# Cost estimate, run, render
# --------------------------------------------------------------------------

def estimate(case, n_disk, bins, frames):
    cfg = CASES[case]
    n_total = n_disk * (2 if case == "merger" else 1)
    n_total += (n_disk // 2) * 2 if case == "merger" else n_disk // 5
    n_steps = int(round(cfg["t_end"] / cfg["dt"]))
    n_snap = int(round(cfg["t_end"] / cfg["dt_out"])) + 1
    snap_gb = n_total * 3 * 8 * 2 * n_snap / 1e9
    map_gb = frames * bins * bins * 4 * 2 / 1e9
    n_disk = n_disk  # named for the occupancy estimate below
    print(f"case            {case}")
    print(f"particles       {n_total:,}")
    print(f"steps           {n_steps:,}   ({cfg['t_end']} Gyr at dt={cfg['dt']})")
    print(f"stored snapshots{n_snap:>6,}")
    print(f"snapshot RAM    {snap_gb:6.2f} GB   <- the thing that will bite you")
    print(f"density maps    {map_gb:6.2f} GB   ({frames} frames at {bins}^2)")

    # The map is only as good as its occupancy. Most of the disk sits well
    # inside the plotted extent, so take a quarter of the area as the part that
    # actually holds galaxy -- a rough but honest guide.
    per_bin = n_disk / (0.25 * bins * bins)
    print(f"particles/bin  ~{per_bin:6.1f} in the occupied region")
    if per_bin < 2:
        print("   ^ too sparse: this will render as shot noise, not a density map.")
        print(f"     Use --bins {int(bins / 2)} or raise --n-disk by "
              f"{int(np.ceil(2 / max(per_bin, 1e-9)))}x.")
    print(f"\nRough wall time scales as N log N * steps; measure one short run first.")
    return snap_gb


def run(case, n_disk, out, bins, frames, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CASES[case]

    print(f"[{case}] building initial conditions ({n_disk:,} disk particles)...")
    t0 = time.time()
    sim, eps, cfg, comps = setup(case, n_disk, seed=seed)
    print(f"[{case}] ICs built in {time.time() - t0:.1f}s")

    n_steps = int(round(cfg["t_end"] / cfg["dt"]))
    every = max(n_steps // frames, 1)
    hook = make_density_hook(comps, cfg["extent"], bins, every_n_steps=every)
    sim.add_hook(hook)

    print(f"[{case}] running: {cfg['t_end']} Gyr, dt={cfg['dt']}, "
          f"{n_steps:,} steps, map every {every} steps")
    t0 = time.time()
    sim.run(t_end=cfg["t_end"], dt=cfg["dt"], dt_out=cfg["dt_out"],
            eps=eps, theta=0.6)
    wall = time.time() - t0
    print(f"[{case}] done in {wall / 60:.1f} min, "
          f"|dE/E0| = {sim.monitor.drift['energy'][-1]:.2e}")

    t, xy, xz = hook.as_arrays()
    # Start from the case config, then overlay the run-specific fields. Doing
    # it the other way round collides on keys the config already defines.
    meta = {k: v for k, v in cfg.items() if isinstance(v, (int, float))}
    meta.update(
        case=case, n_disk=n_disk, bins=bins, extent=cfg["extent"],
        wall_seconds=wall, components=comps,
        drift=float(sim.monitor.drift["energy"][-1]),
    )
    npz = out / f"{case}_maps.npz"
    np.savez_compressed(
        npz, t=t, meta=json.dumps(meta),
        **{f"xy_{c}": xy[c] for c in comps},
        **{f"xz_{c}": xz[c] for c in comps},
    )
    print(f"[{case}] wrote {npz} ({npz.stat().st_size / 1e6:.1f} MB), "
          f"{len(t)} frames")
    return npz


def render(npz, out=None):
    """Render a density-map PNG grid (and an MP4 if ffmpeg is available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    npz = Path(npz)
    out = Path(out) if out else npz.parent
    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    t = d["t"]
    comps = meta["components"]
    ext = meta["extent"]
    case = meta["case"]

    def frame(i, proj="xy"):
        tot = None
        for c in comps:
            a = d[f"{proj}_{c}"][i]
            tot = a if tot is None else tot + a
        return tot.T

    cell = (2 * ext / meta["bins"]) ** 2

    def smoothed(i, proj="xy"):
        """Light smoothing: one bin of blur hides shot noise without inventing
        structure. Falls back gracefully if scipy is not installed."""
        a = frame(i, proj) / cell
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(a, 1.0)
        except ImportError:
            return a

    # Scale from percentiles of the occupied pixels: a fixed vmax/3e3 floor
    # turns every empty bin into visible speckle.
    sample = np.concatenate(
        [smoothed(i).ravel() for i in range(0, len(t), max(len(t) // 8, 1))]
    )
    occupied = sample[sample > 0]
    vmax = np.percentile(occupied, 99.8) if occupied.size else 1.0
    vmin = max(np.percentile(occupied, 40.0), vmax / 1e3) if occupied.size else 1e-3

    picks = np.linspace(0, len(t) - 1, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6))
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(smoothed(i), origin="lower", extent=[-ext, ext, -ext, ext],
                  cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax),
                  interpolation="nearest")
        ax.set_title(f"$t = {t[i]:.2f}$ Gyr")
        ax.set_xlabel("$x$ [kpc]")
        ax.set_ylabel("$y$ [kpc]")
        ax.set_aspect("equal")
    fig.suptitle(f"{case} — {meta['n_disk']:,} disk particles, "
                 f"{meta['bins']}$^2$ density maps")
    fig.tight_layout()
    png = out / f"{case}_density.png"
    fig.savefig(png, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    return png


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--n-disk", type=int, default=200_000,
                    help="disk particles (per galaxy for the merger)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--bins", type=int, default=320, help="density map resolution")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cost estimate and stop")
    ap.add_argument("--render-only", metavar="NPZ",
                    help="skip the simulation, just re-render this .npz")
    a = ap.parse_args()

    if a.render_only:
        render(a.render_only, a.out)
        return 0

    estimate(a.case, a.n_disk, a.bins, a.frames)
    if a.dry_run:
        return 0
    print()
    npz = run(a.case, a.n_disk, a.out, a.bins, a.frames, a.seed)
    render(npz, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

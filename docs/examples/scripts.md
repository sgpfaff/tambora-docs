---
myst:
  html_meta:
    description: "High-resolution reruns of the tambora galaxy examples, written to run unattended on a server and produce surface density maps."
---

# Scripts to run on a server

The notebooks are sized to finish while you watch. That is the right trade for
learning and the wrong one for a picture: past roughly a hundred thousand
particles a scatter plot stops conveying anything, and what you want instead is a
**surface density map**.

[`scripts/hires_galaxies.py`](https://github.com/{{ gh_user }}/{{ gh_docs_repo }}/blob/master/docs/examples/scripts/hires_galaxies.py)
reruns three of the galaxy examples at whatever resolution your machine can
stand, and writes density maps rather than particle dumps. It is a single file
with no imports from this repository, so you can copy it to a cluster on its own.

```bash
python hires_galaxies.py --case ring    --n-disk 200000
python hires_galaxies.py --case spiral  --n-disk 200000
python hires_galaxies.py --case merger  --n-disk 150000
```

## Check the cost before you commit to it

`--dry-run` prints the shape of the job and stops:

```console
$ python hires_galaxies.py --case merger --n-disk 400000 --dry-run
case            merger
particles       1,200,000
steps           3,600   (1.8 Gyr at dt=0.0005)
stored snapshots   181
snapshot RAM     10.43 GB   <- the thing that will bite you
density maps      0.31 GB   (150 frames at 512^2)
```

That RAM figure is the one to look at. `Sim` keeps every snapshot in memory so
its accessors can answer questions about any time, which costs
`N x 3 x 8 x 2 x n_snapshots` bytes. At a million particles it dominates
everything else, and a coarser `dt_out` is the cheapest lever you have.

## Why the maps come from a hook

The natural instinct is to run the simulation, save snapshots, and bin them
afterwards. That does not scale: the snapshots are the expensive thing.

Instead the script attaches a `DensityMapHook` — a
{class}`~tambora.dynamics.hooks.Hook` subclass that bins particles into a 2D
histogram *during* the run:

```python
class DensityMapHook(Hook):
    default_cadence = EveryNSteps(every_n_steps)

    def __call__(self, state):
        self.t.append(float(state.t))
        for name in self.components:
            s = state.component(name)
            self.xy[name].append(self._bin(s.x(), s.y(), s.mass))
```

A map is `bins x bins` floats no matter how many particles went into it, so the
cost of the movie is decoupled from the resolution of the simulation. Because the
hook carries its own {class}`~tambora.dynamics.hooks.Cadence`, it can also fire
far more often than `dt_out` — giving finely sampled frames while the stored
snapshots stay few. That combination is the whole trick.

## Output

Each run writes:

- `<out>/<case>_maps.npz` — the face-on and edge-on maps, the times, and a JSON
  metadata blob recording every parameter, the wall time and the energy drift.
- `<out>/<case>_density.png` — a six-panel log-scaled density figure.

Rendering is deliberately separate from running, so a twelve-hour job is never
lost to a typo in a plotting call:

```bash
python hires_galaxies.py --render-only runs/merger_400k/merger_maps.npz
```

## Running it unattended

```bash
nohup python hires_galaxies.py --case merger --n-disk 400000 \
      --out runs/merger_400k > merger.log 2>&1 &
```

:::{admonition} Measure one short run first
:class: warning

Wall time scales roughly as $N \log N$ per step times the number of steps, but
the constant depends on your machine, your BLAS and `theta`. Run the case once at
low `--n-disk`, look at the reported wall time, and scale from there — rather
than discovering the cost eight hours in.
:::

## What these are not

These reruns are for pictures and for convergence checks. They use the same
physics as the notebooks, which means the same caveats apply: rigid haloes where
the notebooks used them, no gas anywhere, and idealised coplanar encounters.
More particles makes the images better; it does not make the model more
realistic.

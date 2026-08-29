# Installation

## The short version

```bash
pip install --pre "tambora==0.1.0a1" galpy
```

The `--pre` matters: tambora is currently an alpha release, and pip skips
pre-releases unless you ask for them. Without it you will get
`No matching distribution found for tambora`, which is misleading — the package
is there, pip is just being conservative.

## Requirements

| | |
| --- | --- |
| Python | 3.10, 3.11, 3.12 or 3.13 |
| Platform | Linux (`manylinux`) and macOS (Intel + Apple Silicon) |
| Required | `numpy`, `tqdm` |
| Optional | `galpy` ≥ 1.9, `matplotlib`, `astropy` |

Windows is not currently supported. Use WSL2, where the Linux wheels install
normally.

tambora ships **binary wheels** containing its compiled falcON and direct-summation
extensions, so a normal `pip install` needs no compiler. If pip falls back to
building the source distribution, that means no wheel matched your
Python/platform combination — check `python --version` against the table above.

## Do I need galpy?

`galpy` is optional, but in practice you almost certainly want it. Without it
you can still build simulations from your own arrays and run them under
self-gravity. With it you additionally get:

- **External potentials** — {class}`~tambora.dynamics.ExternalGalpyPotential`
  and {class}`~tambora.dynamics.TidalTensorGalpyForce`.
- **Initial conditions** — {func}`~tambora.tools.mkPlummer_galpy`,
  {func}`~tambora.tools.mkKing_galpy`, {func}`~tambora.tools.mkNFW_galpy`,
  and sampling from any spherical galpy distribution function.
- **Orbit interoperability** — {func}`~tambora.tools.galpy_orbit_to_tambora`.

tambora imports cleanly without galpy; the galpy-backed names stay importable and
raise a pointed `ImportError` only when you actually call them. So this is a
perfectly valid thing to see in a stack trace, and it means exactly what it says:

```text
ImportError: galpy is required to use tambora's galpy tools but could not be
imported. See https://docs.galpy.org/en/stable/installation.html
```

There is also an extra that pulls galpy in for you:

```bash
pip install --pre "tambora[galpy]==0.1.0a1"
```

## A clean environment

Mixing tambora into a crowded environment is the most common source of confusing
import errors. A fresh one costs nothing:

::::{tab-set}

:::{tab-item} venv
```bash
python3 -m venv tambora-env
source tambora-env/bin/activate
pip install --pre "tambora==0.1.0a1" galpy matplotlib astropy jupyterlab
```
:::

:::{tab-item} conda
```bash
conda create -n tambora python=3.12
conda activate tambora
pip install --pre "tambora==0.1.0a1" galpy matplotlib astropy jupyterlab
```
:::

::::

## Check it worked

```bash
python -c "
import tambora
from tambora.simulation import Sim
from tambora.tools import mkPlummer_galpy
import numpy as np

pos, vel, mass = mkPlummer_galpy(m=1e6, b=0.1, n=500)
s = Sim(); s.add_particles('c', pos, vel, mass)
s.run(t_end=0.05, dt=1e-3, dt_out=1e-2, eps=0.02, progress=False)
print('tambora', tambora.__version__, 'OK — dE/E0 =', s.monitor.drift['energy'][-1])
"
```

You should see something like:

```text
tambora 0.1.0a1 OK — dE/E0 = 0.000188...
```

If that runs, both the C extensions and the galpy bridge are working.

:::{admonition} A trap worth knowing about
:class: caution

If you also have a git clone of tambora, **do not run Python from inside that
clone's root directory** while relying on the pip-installed version. Python puts
the working directory at the front of `sys.path`, so `import tambora` will pick
up the local source tree instead of the installed package — silently, and with a
different API. If you get an `AttributeError` for something the docs say exists,
check `python -c "import tambora; print(tambora.__file__)"` first.
:::

## Building from source

Only needed if you are developing tambora itself or are on an unsupported
platform. You will need a C compiler.

```bash
git clone https://github.com/sgpfaff/tambora.git
cd tambora
pip install -e ".[galpy,test]"
pytest tambora
```

## Next

→ [Quickstart](quickstart.md)

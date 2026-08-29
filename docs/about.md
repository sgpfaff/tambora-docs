# Citing, licence and acknowledgements

tambora is a thin, friendly layer over a lot of other people's hard work. The
fast-multipole solver at its core is Walter Dehnen's falcON, reached through the
interface Eugene Vasiliev built for pyfalcon; every potential, distribution
function and coordinate transform comes from Jo Bovy's galpy. If you publish
something tambora computed, most of the credit belongs upstream.

This page tells you exactly what to cite for what, and explains why tambora is
GPL-licensed.

## What to cite

Cite tambora **plus** whatever produced the physics in your figure. The table
maps what you used onto what to cite; details and full references follow.

```{list-table}
:header-rows: 1
:widths: 32 68

* - If you used…
  - Cite
* - tambora at all
  - **tambora** (software, below)
* - self-gravity — the default `method='falcON'`
  - **Dehnen (2000, 2002)** and **Vasiliev, Belokurov & Evans (2022)**
* - any external potential, e.g. `MWPotential2014`
  - **Bovy (2015)**
* - `MWPotential2014` specifically as a Milky Way model
  - **Bovy (2015)**
* - initial conditions from `mkPlummer_galpy`, `mkKing_galpy`, `mkNFW_galpy`, `galpydfsampler`, `galpysampler`
  - **Bovy (2015)** (the DFs and sampling are galpy's)
* - stream tracks and observables (`StreamTrack`)
  - **Bovy (2015)**
* - the impulse approximation for subhalo encounters
  - **Erkal & Belokurov (2015)**, **Sanders, Bovy & Erkal (2016)**, and **Bovy (2015)** for the implementation
```

Almost every tambora run uses falcON, so in practice **Dehnen and Vasiliev et al.
should appear in nearly every paper that uses tambora**, and galpy in nearly
every one that puts something in a galaxy.

### tambora

tambora is pre-1.0 and has no paper yet. Cite the software and state the version:

```bibtex
@software{tambora,
  author  = {Pfaffman, Gabriel},
  title   = {{tambora}: a publicly maintained, modular N-body Python package
             for small galactic dynamics tasks},
  version = {0.1.0a1},
  url     = {https://github.com/sgpfaff/tambora},
  year    = {2026},
}
```

Because the API is still changing, **record the exact version**:

```python
import tambora
print(tambora.__version__)
```

### falcON — the self-gravity solver

The $O(N)$ fast-multipole tree that makes tambora fast. Used whenever
`method='falcON'`, which is the default.

> Dehnen, W. (2000), *A Very Fast and Momentum-Conserving Tree Code*,
> ApJ 536, L39. [doi:10.1086/312724](https://doi.org/10.1086/312724)
>
> Dehnen, W. (2002), *A Hierarchical $O(N)$ Force Calculation Algorithm*,
> J. Comput. Phys. 179, 27.
> [doi:10.1006/jcph.2002.7026](https://doi.org/10.1006/jcph.2002.7026)

### pyfalcon — the Python interface to falcON

tambora's `_falcon` extension follows the interface that
[pyfalcon](https://github.com/GalacticDynamics-Oxford/pyfalcon) established for
calling falcON from Python — the same
`gravity(pos, mass, eps, theta, kernel) -> (acc, pot)` entry point. pyfalcon is
by Eugene Vasiliev and the Galactic Dynamics Oxford group, and was developed for:

> Vasiliev, E., Belokurov, V. & Evans, N. W. (2022), *Radialization of Satellite
> Orbits in Galaxy Mergers*, ApJ 926, 203.
> [doi:10.3847/1538-4357/ac4fbc](https://doi.org/10.3847/1538-4357/ac4fbc) ·
> [arXiv:2108.00010](https://arxiv.org/abs/2108.00010)

**Please cite this alongside Dehnen whenever you use tambora's falcON backend.**

### galpy — potentials, distribution functions, coordinates

> Bovy, J. (2015), *galpy: A Python Library for Galactic Dynamics*,
> ApJS 216, 29.
> [doi:10.1088/0067-0049/216/2/29](https://doi.org/10.1088/0067-0049/216/2/29)

This covers external potentials, `MWPotential2014`, the distribution functions
behind every initial-conditions helper, `StreamTrack`, and the
impulse-approximation functions. See galpy's own
[acknowledging page](https://docs.galpy.org/en/stable/index.html) — individual
potentials and DFs sometimes ask for an additional reference.

### Stream–subhalo encounters

The analytic kick formulae compared against N-body in
[notebook 07](examples/07-stream-gaps.ipynb):

> Erkal, D. & Belokurov, V. (2015), *Forensics of subhalo–stream encounters:
> the three phases of gap growth*, MNRAS 450, 1136.
> [doi:10.1093/mnras/stv655](https://doi.org/10.1093/mnras/stv655)
>
> Sanders, J. L., Bovy, J. & Erkal, D. (2016), *Dynamics of stream–subhalo
> interactions*, MNRAS 457, 3817.
> [doi:10.1093/mnras/stw232](https://doi.org/10.1093/mnras/stw232)

### Reproducibility

Citations are half of it. Also report the numbers that let someone rerun your
result: particle count, `dt`, `eps` and the softening kernel, the self-gravity
method and `theta`, and the energy conservation you achieved. See
[Reliable N-body simulations](guide/reliable-nbody.md#reporting).

## Licence

:::{important}
tambora is licensed under the **GNU General Public License, version 2 or later**
(`GPL-2.0-or-later`) — *not* a permissive licence. This follows directly from
the fact that tambora bundles and compiles falcON.
:::

tambora is Copyright © 2026 Gabriel Pfaffman.

> tambora is free software: you can redistribute it and/or modify it under the
> terms of the GNU General Public License as published by the Free Software
> Foundation; either version 2 of the License, or (at your option) any later
> version.
>
> tambora is distributed in the hope that it will be useful, but WITHOUT ANY
> WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
> A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Full text: [gnu.org/licenses/gpl-2.0.html](https://www.gnu.org/licenses/gpl-2.0.html),
and [`LICENSE`](https://github.com/sgpfaff/tambora/blob/master/LICENSE) in the
repository.

### Why GPL: tambora is a derived work

tambora **bundles the falcON source** — the same code distributed as
`gyrfalcON` in [NEMO](https://astronemo.readthedocs.io/) — under

```text
tambora/dynamics/forces/self_gravity/falcON/_falcON_src/
```

and `setup.py` compiles it directly into the `_falcon` extension module shipped
inside the package. Those sources are Copyright © Walter Dehnen and are
GPL-2.0-or-later. Two files carry additional copyright:
`inc/utils/timer.h` (Song Ho Ahn, Walter Dehnen) and `inc/public/simd.h`
(Walter Dehnen, Paul McMillan).

tambora is therefore **a work based on falcON**, and the GPL requires the whole
to be distributed under the same terms. As tambora's own licence file puts it,
this *"is not a preference; it follows from bundling falcON."*

The Python-facing side is likewise **partially derived from
[pyfalcon](https://github.com/GalacticDynamics-Oxford/pyfalcon)**, Eugene
Vasiliev's Python interface to falcON, which is where the extraction of falcON
from NEMO and the shape of the `gravity()` entry point come from. pyfalcon is
itself GPL, consistent with the licence tambora inherits.

What this means in practice: if you distribute tambora, or software that
incorporates it, you must do so under the GPL and make the source available.
Using tambora to produce research results carries no such obligation — just
cite it, and the work above.

### Other bundled material

**OpenAstronomy packaging guide.** tambora's packaging scaffolding is based on
the [OpenAstronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide),
which is BSD 3-Clause. BSD is GPL-compatible in this direction, so that material
may be included in a GPL-licensed work. See `licenses/TEMPLATE_LICENSE.rst`.

**galpy** is a runtime dependency, not bundled, and is separately licensed
(BSD 3-Clause).

### This documentation

The prose, figures and example notebooks here are released under the same
GPL-2.0-or-later terms as tambora, so the example code can be reused in your own
GPL-compatible work.

## Acknowledgements

Beyond the works cited above:

- **NEMO**, the stellar-dynamics toolbox that distributes falcON as `gyrfalcON`,
  and from which pyfalcon extracted it.
- **The scientific Python stack** — NumPy, SciPy, matplotlib, astropy, tqdm and
  Jupyter.
- **Documentation toolchain** — [Sphinx](https://www.sphinx-doc.org/),
  [Furo](https://pradyunsg.me/furo/),
  [MyST-NB](https://myst-nb.readthedocs.io/) and
  [jupytext](https://jupytext.readthedocs.io/).

## Contributing

Contributions are welcome, including to these docs — a confusing paragraph is a
bug worth reporting.

- **tambora:** [github.com/sgpfaff/tambora](https://github.com/sgpfaff/tambora)
- **These docs:** [github.com/sgpfaff/tambora-docs](https://github.com/sgpfaff/tambora-docs)

:::{admonition} Imposter syndrome disclaimer
:class: note

We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you — the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source projects
is a fantastic way to advance one's coding skills. Writing perfect code isn't the
measure of a good developer (that would disqualify all of us!); it's trying to
create something, making mistakes, and learning from those mistakes.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project — and yes, that includes feedback about the contribution process. Some of
these contributions may be the most valuable to the project as a whole, because
you're coming to it with fresh eyes.

*This disclaimer was originally written by
[Adrienne Lowe](https://github.com/adriennefriend) for a
[PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted by
tambora based on its use in the README file for the
[MetPy project](https://github.com/Unidata/MetPy).*
:::

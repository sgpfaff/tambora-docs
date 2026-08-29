# Licence and acknowledgements

## Citing tambora

tambora is pre-1.0 and does not yet have a citable paper. If you use it in
published work, please cite the software directly and state the version:

```text
Pfaffman, G. tambora: an N-body code for the modern era, version 0.1.0a1.
https://github.com/sgpfaff/tambora
```

```bibtex
@software{tambora,
  author  = {Pfaffman, Gabriel},
  title   = {{tambora}: an N-body code for the modern era},
  version = {0.1.0a1},
  url     = {https://github.com/sgpfaff/tambora},
  year    = {2026},
}
```

Because tambora is an alpha and its API is still changing, **record the exact
version** you ran with:

```python
import tambora
print(tambora.__version__)
```

Alongside it, report the numbers that make the run reproducible — particle
count, `dt`, `eps`, softening kernel, self-gravity method and `theta`, and the
energy conservation you achieved. See
[Reliable N-body simulations](guide/reliable-nbody.md#reporting).

## Licence

### tambora

tambora is Copyright © 2026 Gabriel Pfaffman and is released under the
**BSD 3-Clause Licence**.

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the tambora team nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

The full text is in
[`licenses/LICENSE.rst`](https://github.com/sgpfaff/tambora/blob/master/licenses/LICENSE.rst).

### This documentation

The prose, figures and example notebooks in this documentation are released
under the same BSD 3-Clause licence, so you are free to reuse the example code
in your own work.

## Acknowledgements

tambora stands on a good deal of other people's work.

**falcON.** The fast-multipole self-gravity solver is Walter Dehnen's falcON
algorithm, which is what makes $O(N)$ force evaluation — and therefore
laptop-scale N-body — possible at all.

> Dehnen, W. (2000), *A Very Fast and Momentum-Conserving Tree Code*,
> ApJ 536, L39. [doi:10.1086/312724](https://doi.org/10.1086/312724)
>
> Dehnen, W. (2002), *A Hierarchical O(N) Force Calculation Algorithm*,
> J. Comput. Phys. 179, 27.
> [doi:10.1006/jcph.2002.7026](https://doi.org/10.1006/jcph.2002.7026)

**galpy.** Every external potential, distribution function and coordinate
transform in these pages comes from Jo Bovy's galpy. tambora's galactic-dynamics
capability is very largely galpy's, reached through a thin bridge.

> Bovy, J. (2015), *galpy: A Python Library for Galactic Dynamics*,
> ApJS 216, 29. [doi:10.1088/0067-0049/216/2/29](https://doi.org/10.1088/0067-0049/216/2/29)

If you use tambora's external potentials or samplers, **cite galpy too**.

**MWPotential2014**, used throughout the examples, is the Milky Way model
introduced in that same paper.

**Stream tracks.** The `StreamTrack` machinery used in
[notebook 06](examples/06-stream-track.ipynb) is galpy's, and the
impulse-approximation formulae compared against N-body in
[notebook 07](examples/07-stream-gaps.ipynb) implement expressions from:

> Erkal, D. & Belokurov, V. (2015), *Forensics of subhalo–stream encounters*,
> MNRAS 450, 1136. [doi:10.1093/mnras/stv655](https://doi.org/10.1093/mnras/stv655)
>
> Sanders, J. L., Bovy, J. & Erkal, D. (2016), *Dynamics of stream–subhalo
> interactions*, MNRAS 457, 3817.
> [doi:10.1093/mnras/stw232](https://doi.org/10.1093/mnras/stw232)

**The scientific Python stack.** NumPy, SciPy, matplotlib, astropy, tqdm and
Jupyter. tambora's packaging follows the
[OpenAstronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide).

**Documentation toolchain.** Built with
[Sphinx](https://www.sphinx-doc.org/), the
[Furo](https://pradyunsg.me/furo/) theme,
[MyST-NB](https://myst-nb.readthedocs.io/) and
[jupytext](https://jupytext.readthedocs.io/).

## Contributing

Contributions are welcome, including to these docs — a confusing paragraph is a
bug worth reporting.

- **tambora itself:** [github.com/sgpfaff/tambora](https://github.com/sgpfaff/tambora)
- **This documentation:** [github.com/sgpfaff/tambora-docs](https://github.com/sgpfaff/tambora-docs)

:::{admonition} Imposter syndrome disclaimer
:class: note

We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you — the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Writing perfect code isn't the
measure of a good developer (that would disqualify all of us); it's trying to
create something, making mistakes, and learning from those mistakes.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project — and yes, that includes feedback about the contribution process. Some
of these contributions may be the most valuable to the project as a whole,
because you're coming to it with fresh eyes.

*This disclaimer was originally written by
[Adrienne Lowe](https://github.com/adriennefriend) for a
[PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted by
tambora based on its use in the README file for the
[MetPy project](https://github.com/Unidata/MetPy).*
:::

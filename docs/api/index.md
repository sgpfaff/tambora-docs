---
myst:
  html_meta:
    description: "Complete API reference for tambora: classes, methods, functions and their arguments."
---

# API reference

Every entry on this page is generated from the docstrings of the installed
`tambora` {{ tambora_version }}, so signatures and argument descriptions here
are always exactly what the code does.

:::{tip}
Looking for a concept rather than a name? The [user guide](../guide/index.md)
explains *why* each of these exists. This page is the lookup table.
:::

## The two objects you use constantly

```{eval-rst}
.. currentmodule:: tambora.simulation

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   Sim
   Component
```

`Sim` is the simulation itself: you build it, run it, and query it. `Component`
is the view you get back from `sim.<name>` for one named particle set — it
carries the same accessors, restricted to that component's particles.

## Initial conditions and galpy interoperability

```{eval-rst}
.. currentmodule:: tambora.tools

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   :nosignatures:

   mkPlummer_galpy
   mkKing_galpy
   mkNFW_galpy
   galpysampler
   galpydfsampler
   galpy_orbit_to_tambora
```

All of these return the same `(pos, vel, mass)` triple in tambora's user units
(kpc, km/s, M☉), ready to hand to {meth}`~tambora.simulation.Sim.add_particles`.
They require the optional `galpy` dependency; see [Installation](../installation.md).

## Forces

The force hierarchy is small on purpose. `Force` is the abstract root;
`Conservative` is the mixin that adds a potential. Everything else is a leaf you
either construct directly or get handed by `Sim`.

```{eval-rst}
.. currentmodule:: tambora.dynamics

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   Force
   Conservative
   NullForce
```

### Self-gravity solvers

```{eval-rst}
.. currentmodule:: tambora.dynamics

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   SelfGravityForce
   FalcONGravity
   DirectSummationGravity
   NullSelfGravity
```

There is also a one-shot procedural entry point, useful when you want
accelerations for a set of positions without building a `Sim`:

```{eval-rst}
.. currentmodule:: tambora.dynamics.forces.self_gravity.self_gravity

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   :nosignatures:

   self_gravity
```

### External forces

```{eval-rst}
.. currentmodule:: tambora.dynamics

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   ExternalConservativeForce
   ExternalGalpyPotential
   TidalTensorGalpyForce
```

Forces compose with `+`. Adding two conservative forces gives you something that
still has a `.potential()`; mixing in a non-conservative one deliberately gives
you an object without one, so a half-defined energy is an `AttributeError`
rather than a wrong number.

## Hooks — measuring things *during* a run

```{eval-rst}
.. currentmodule:: tambora.dynamics.hooks

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   Hook
   ConservationMonitor
   BoundednessHook
```

### Cadences

A cadence decides *when* a hook fires.

```{eval-rst}
.. currentmodule:: tambora.dynamics.hooks

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   Cadence
   EveryStep
   EveryNSteps
   EveryOutput
   EveryNOutputs
```

### What a hook receives

```{eval-rst}
.. currentmodule:: tambora.dynamics.integration

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   StepState
```

## Integration

```{eval-rst}
.. currentmodule:: tambora.dynamics.integration

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   BaseIntegrator
   LeapfrogIntegrator
```

```{eval-rst}
.. currentmodule:: tambora.dynamics.integration.BaseIntegrator

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   StepResult
```

## Standalone diagnostics

These are the pure functions the hooks are built from. Use them directly when
you want to answer the same question about a saved snapshot after the fact.

```{eval-rst}
.. currentmodule:: tambora.dynamics.diagnostics

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   :nosignatures:

   bound_mask
   reconstruct_mask
```

## Units and constants

tambora talks to you in kpc, km/s, M☉ and Gyr, and works internally in
kpc, kpc/Gyr, M☉ and Gyr. See [Units](../guide/units.md) for when that
distinction can bite you.

```{eval-rst}
.. currentmodule:: tambora.tools.util.units

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   :nosignatures:

   unit_handler
```

```{list-table} Module-level constants in `tambora.tools.util.units`
:header-rows: 1
:widths: 26 22 52

* - Name
  - Value
  - Meaning
* - `G_INTERNAL`
  - `4.4985e-06`
  - G in kpc³ M☉⁻¹ Gyr⁻² — the internal-unit gravitational constant.
* - `G_KPC_KMS`
  - `4.3009e-06`
  - G in kpc (km/s)² M☉⁻¹, for hand-checks in user units.
* - `KMS_TO_KPCGYR`
  - `1.022712`
  - Multiply km/s by this to get kpc/Gyr (user → internal velocity).
* - `KPCGYR_TO_KMS`
  - `0.977792`
  - Multiply kpc/Gyr by this to get km/s (internal → user velocity).
* - `KM_TO_KPC`
  - `3.2408e-17`
  - Kilometres per kiloparsec, inverted.
* - `GYR_TO_MYR`
  - `1000.0`
  - Gyr → Myr.
* - `KMS_TO_KPCMYR` / `KPCMYR_TO_KMS`
  - —
  - Legacy kpc/Myr conversions, kept for backward compatibility.
* - `INTERNAL_TO_USER_UNITS`
  - `dict`
  - Per-quantity conversion factors used by the accessor decorator.
```

## Registries

Two dictionaries map the string names accepted by
{meth}`~tambora.simulation.Sim.run` onto implementations. They are the
authoritative answer to "what can I pass here?", and are extensible.

```{eval-rst}
.. currentmodule:: tambora.dynamics

.. autodata:: SELF_GRAVITY_METHODS
   :no-value:

.. autodata:: INTEGRATORS
   :no-value:
```

| Registry | Keys in {{ tambora_version }} |
| --- | --- |
| `SELF_GRAVITY_METHODS` | `'falcON'`, `'direct'`, `None` |
| `INTEGRATORS` | `'leapfrog'` |

The procedural `self_gravity()` function additionally accepts `'direct_C'` to
force the C implementation of direct summation.

```{toctree}
:hidden:

generated/tambora.simulation.Sim
generated/tambora.simulation.Component
```

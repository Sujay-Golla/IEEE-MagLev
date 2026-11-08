"""Registries: the plug-in points of the toolkit.

Every mover (magnet array), coil shape and tile layout is a *builder function*
registered under a name. A YAML config refers to that name, e.g.

    mover:
      type: halbach_2d

To add your own idea, write one function and decorate it:

    from maglev_sim.registry import register_mover

    @register_mover("my_ring")
    def my_ring(params):
        ...
        return Mover(...)

Nothing else in the code needs to change. See docs/CONTRIBUTING.md.
"""

from __future__ import annotations

from typing import Callable, Dict

MOVERS: Dict[str, Callable] = {}
COILS: Dict[str, Callable] = {}
LAYOUTS: Dict[str, Callable] = {}


def _register(table: Dict[str, Callable], kind: str):
    def decorator_factory(name: str):
        def decorator(fn: Callable) -> Callable:
            if name in table and table[name] is not fn:
                raise ValueError(f"A {kind} called '{name}' is already registered.")
            table[name] = fn
            return fn

        return decorator

    return decorator_factory


register_mover = _register(MOVERS, "mover")
register_coil = _register(COILS, "coil")
register_layout = _register(LAYOUTS, "layout")


def lookup(table: Dict[str, Callable], kind: str, name: str) -> Callable:
    """Find a builder by name, with a helpful error that lists the options."""
    if name not in table:
        import difflib

        close = difflib.get_close_matches(name, table.keys(), n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise KeyError(
            f"Unknown {kind} type '{name}'.{hint} Available: {sorted(table)}"
        )
    return table[name]

"""Common input checks and domain exceptions."""

import math
from numbers import Integral, Real


class HeatLabError(Exception):
    """Base class for expected simulation errors."""


class InputError(HeatLabError, ValueError):
    """Invalid geometry, material, boundary condition, or time settings."""


class IncompatibleSteadyError(InputError):
    """The prescribed fluxes and generation admit no steady solution."""


class NonUniqueSteadyError(InputError):
    """The steady field is defined up to a constant and needs a reference mean."""


class SolverError(HeatLabError, RuntimeError):
    """A finite numerical solution could not be computed."""


def real_number(value: float, label: str, *, minimum: float | None = None,
                strict: bool = False) -> float:
    try:
        number = float(value) if isinstance(value, Real) and not isinstance(value, bool) else math.nan
    except OverflowError:
        number = math.nan
    if not math.isfinite(number):
        raise InputError(f"{label} must be finite. (مقدار باید عددی و متناهی باشد.)")
    if minimum is not None:
        invalid = number <= minimum if strict else number < minimum
        if invalid:
            relation = ">" if strict else "≥"
            raise InputError(f"{label} must be {relation} {minimum:g}. "
                             f"(شرط لازم برای مقدار: {relation} {minimum:g})")
    return number


def integer(value: int, label: str, minimum: int, maximum: int) -> int:
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or not minimum <= value <= maximum):
        raise InputError(f"{label} must be an integer in [{minimum}, {maximum}]. "
                         f"(مقدار باید عدد صحیح در این بازه باشد.)")
    return int(value)

"""Formatting utilities for exact rational probabilities, odds, and frequencies."""

from __future__ import annotations

from fractions import Fraction


def format_percentage(value: Fraction | float, decimals: int = 2) -> str:
    """Format a probability in [0, 1] as a human-readable percentage string."""
    pct = float(value) * 100.0
    return f"{pct:.{decimals}f} %"


def format_signed_percentage_delta(delta: Fraction | float, decimals: int = 2) -> str:
    """Format a probability delta in percentage points (pp) with explicit sign."""
    pp = float(delta) * 100.0
    sign = "+" if pp >= 0.0 else ""
    return f"{sign}{pp:.{decimals}f} pp"


def format_fraction(value: Fraction) -> str:
    """Format an exact Fraction as 'numerator / denominator'."""
    return f"{value.numerator} / {value.denominator}"


def format_latex_fraction(value: Fraction) -> str:
    """Format an exact Fraction as a LaTeX \\frac{num}{den}."""
    return f"\\frac{{{value.numerator}}}{{{value.denominator}}}"


def format_likelihood_ratio(value: Fraction | float | None, decimals: int = 2) -> str:
    """Format a likelihood ratio LR+ or LR- as a decimal multiplier."""
    if value is None:
        return "∞"
    return f"{float(value):.{decimals}f}"


def format_integer(count: int) -> str:
    """Format an integer population count using thin space thousands separator."""
    return f"{count:,}".replace(",", " ")

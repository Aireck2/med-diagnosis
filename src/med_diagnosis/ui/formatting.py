"""Funciones de formateo para probabilidades racionales exactas y frecuencias naturales."""

from __future__ import annotations

from fractions import Fraction


def format_percentage(value: Fraction | float, decimals: int = 2) -> str:
    """Formatea una probabilidad en [0, 1] como cadena de porcentaje legible."""
    pct = float(value) * 100.0
    return f"{pct:.{decimals}f} %"


def format_signed_percentage_delta(delta: Fraction | float, decimals: int = 2) -> str:
    """Formatea una variación de probabilidad en puntos porcentuales (pp) con signo explícito."""
    pp = float(delta) * 100.0
    sign = "+" if pp >= 0.0 else ""
    return f"{sign}{pp:.{decimals}f} pp"


def format_fraction(value: Fraction) -> str:
    """Formatea un objeto Fraction exacto como 'numerador / denominador'."""
    return f"{value.numerator} / {value.denominator}"


def format_latex_fraction(value: Fraction) -> str:
    """Formatea un objeto Fraction exacto como fracción LaTeX \\frac{num}{den}."""
    return f"\\frac{{{value.numerator}}}{{{value.denominator}}}"


def format_integer(count: int) -> str:
    """Formatea un conteo entero de personas usando espacio como separador de miles."""
    return f"{count:,}".replace(",", " ")

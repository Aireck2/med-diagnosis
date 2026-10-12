"""Cálculo de incertidumbre mediante entropía de Shannon."""

from __future__ import annotations

from math import isclose, isfinite, log2
from typing import Mapping

from diagnosis_simulator.domain import PROBABILITY_TOLERANCE


def shannon_entropy(distribution: Mapping[str, float]) -> float:
    """Devuelve la entropía de la distribución, medida en bits."""
    if not distribution:
        raise ValueError("La distribución debe incluir al menos un resultado.")

    probabilities = tuple(distribution.values())
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not isfinite(probability)
        or not 0.0 <= probability <= 1.0
        for probability in probabilities
    ):
        raise ValueError("Cada probabilidad debe ser un número finito entre 0 y 1.")
    if not isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
    ):
        raise ValueError("La suma de la distribución debe ser 1.")

    # Por definición, el límite de -p*log2(p) cuando p tiende a cero es cero.
    return -sum(probability * log2(probability) for probability in probabilities if probability > 0.0)

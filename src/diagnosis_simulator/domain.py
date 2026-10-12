"""Tipos y validaciones del dominio probabilístico del simulador."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from typing import Mapping

PROBABILITY_TOLERANCE = 1e-9


class SymptomStatus(str, Enum):
    """Estado que el usuario asigna a un síntoma evaluado."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Symptom:
    """Síntoma que puede observarse en el catálogo de ejemplo."""

    id: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class Condition:
    """Condición ilustrativa, su prior y probabilidades de síntomas."""

    id: str
    label: str
    prior: float
    symptom_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class DiagnosisCatalog:
    """Catálogo completo que define el modelo probabilístico de ejemplo."""

    conditions: tuple[Condition, ...]
    symptoms: tuple[Symptom, ...]

    def validate(self) -> None:
        """Valida identificadores, priors y probabilidades condicionales."""
        if not self.conditions:
            raise ValueError("El catálogo debe incluir al menos una condición.")
        if not self.symptoms:
            raise ValueError("El catálogo debe incluir al menos un síntoma.")

        condition_ids = [condition.id for condition in self.conditions]
        symptom_ids = [symptom.id for symptom in self.symptoms]
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("Los identificadores de condición deben ser únicos.")
        if len(set(symptom_ids)) != len(symptom_ids):
            raise ValueError("Los identificadores de síntoma deben ser únicos.")
        if any(not identifier for identifier in condition_ids + symptom_ids):
            raise ValueError("Los identificadores no pueden estar vacíos.")

        prior_total = 0.0
        for condition in self.conditions:
            _validate_probability(condition.prior, f"prior de {condition.id}")
            prior_total += condition.prior
            if set(condition.symptom_probabilities) != set(symptom_ids):
                raise ValueError(
                    f"La condición {condition.id} debe definir cada síntoma exactamente una vez."
                )
            for symptom_id, probability in condition.symptom_probabilities.items():
                _validate_probability(
                    probability, f"P({symptom_id} presente | {condition.id})"
                )

        if not isclose(prior_total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE):
            raise ValueError("La suma de los priors debe ser 1.")


def _validate_probability(value: float, name: str) -> None:
    """Comprueba que una probabilidad sea un número finito entre cero y uno."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} debe ser una probabilidad entre 0 y 1.")

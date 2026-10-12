"""Pruebas de validación del dominio probabilístico."""

import pytest

from diagnosis_simulator.domain import Condition, DiagnosisCatalog, Symptom


def make_catalog(prior: float = 1.0, probability: float = 0.5) -> DiagnosisCatalog:
    """Construye un catálogo mínimo para aislar las reglas de validación."""
    return DiagnosisCatalog(
        conditions=(
            Condition("condition-a", "Condición A", prior, {"cough": probability}),
        ),
        symptoms=(Symptom("cough", "Tos"),),
    )


def test_valid_catalog_is_accepted() -> None:
    make_catalog().validate()


@pytest.mark.parametrize(
    "probability", [-0.1, 1.1, float("nan"), float("inf"), True]
)
def test_catalog_rejects_invalid_symptom_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        make_catalog(probability=probability).validate()


def test_catalog_requires_normalized_priors() -> None:
    catalog = DiagnosisCatalog(
        conditions=(
            Condition("a", "A", 0.4, {"cough": 0.5}),
            Condition("b", "B", 0.4, {"cough": 0.5}),
        ),
        symptoms=(Symptom("cough", "Tos"),),
    )

    with pytest.raises(ValueError, match="priors"):
        catalog.validate()


def test_catalog_requires_every_symptom_probability() -> None:
    catalog = DiagnosisCatalog(
        conditions=(Condition("a", "A", 1.0, {}),),
        symptoms=(Symptom("cough", "Tos"),),
    )

    with pytest.raises(ValueError, match="cada síntoma"):
        catalog.validate()


def test_catalog_rejects_duplicate_identifiers() -> None:
    catalog = DiagnosisCatalog(
        conditions=(
            Condition("a", "A", 0.5, {"cough": 0.5}),
            Condition("a", "B", 0.5, {"cough": 0.5}),
        ),
        symptoms=(Symptom("cough", "Tos"),),
    )

    with pytest.raises(ValueError, match="únicos"):
        catalog.validate()

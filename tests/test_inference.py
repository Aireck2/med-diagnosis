"""Casos analíticos del motor de inferencia bayesiana."""

from math import isclose

import pytest

from diagnosis_simulator.domain import Condition, DiagnosisCatalog, Symptom, SymptomStatus
from diagnosis_simulator.inference import infer_diagnoses


@pytest.fixture
def catalog() -> DiagnosisCatalog:
    """Devuelve un modelo pequeño con posterior calculable a mano."""
    return DiagnosisCatalog(
        conditions=(
            Condition("a", "Condición A", 0.6, {"cough": 0.8, "fever": 0.2}),
            Condition("b", "Condición B", 0.4, {"cough": 0.25, "fever": 0.5}),
        ),
        symptoms=(Symptom("cough", "Tos"), Symptom("fever", "Fiebre")),
    )


def test_no_evidence_preserves_priors(catalog: DiagnosisCatalog) -> None:
    result = infer_diagnoses(catalog, {})

    assert result.posterior_probabilities == {"a": 0.6, "b": 0.4}
    assert result.evidence_probability == 1.0


def test_present_symptom_matches_analytical_bayes(catalog: DiagnosisCatalog) -> None:
    result = infer_diagnoses(catalog, {"cough": SymptomStatus.PRESENT})

    # P(tos) = 0.6*0.8 + 0.4*0.25 = 0.58; P(A|tos) = 0.48/0.58.
    assert isclose(result.evidence_probability, 0.58)
    assert isclose(result.posterior_probabilities["a"], 0.48 / 0.58)
    assert isclose(result.posterior_probabilities["b"], 0.10 / 0.58)


def test_absent_symptom_uses_complement(catalog: DiagnosisCatalog) -> None:
    result = infer_diagnoses(catalog, {"cough": SymptomStatus.ABSENT})

    # P(no tos) = 0.6*0.2 + 0.4*0.75 = 0.42; P(A|no tos) = 0.12/0.42.
    assert isclose(result.evidence_probability, 0.42)
    assert isclose(result.posterior_probabilities["a"], 0.12 / 0.42)


def test_multiple_symptoms_use_conditional_independence(catalog: DiagnosisCatalog) -> None:
    result = infer_diagnoses(
        catalog,
        {"cough": SymptomStatus.PRESENT, "fever": SymptomStatus.ABSENT},
    )

    # P(E|A)=0.8*0.8 y P(E|B)=0.25*0.5 bajo la independencia declarada.
    evidence_probability = 0.6 * 0.8 * 0.8 + 0.4 * 0.25 * 0.5
    assert isclose(result.evidence_probability, evidence_probability)
    assert isclose(result.posterior_probabilities["a"], 0.384 / evidence_probability)
    assert isclose(sum(result.posterior_probabilities.values()), 1.0)


def test_unknown_symptom_does_not_change_result(catalog: DiagnosisCatalog) -> None:
    result = infer_diagnoses(catalog, {"cough": SymptomStatus.UNKNOWN})

    assert result.posterior_probabilities == {"a": 0.6, "b": 0.4}


def test_impossible_evidence_is_reported(catalog: DiagnosisCatalog) -> None:
    impossible_catalog = DiagnosisCatalog(
        conditions=(Condition("a", "A", 1.0, {"cough": 0.0}),),
        symptoms=(Symptom("cough", "Tos"),),
    )

    with pytest.raises(ValueError, match="probabilidad cero"):
        infer_diagnoses(impossible_catalog, {"cough": SymptomStatus.PRESENT})


def test_very_small_likelihoods_keep_normalized_posterior() -> None:
    catalog = DiagnosisCatalog(
        conditions=(
            Condition("a", "A", 0.5, {"s1": 1e-200, "s2": 1e-200}),
            Condition("b", "B", 0.5, {"s1": 2e-200, "s2": 2e-200}),
        ),
        symptoms=(Symptom("s1", "S1"), Symptom("s2", "S2")),
    )

    result = infer_diagnoses(
        catalog,
        {"s1": SymptomStatus.PRESENT, "s2": SymptomStatus.PRESENT},
    )

    assert isclose(result.posterior_probabilities["a"], 0.2)
    assert isclose(result.posterior_probabilities["b"], 0.8)
    assert isclose(sum(result.posterior_probabilities.values()), 1.0)


def test_unknown_symptom_id_is_rejected(catalog: DiagnosisCatalog) -> None:
    with pytest.raises(ValueError, match="no incluidos"):
        infer_diagnoses(catalog, {"unknown": SymptomStatus.PRESENT})

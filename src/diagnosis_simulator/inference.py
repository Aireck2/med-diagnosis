"""Motor de inferencia bayesiana independiente de la interfaz."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping

from diagnosis_simulator.domain import DiagnosisCatalog, SymptomStatus


@dataclass(frozen=True)
class DiagnosisResult:
    """Probabilidades y cantidades intermedias de una inferencia."""

    prior_probabilities: Mapping[str, float]
    likelihoods: Mapping[str, float]
    weighted_likelihoods: Mapping[str, float]
    posterior_probabilities: Mapping[str, float]
    evidence_probability: float
    log_evidence_probability: float


def infer_diagnoses(
    catalog: DiagnosisCatalog,
    evidence: Mapping[str, SymptomStatus],
) -> DiagnosisResult:
    """Calcula la posterior usando Bayes y la ley de probabilidad total.

    Los síntomas se combinan bajo el supuesto didáctico de independencia
    condicional dada cada condición. Los estados no evaluados se omiten.
    """
    catalog.validate()
    symptom_ids = {symptom.id for symptom in catalog.symptoms}
    unknown_ids = set(evidence) - symptom_ids
    if unknown_ids:
        raise ValueError(f"Síntomas no incluidos en el catálogo: {sorted(unknown_ids)}")

    evaluated_evidence: dict[str, SymptomStatus] = {}
    for symptom_id, status in evidence.items():
        normalized_status = _coerce_status(status, symptom_id)
        if normalized_status is not SymptomStatus.UNKNOWN:
            evaluated_evidence[symptom_id] = normalized_status
    priors = {condition.id: condition.prior for condition in catalog.conditions}

    # Sin observaciones, la identidad de Bayes conserva exactamente los priors.
    if not evaluated_evidence:
        return DiagnosisResult(
            prior_probabilities=priors,
            likelihoods={condition.id: 1.0 for condition in catalog.conditions},
            weighted_likelihoods=priors.copy(),
            posterior_probabilities=priors.copy(),
            evidence_probability=1.0,
            log_evidence_probability=0.0,
        )

    log_likelihoods: dict[str, float] = {}
    log_weighted_likelihoods: dict[str, float] = {}
    for condition in catalog.conditions:
        log_likelihood = 0.0
        for symptom_id, status in evaluated_evidence.items():
            probability_present = condition.symptom_probabilities[symptom_id]
            probability_observed = (
                probability_present
                if status is SymptomStatus.PRESENT
                else 1.0 - probability_present
            )
            log_likelihood += _safe_log(probability_observed)
        log_likelihoods[condition.id] = log_likelihood
        log_weighted_likelihoods[condition.id] = (
            _safe_log(condition.prior) + log_likelihood
        )

    # Log-sum-exp mantiene estable la suma total cuando los productos son diminutos.
    log_evidence_probability = _logsumexp(tuple(log_weighted_likelihoods.values()))
    if log_evidence_probability == float("-inf"):
        raise ValueError("La evidencia seleccionada tiene probabilidad cero en el modelo.")

    posterior_probabilities = {
        condition_id: exp(log_weighted - log_evidence_probability)
        if log_weighted != float("-inf")
        else 0.0
        for condition_id, log_weighted in log_weighted_likelihoods.items()
    }
    likelihoods = {
        condition_id: _safe_exp(log_likelihood)
        for condition_id, log_likelihood in log_likelihoods.items()
    }
    weighted_likelihoods = {
        condition_id: _safe_exp(log_weighted)
        for condition_id, log_weighted in log_weighted_likelihoods.items()
    }

    return DiagnosisResult(
        prior_probabilities=priors,
        likelihoods=likelihoods,
        weighted_likelihoods=weighted_likelihoods,
        posterior_probabilities=posterior_probabilities,
        evidence_probability=_safe_exp(log_evidence_probability),
        log_evidence_probability=log_evidence_probability,
    )


def _coerce_status(status: SymptomStatus, symptom_id: str) -> SymptomStatus:
    """Normaliza el estado recibido o informa de una entrada no reconocida."""
    try:
        return SymptomStatus(status)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Estado inválido para el síntoma {symptom_id}.") from error


def _safe_log(probability: float) -> float:
    """Representa log(0) como menos infinito para la suma de log-probabilidades."""
    return log(probability) if probability > 0.0 else float("-inf")


def _safe_exp(log_probability: float) -> float:
    """Evita desbordamientos y acepta el subdesbordamiento de una probabilidad."""
    if log_probability == float("-inf"):
        return 0.0
    try:
        return exp(log_probability)
    except OverflowError:
        return float("inf")


def _logsumexp(log_values: tuple[float, ...]) -> float:
    """Calcula log(sum(exp(x))) sin perder precisión en valores muy pequeños."""
    max_log_value = max(log_values)
    if max_log_value == float("-inf"):
        return max_log_value
    return max_log_value + log(sum(exp(value - max_log_value) for value in log_values))

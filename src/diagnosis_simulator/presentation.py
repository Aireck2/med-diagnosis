"""Componentes de presentación para la interfaz Streamlit."""

from __future__ import annotations

import streamlit as st

from diagnosis_simulator.domain import DiagnosisCatalog, SymptomStatus
from diagnosis_simulator.entropy import shannon_entropy
from diagnosis_simulator.inference import DiagnosisResult, infer_diagnoses

STATUS_LABELS = {
    SymptomStatus.UNKNOWN: "No evaluado",
    SymptomStatus.PRESENT: "Presente",
    SymptomStatus.ABSENT: "Ausente",
}
LABEL_STATUSES = {label: status for status, label in STATUS_LABELS.items()}


def render_simulator(catalog: DiagnosisCatalog) -> None:
    """Dibuja controles, cálculos y visualizaciones del simulador educativo."""
    catalog.validate()
    st.title("Simulador educativo de inferencia bayesiana")
    st.caption(
        "Modelo demostrativo con parámetros sintéticos plausibles. "
        "No está diseñado para diagnosticar ni orientar tratamientos."
    )
    st.warning(
        "Los resultados son una simulación educativa, no un diagnóstico médico. "
        "Los parámetros no son datos clínicos validados."
    )

    with st.expander("¿Cómo calcula el modelo?", expanded=False):
        st.markdown(
            "Se parte de una probabilidad previa para cada condición. Para cada "
            "síntoma evaluado, el modelo usa la probabilidad de observarlo dada "
            "cada condición (o su complemento si se marca como ausente). Multiplica "
            "esas verosimilitudes suponiendo independencia condicional y aplica "
            "Bayes. El denominador `P(E)` se obtiene sumando `P(E|D) × P(D)` "
            "sobre las condiciones. La entropía de Shannon resume la incertidumbre "
            "en bits. En la realidad, los síntomas pueden estar correlacionados."
        )

    evidence = _render_symptom_controls(catalog)
    try:
        result = infer_diagnoses(catalog, evidence)
    except ValueError as error:
        st.error(f"No se pudo calcular la inferencia: {error}")
        return

    _render_summary(result)
    _render_results(catalog, result)


def _render_symptom_controls(catalog: DiagnosisCatalog) -> dict[str, SymptomStatus]:
    """Presenta un selector de estado para cada síntoma del catálogo."""
    st.subheader("Selecciona el estado de los síntomas")
    st.caption("Los síntomas sin evaluar se excluyen del cálculo.")
    evidence: dict[str, SymptomStatus] = {}

    columns = st.columns(2)
    for index, symptom in enumerate(catalog.symptoms):
        with columns[index % len(columns)]:
            selected_label = st.selectbox(
                symptom.label,
                options=tuple(STATUS_LABELS.values()),
                index=0,
                key=f"symptom_{symptom.id}",
                help=symptom.description or None,
            )
        evidence[symptom.id] = LABEL_STATUSES[selected_label]
    return evidence


def _render_summary(result: DiagnosisResult) -> None:
    """Muestra probabilidad total y cambio en la incertidumbre diagnóstica."""
    prior_entropy = shannon_entropy(result.prior_probabilities)
    posterior_entropy = shannon_entropy(result.posterior_probabilities)
    probability_label = _format_probability(result.evidence_probability)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Probabilidad de la evidencia P(E)", probability_label)
    metric_columns[1].metric("Entropía previa", f"{prior_entropy:.3f} bits")
    metric_columns[2].metric("Entropía posterior", f"{posterior_entropy:.3f} bits")
    st.caption(
        f"Cambio de entropía: {posterior_entropy - prior_entropy:+.3f} bits. "
        "Un valor negativo indica menor incertidumbre en esta distribución."
    )


def _render_results(catalog: DiagnosisCatalog, result: DiagnosisResult) -> None:
    """Compara las distribuciones previa y posterior en tabla y gráfico."""
    rows = [
        {
            "Condición": condition.label,
            "Verosimilitud P(E|D)": result.likelihoods[condition.id],
            "Probabilidad previa": result.prior_probabilities[condition.id],
            "Probabilidad posterior": result.posterior_probabilities[condition.id],
        }
        for condition in catalog.conditions
    ]
    st.subheader("Probabilidades por condición")
    st.dataframe(rows, hide_index=True)
    st.subheader("Impacto de los síntomas")
    st.bar_chart(
        rows,
        x="Condición",
        y=["Probabilidad previa", "Probabilidad posterior"],
        horizontal=True,
    )
    with st.expander("Detalles del cálculo"):
        st.write(
            "La contribución de cada condición al denominador es "
            "`P(E|D) × P(D)`. La suma de esas contribuciones da `P(E)`."
        )
        detail_rows = [
            {
                "Condición": condition.label,
                "P(E|D) × P(D)": result.weighted_likelihoods[condition.id],
            }
            for condition in catalog.conditions
        ]
        st.dataframe(detail_rows, hide_index=True)
        if result.evidence_probability == 0.0:
            st.caption(
                "P(E) es menor que el rango representable directamente. "
                f"log P(E) = {result.log_evidence_probability:.3f}."
            )


def _format_probability(probability: float) -> str:
    """Formatea probabilidades pequeñas sin ocultar subdesbordamientos."""
    if probability == 0.0:
        return "≈ 0 (muy pequeña)"
    if probability < 0.001:
        return f"{probability:.2e}"
    return f"{probability:.4f} ({probability:.1%})"

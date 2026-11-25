"""Plotly figure builders for Bayesian probability evolution and natural frequencies."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from med_diagnosis.domain.models import NaturalFrequencyBreakdown, SimulationTrace
from med_diagnosis.ui.formatting import (
    format_fraction,
    format_integer,
    format_percentage,
)


def _stage_labels(trace: SimulationTrace) -> list[str]:
    """Generate human-readable x-axis labels for t = 0 .. n."""
    labels = [f"t=0: {trace.scenario.stratum_name}"]
    for step in trace.steps:
        obs_str = step.application.observation.value
        labels.append(f"t={step.step_index}: {step.application.test.name} ({obs_str})")
    return labels


def build_bayesian_evolution_chart(
    trace: SimulationTrace,
    permuted_trace: SimulationTrace | None = None,
) -> go.Figure:
    """Build an interactive curve of P_t(E) and P_t(E^c) over discrete stages t = 0..n."""
    x_labels = _stage_labels(trace)
    p_disease_pct = [float(p) * 100.0 for p in trace.probabilities_by_stage]
    p_healthy_pct = [float(q) * 100.0 for q in trace.complements_by_stage]
    exact_fractions = [format_fraction(p) for p in trace.probabilities_by_stage]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(x_labels))),
            y=p_disease_pct,
            mode="lines+markers+text",
            name="Pₜ(E) — Presencia de patología",
            line={"color": "#0b6e4f", "width": 3.5},
            marker={"size": 11, "color": "#0b6e4f", "symbol": "circle"},
            text=[f"{v:.2f} %" for v in p_disease_pct],
            textposition="top center",
            customdata=list(zip(x_labels, exact_fractions, strict=True)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pₜ(E) = %{y:.2f} %<br>"
                "Fracción exacta: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(x_labels))),
            y=p_healthy_pct,
            mode="lines+markers",
            name="Pₜ(Eᶜ) = 1 − Pₜ(E) — Ausencia (sano)",
            line={"color": "#6c757d", "width": 2.2, "dash": "dot"},
            marker={"size": 8, "color": "#6c757d", "symbol": "diamond"},
            customdata=x_labels,
            hovertemplate=("<b>%{customdata}</b><br>Pₜ(Eᶜ) = %{y:.2f} %<extra></extra>"),
        )
    )

    if permuted_trace is not None and len(permuted_trace.steps) == len(trace.steps):
        perm_labels = _stage_labels(permuted_trace)
        perm_pct = [float(p) * 100.0 for p in permuted_trace.probabilities_by_stage]
        perm_fracs = [format_fraction(p) for p in permuted_trace.probabilities_by_stage]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(perm_labels))),
                y=perm_pct,
                mode="lines+markers",
                name="Pₜ(E) — Orden invertido (conmutatividad)",
                line={"color": "#d97706", "width": 2.8, "dash": "dash"},
                marker={"size": 9, "color": "#d97706", "symbol": "square"},
                customdata=list(zip(perm_labels, perm_fracs, strict=True)),
                hovertemplate=(
                    "<b>Orden invertido — %{customdata[0]}</b><br>"
                    "Pₜ(E) = %{y:.2f} %<br>"
                    "Fracción exacta: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=50.0,
        line_dash="dash",
        line_color="#adb5bd",
        annotation_text="Equilibrio de creencia (50 %)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Evolución Secuencial de la Probabilidad Bayesiana Pₜ(E) y Pₜ(Eᶜ)",
        xaxis={
            "title": "Etapa discreta del protocolo diagnóstico (t)",
            "tickmode": "array",
            "tickvals": list(range(len(x_labels))),
            "ticktext": x_labels,
        },
        yaxis={
            "title": "Probabilidad (%)",
            "range": [-5, 108],
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 25, "t": 75, "b": 45},
        height=430,
        template="plotly_white",
    )
    return fig


def build_multi_stratum_comparison_chart(
    traces_by_stratum: Sequence[SimulationTrace],
) -> go.Figure:
    """Compare how the same diagnostic test chain evolves across all t = 0 strata."""
    palette = ["#0b6e4f", "#1d4ed8", "#b45309", "#7c3aed"]
    fig = go.Figure()

    if not traces_by_stratum:
        return fig

    ref_trace = traces_by_stratum[0]
    x_ticks = ["t=0 (Prevalencia P₀)"] + [
        f"t={s.step_index}: {s.application.test.name} ({s.application.observation.value})"
        for s in ref_trace.steps
    ]

    for idx, trace in enumerate(traces_by_stratum):
        color = palette[idx % len(palette)]
        y_vals = [float(p) * 100.0 for p in trace.probabilities_by_stage]
        fracs = [format_fraction(p) for p in trace.probabilities_by_stage]
        p0_str = format_percentage(trace.scenario.prior)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(x_ticks))),
                y=y_vals,
                mode="lines+markers+text",
                name=f"{trace.scenario.stratum_name} (P₀ = {p0_str})",
                line={"color": color, "width": 3.0},
                marker={"size": 10, "color": color},
                text=[f"{v:.2f} %" for v in y_vals],
                textposition="top center",
                customdata=fracs,
                hovertemplate=(
                    f"<b>{trace.scenario.stratum_name}</b><br>"
                    "Pₜ(E) = %{y:.2f} %<br>"
                    "Fracción exacta: %{customdata}<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=50.0,
        line_dash="dash",
        line_color="#adb5bd",
        annotation_text="Umbral 50 %",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Comparativa Simultánea entre Estratos Sintomáticos/Epidemiológicos (t = 0)",
        xaxis={
            "title": "Etapa de la secuencia diagnóstica",
            "tickmode": "array",
            "tickvals": list(range(len(x_ticks))),
            "ticktext": x_ticks,
        },
        yaxis={"title": "Probabilidad Posterior Pₜ(E) (%)", "range": [-5, 108]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 25, "t": 75, "b": 45},
        height=430,
        template="plotly_white",
    )
    return fig


def build_natural_frequencies_chart(
    breakdown: NaturalFrequencyBreakdown,
) -> go.Figure:
    """Build a visual bar comparison of natural frequencies over |Omega| = cohort_size."""
    fig = go.Figure()

    if breakdown.step_index == 0 or breakdown.observation is None:
        categories = ["Enfermos reales (E)", "Sanos reales (Eᶜ)"]
        counts = [breakdown.rounded_diseased, breakdown.rounded_healthy]
        colors = ["#dc2626", "#16a34a"]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                marker_color=colors,
                text=[format_integer(c) for c in counts],
                textposition="auto",
                name="Partición inicial t = 0",
            )
        )
        fig.update_layout(
            title=(
                f"Partición del Espacio Muestral en t = 0 "
                f"(|Ω| = {format_integer(breakdown.cohort_size)} personas)"
            ),
            yaxis={"title": "Número de personas"},
            template="plotly_white",
            height=380,
        )
        return fig

    tp = breakdown.rounded_true_positives or 0
    fn = breakdown.rounded_false_negatives or 0
    fp = breakdown.rounded_false_positives or 0
    tn = breakdown.rounded_true_negatives or 0

    fig.add_trace(
        go.Bar(
            name="Enfermos reales (E)",
            x=["Resultado Positivo (+)", "Resultado Negativo (-)"],
            y=[tp, fn],
            marker_color="#dc2626",
            text=[f"VP: {format_integer(tp)}", f"FN: {format_integer(fn)}"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Sanos reales (Eᶜ)",
            x=["Resultado Positivo (+)", "Resultado Negativo (-)"],
            y=[fp, tn],
            marker_color="#2563eb",
            text=[f"FP: {format_integer(fp)}", f"VN: {format_integer(tn)}"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=(
            f"Desglose de Frecuencias Naturales en {breakdown.stage_label} "
            f"(|Ω| = {format_integer(breakdown.cohort_size)})"
        ),
        barmode="group",
        yaxis={"title": "Número esperado de personas"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 25, "t": 75, "b": 40},
        height=390,
        template="plotly_white",
    )
    return fig

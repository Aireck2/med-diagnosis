"""Generador de demostraciones matemáticas LaTeX paso a paso desde t = 0 hasta t = n.

Produce sustituciones simbólicas, fraccionarias y porcentuales alineadas con
MateDiscreta.pdf Sección 1.3:
1. Estrato sintomático o epidemiológico inicial en t = 0: partición booleana
   Omega = E U E^c, probabilidad previa P_0(E), complemento P_0(E^c) y cohorte base.
2. Cada aplicación secuencial de prueba diagnóstica t >= 1:
   - Paso 1: Herencia de la probabilidad previa P_{t-1}(E) y P_{t-1}(E^c).
   - Paso 2: Verosimilitudes condicionales P(O_t | E) y P(O_t | E^c).
   - Paso 3: Ley de la Probabilidad Total P(O_t).
   - Paso 4: Regla de recurrencia del Teorema de Bayes P_t(E) = P(E | O_t).
"""

from __future__ import annotations

from dataclasses import dataclass

from med_diagnosis.domain.models import Observation, SimulationTrace
from med_diagnosis.ui.formatting import (
    format_fraction,
    format_integer,
    format_latex_fraction,
    format_percentage,
    format_signed_percentage_delta,
)


@dataclass(frozen=True)
class InitialStratumWalkthrough:
    """Demostración matemática de la etapa inicial t = 0 para el estrato seleccionado."""

    title: str
    clinical_description: str
    partition_latex: str
    natural_frequency_summary: str


@dataclass(frozen=True)
class StepMathWalkthrough:
    """Desglose algebraico en 4 pasos para una transición individual de t - 1 a t."""

    step_index: int
    title: str
    biological_mechanism: str
    prior_inheritance_latex: str
    likelihood_latex: str
    total_probability_latex: str
    bayes_substitution_latex: str
    natural_frequency_check: str


def build_initial_stratum_walkthrough(
    trace: SimulationTrace,
) -> InitialStratumWalkthrough:
    """Construye la tarjeta de demostración formal en t = 0 para el estrato sintomático."""
    scenario = trace.scenario
    p0_frac = format_latex_fraction(scenario.prior)
    p0_pct = format_percentage(scenario.prior).replace("%", "\\%")
    q0_frac = format_latex_fraction(scenario.complement)
    q0_pct = format_percentage(scenario.complement).replace("%", "\\%")

    # Expresión de la partición disjunta sobre el espacio muestral finito Omega
    partition_latex = (
        f"\\Omega = E \\cup E^c, \\quad E \\cap E^c = \\emptyset \\implies "
        f"P_0(E) = {p0_frac} = {p0_pct}, \\quad "
        f"P_0(E^c) = 1 - P_0(E) = {q0_frac} = {q0_pct}"
    )

    f0 = trace.frequencies[0]
    summary = (
        f"En una cohorte finita de |Ω| = {format_integer(f0.cohort_size)} individuos del "
        f"estrato «{scenario.stratum_name}», la partición inicial antes de aplicar pruebas "
        f"espera {format_integer(f0.rounded_diseased)} personas con la patología (E) y "
        f"{format_integer(f0.rounded_healthy)} personas sanas (Eᶜ)."
    )

    return InitialStratumWalkthrough(
        title=f"Etapa Base t = 0 — Estrato Sintomático/Epidemiológico: {scenario.stratum_name}",
        clinical_description=scenario.clinical_description,
        partition_latex=partition_latex,
        natural_frequency_summary=summary,
    )


def build_step_walkthroughs(
    trace: SimulationTrace,
) -> tuple[StepMathWalkthrough, ...]:
    """Construye las demostraciones LaTeX paso a paso para cada prueba aplicada t >= 1."""
    cards: list[StepMathWalkthrough] = []

    for step, freq in zip(trace.steps, trace.frequencies[1:], strict=True):
        t = step.step_index
        test = step.application.test
        obs = step.application.observation
        obs_sym = f"+_{t}" if obs is Observation.POSITIVE else f"-_{t}"

        # Paso 1: Probabilidad a priori heredada de la etapa t - 1
        prior_ltx = format_latex_fraction(step.prior)
        prior_pct = format_percentage(step.prior).replace("%", "\\%")
        prior_comp_ltx = format_latex_fraction(step.prior_complement)
        prior_comp_pct = format_percentage(step.prior_complement).replace("%", "\\%")

        prior_inheritance_latex = (
            f"P_{{{t - 1}}}(E) = {prior_ltx} \\approx {prior_pct}, \\quad "
            f"P_{{{t - 1}}}(E^c) = 1 - P_{{{t - 1}}}(E) = "
            f"{prior_comp_ltx} \\approx {prior_comp_pct}"
        )

        # Paso 2: Verosimilitudes condicionales según resultado positivo o negativo
        like_e_ltx = format_latex_fraction(step.likelihood_disease)
        like_e_pct = format_percentage(step.likelihood_disease).replace("%", "\\%")
        like_ec_ltx = format_latex_fraction(step.likelihood_healthy)
        like_ec_pct = format_percentage(step.likelihood_healthy).replace("%", "\\%")

        if obs is Observation.POSITIVE:
            e_rule = "\\text{Sensibilidad}"
            ec_rule = "1 - \\text{Especificidad (Falsos Positivos)}"
        else:
            e_rule = "1 - \\text{Sensibilidad (Falsos Negativos)}"
            ec_rule = "\\text{Especificidad (Verdaderos Negativos)}"

        likelihood_latex = (
            f"P({obs_sym} \\mid E) = {e_rule} = {like_e_ltx} = {like_e_pct}, \\quad "
            f"P({obs_sym} \\mid E^c) = {ec_rule} = {like_ec_ltx} = {like_ec_pct}"
        )

        # Paso 3: Ley de la Probabilidad Total P(O_t)
        joint_e_ltx = format_latex_fraction(step.joint_disease)
        joint_ec_ltx = format_latex_fraction(step.joint_healthy)
        marginal_ltx = format_latex_fraction(step.marginal_observation_prob)
        marginal_pct = format_percentage(step.marginal_observation_prob).replace("%", "\\%")

        total_probability_latex = (
            f"P({obs_sym}) = P({obs_sym} \\mid E)\\cdot P_{{{t - 1}}}(E) + "
            f"P({obs_sym} \\mid E^c)\\cdot P_{{{t - 1}}}(E^c) = "
            f"\\left({like_e_ltx} \\cdot {prior_ltx}\\right) + "
            f"\\left({like_ec_ltx} \\cdot {prior_comp_ltx}\\right) = "
            f"{joint_e_ltx} + {joint_ec_ltx} = {marginal_ltx} \\approx {marginal_pct}"
        )

        # Paso 4: Sustitución en la fórmula de recurrencia del Teorema de Bayes
        post_ltx = format_latex_fraction(step.posterior)
        post_pct = format_percentage(step.posterior).replace("%", "\\%")
        delta_str = format_signed_percentage_delta(step.belief_delta)

        bayes_substitution_latex = (
            f"P_{t}(E) = P(E \\mid {obs_sym}) = "
            f"\\frac{{P({obs_sym} \\mid E)\\cdot P_{{{t - 1}}}(E)}}{{P({obs_sym})}} = "
            f"\\frac{{{joint_e_ltx}}}{{{marginal_ltx}}} = "
            f"{post_ltx} \\approx {post_pct} \\quad (\\Delta P_{t} = \\text{{{delta_str}}})"
        )

        # Resumen de verificación intuitiva mediante conteo de frecuencias naturales
        post_frac_str = format_fraction(step.posterior)
        post_pct_str = format_percentage(step.posterior)
        if obs is Observation.POSITIVE:
            tp = freq.rounded_true_positives or 0
            fp = freq.rounded_false_positives or 0
            nat_check = (
                f"Cociente en frecuencias naturales ({format_integer(freq.cohort_size)} personas): "
                f"Verdaderos Positivos ≈ {format_integer(tp)}, "
                f"Falsos Positivos ≈ {format_integer(fp)} → "
                f"P_{t}(E) = {post_frac_str} ({post_pct_str})."
            )
        else:
            fn = freq.rounded_false_negatives or 0
            tn = freq.rounded_true_negatives or 0
            nat_check = (
                f"Cociente en frecuencias naturales ({format_integer(freq.cohort_size)} personas): "
                f"Falsos Negativos ≈ {format_integer(fn)}, "
                f"Verdaderos Negativos ≈ {format_integer(tn)} → "
                f"P_{t}(E) = {post_frac_str} ({post_pct_str})."
            )

        cards.append(
            StepMathWalkthrough(
                step_index=t,
                title=f"Etapa t = {t} — {test.name} → Resultado: {obs.label_es}",
                biological_mechanism=test.biological_mechanism,
                prior_inheritance_latex=prior_inheritance_latex,
                likelihood_latex=likelihood_latex,
                total_probability_latex=total_probability_latex,
                bayes_substitution_latex=bayes_substitution_latex,
                natural_frequency_check=nat_check,
            )
        )

    return tuple(cards)

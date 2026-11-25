"""Step-by-step LaTeX mathematical demonstration generator from t = 0 to t = n.

Produces symbolic, fractional, and percentage substitutions for:
1. Initial symptomatic/epidemiological stratum at t = 0: boolean partition
   Omega = E U E^c, prior P_0(E), complement P_0(E^c), initial odds, and cohort split.
2. Each sequential diagnostic test application t >= 1: prior inheritance,
   conditional likelihoods P(O_t | E) and P(O_t | E^c), Law of Total Probability
   P(O_t), Bayes recurrence ratio, and equivalent Likelihood Ratio update.
"""

from __future__ import annotations

from dataclasses import dataclass

from med_diagnosis.domain.models import Observation, SimulationTrace
from med_diagnosis.ui.formatting import (
    format_fraction,
    format_integer,
    format_latex_fraction,
    format_likelihood_ratio,
    format_percentage,
    format_signed_percentage_delta,
)


@dataclass(frozen=True)
class InitialStratumWalkthrough:
    """Mathematical demonstration of the selected t = 0 symptomatic stratum."""

    title: str
    clinical_description: str
    partition_latex: str
    odds_latex: str
    natural_frequency_summary: str


@dataclass(frozen=True)
class StepMathWalkthrough:
    """Complete algebraic walkthrough for a single test transition t - 1 -> t."""

    step_index: int
    title: str
    biological_mechanism: str
    prior_inheritance_latex: str
    likelihood_latex: str
    total_probability_latex: str
    bayes_substitution_latex: str
    odds_check_latex: str
    natural_frequency_check: str


def build_initial_stratum_walkthrough(
    trace: SimulationTrace,
) -> InitialStratumWalkthrough:
    """Build the t = 0 formal demonstration card for the selected symptomatic stratum."""
    scenario = trace.scenario
    p0_frac = format_latex_fraction(scenario.prior)
    p0_pct = format_percentage(scenario.prior).replace("%", "\\%")
    q0_frac = format_latex_fraction(scenario.complement)
    q0_pct = format_percentage(scenario.complement).replace("%", "\\%")

    partition_latex = (
        f"\\Omega = E \\cup E^c, \\quad E \\cap E^c = \\emptyset \\implies "
        f"P_0(E) = {p0_frac} = {p0_pct}, \\quad "
        f"P_0(E^c) = 1 - P_0(E) = {q0_frac} = {q0_pct}"
    )

    odds0_frac = format_latex_fraction(scenario.initial_odds)
    odds0_dec = f"{float(scenario.initial_odds):.4f}"
    odds_latex = (
        f"\\text{{Odds}}_0(E) = \\frac{{P_0(E)}}{{P_0(E^c)}} = "
        f"\\frac{{{p0_frac}}}{{{q0_frac}}} = {odds0_frac} \\approx {odds0_dec}"
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
        odds_latex=odds_latex,
        natural_frequency_summary=summary,
    )


def build_step_walkthroughs(
    trace: SimulationTrace,
) -> tuple[StepMathWalkthrough, ...]:
    """Build step-by-step LaTeX demonstrations for each applied diagnostic test t >= 1."""
    cards: list[StepMathWalkthrough] = []

    for step, freq in zip(trace.steps, trace.frequencies[1:], strict=True):
        t = step.step_index
        test = step.application.test
        obs = step.application.observation
        obs_sym = f"+_{t}" if obs is Observation.POSITIVE else f"-_{t}"

        prior_ltx = format_latex_fraction(step.prior)
        prior_pct = format_percentage(step.prior).replace("%", "\\%")
        prior_comp_ltx = format_latex_fraction(step.prior_complement)
        prior_comp_pct = format_percentage(step.prior_complement).replace("%", "\\%")

        prior_inheritance_latex = (
            f"P_{{{t - 1}}}(E) = {prior_ltx} \\approx {prior_pct}, \\quad "
            f"P_{{{t - 1}}}(E^c) = 1 - P_{{{t - 1}}}(E) = "
            f"{prior_comp_ltx} \\approx {prior_comp_pct}"
        )

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

        post_ltx = format_latex_fraction(step.posterior)
        post_pct = format_percentage(step.posterior).replace("%", "\\%")
        delta_str = format_signed_percentage_delta(step.belief_delta)

        bayes_substitution_latex = (
            f"P_{t}(E) = P(E \\mid {obs_sym}) = "
            f"\\frac{{P({obs_sym} \\mid E)\\cdot P_{{{t - 1}}}(E)}}{{P({obs_sym})}} = "
            f"\\frac{{{joint_e_ltx}}}{{{marginal_ltx}}} = "
            f"{post_ltx} \\approx {post_pct} \\quad (\\Delta P_{t} = \\text{{{delta_str}}})"
        )

        lr_ltx = format_latex_fraction(step.likelihood_ratio)
        lr_dec = format_likelihood_ratio(step.likelihood_ratio)
        cum_lr_dec = format_likelihood_ratio(step.cumulative_likelihood_ratio)
        post_odds_ltx = format_latex_fraction(step.posterior_odds)

        odds_check_latex = (
            f"LR({obs_sym}) = \\frac{{P({obs_sym} \\mid E)}}{{P({obs_sym} \\mid E^c)}} = "
            f"{lr_ltx} \\approx {lr_dec}, \\quad "
            f"\\text{{Odds}}_{t}(E) = \\text{{Odds}}_0(E) \\cdot \\prod_{{k=1}}^{{{t}}} LR(O_k) = "
            f"{post_odds_ltx} \\quad (LR_{{\\text{{acum}}}} \\approx {cum_lr_dec})"
        )

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
                odds_check_latex=odds_check_latex,
                natural_frequency_check=nat_check,
            )
        )

    return tuple(cards)

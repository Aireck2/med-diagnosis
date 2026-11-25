"""Motor racional exacto de Inferencia Bayesiana Secuencial.

Implementa (MateDiscreta.pdf Sección 1.3):
1. Verosimilitudes condicionales P(O_t | E) y P(O_t | E^c).
2. Ley de la Probabilidad Total P(O_t) = P(O_t | E) P_{t-1}(E) + P(O_t | E^c) P_{t-1}(E^c).
3. Relación de recurrencia bayesiana:
       P_t(E) = P(O_t | E) P_{t-1}(E) / P(O_t)
4. Constructores de secuencias en modo guiado + flexible y comparativas multi-estrato.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from med_diagnosis.domain.frequencies import compute_natural_frequencies
from med_diagnosis.domain.models import (
    BayesianStepResult,
    DiagnosticTestSpec,
    Observation,
    PathologyProfile,
    PrevalenceScenario,
    SimulationTrace,
    TestApplication,
)


def likelihood_given_disease(test: DiagnosticTestSpec, obs: Observation) -> Fraction:
    """Retorna P(O_t | E): sensibilidad si O_t es '+', o 1 - sensibilidad si es '-'."""
    return test.sensitivity if obs is Observation.POSITIVE else test.false_negative_rate


def likelihood_given_healthy(test: DiagnosticTestSpec, obs: Observation) -> Fraction:
    """Retorna P(O_t | E^c): 1 - especificidad si O_t es '+', o especificidad si es '-'."""
    return test.false_positive_rate if obs is Observation.POSITIVE else test.specificity


def apply_bayes_step(
    step_index: int,
    prior: Fraction,
    application: TestApplication,
) -> BayesianStepResult:
    """Calcula una transición secuencial de t - 1 a t utilizando fracciones exactas."""
    if step_index < 1:
        raise ValueError(f"step_index must be >= 1, got {step_index}")
    if not (Fraction(0) <= prior <= Fraction(1)):
        raise ValueError(f"Prior probability must be in [0, 1], got {prior!r}")

    test = application.test
    obs = application.observation

    # Probabilidad del complemento sano en la etapa previa: P_{t-1}(E^c) = 1 - P_{t-1}(E)
    prior_comp = Fraction(1) - prior
    p_obs_e = likelihood_given_disease(test, obs)
    p_obs_ec = likelihood_given_healthy(test, obs)

    # Términos conjuntos y Ley de la Probabilidad Total para P(O_t)
    joint_disease = p_obs_e * prior
    joint_healthy = p_obs_ec * prior_comp
    marginal_obs = joint_disease + joint_healthy
    if marginal_obs == 0:
        raise ZeroDivisionError("Marginal observation probability P(O_t) is zero.")

    # Actualización de la creencia mediante la regla de recurrencia del Teorema de Bayes
    posterior = joint_disease / marginal_obs
    posterior_comp = Fraction(1) - posterior

    return BayesianStepResult(
        step_index=step_index,
        application=application,
        prior=prior,
        prior_complement=prior_comp,
        likelihood_disease=p_obs_e,
        likelihood_healthy=p_obs_ec,
        joint_disease=joint_disease,
        joint_healthy=joint_healthy,
        marginal_observation_prob=marginal_obs,
        posterior=posterior,
        posterior_complement=posterior_comp,
        belief_delta=posterior - prior,
    )


def run_bayesian_sequence(
    pathology: PathologyProfile,
    scenario: PrevalenceScenario,
    applications: Sequence[TestApplication],
    cohort_size: int = 10_000,
) -> SimulationTrace:
    """Ejecuta una cadena secuencial de pruebas diagnósticas desde t = 0 hasta t = n."""
    current_prior = scenario.prior
    steps: list[BayesianStepResult] = []

    # En cada transición, la probabilidad posterior previa pasa a ser la nueva probabilidad a priori
    for idx, app in enumerate(applications, start=1):
        step_result = apply_bayes_step(
            step_index=idx,
            prior=current_prior,
            application=app,
        )
        steps.append(step_result)
        current_prior = step_result.posterior

    step_tuple = tuple(steps)
    frequencies = compute_natural_frequencies(
        initial_prior=scenario.prior,
        steps=step_tuple,
        cohort_size=cohort_size,
        initial_label=f"Etapa t = 0 ({scenario.stratum_name})",
    )

    return SimulationTrace(
        pathology_slug=pathology.slug,
        pathology_name=pathology.name,
        scenario=scenario,
        steps=step_tuple,
        frequencies=frequencies,
        cohort_size=cohort_size,
    )


def build_guided_sequence(
    tests: Sequence[DiagnosticTestSpec],
    observations: Sequence[Observation] | None = None,
) -> tuple[TestApplication, ...]:
    """Construye la secuencia clínica guiada por defecto (1 -> 2 -> ... -> k)."""
    ordered = sorted(tests, key=lambda t: t.default_step_order)
    if observations is None:
        obs_list = [Observation.POSITIVE] * len(ordered)
    else:
        if len(observations) != len(ordered):
            raise ValueError(f"Expected {len(ordered)} observations, got {len(observations)}")
        obs_list = list(observations)

    return tuple(
        TestApplication(test=t, observation=o) for t, o in zip(ordered, obs_list, strict=True)
    )


def move_application(
    applications: Sequence[TestApplication],
    index: int,
    direction: int,
) -> tuple[TestApplication, ...]:
    """Desplaza una etapa hacia arriba (direction = -1) o hacia abajo (direction = +1)."""
    items = list(applications)
    target = index + direction
    if not (0 <= index < len(items)) or not (0 <= target < len(items)):
        return tuple(items)
    items[index], items[target] = items[target], items[index]
    return tuple(items)


def reverse_sequence(
    applications: Sequence[TestApplication],
) -> tuple[TestApplication, ...]:
    """Retorna la secuencia de pruebas en orden invertido para demostrar conmutatividad."""
    return tuple(reversed(applications))


def compare_across_scenarios(
    pathology: PathologyProfile,
    applications: Sequence[TestApplication],
    cohort_size: int = 10_000,
) -> tuple[SimulationTrace, ...]:
    """Evalúa la misma secuencia diagnóstica sobre todos los estratos sintomáticos t = 0."""
    return tuple(
        run_bayesian_sequence(
            pathology=pathology,
            scenario=scenario,
            applications=applications,
            cohort_size=cohort_size,
        )
        for scenario in pathology.scenarios
    )

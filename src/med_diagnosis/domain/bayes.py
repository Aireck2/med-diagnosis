"""Exact rational Sequential Bayesian Inference Engine.

Implements:
1. Conditional likelihoods P(O_t | E) and P(O_t | E^c).
2. Law of Total Probability P(O_t) = P(O_t | E) P_{t-1}(E) + P(O_t | E^c) P_{t-1}(E^c).
3. Bayesian recurrence relation:
       P_t(E) = P(O_t | E) P_{t-1}(E) / P(O_t)
4. Equivalent odds & likelihood-ratio recurrence:
       Odds_t(E) = Odds_0(E) * prod_{k=1}^t LR(O_k)
5. Guided + Flexible sequence builders and multi-stratum comparisons.
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
    """Return P(O_t | E): sensitivity if O_t is '+', or 1 - sensitivity if '-'."""
    return test.sensitivity if obs is Observation.POSITIVE else test.false_negative_rate


def likelihood_given_healthy(test: DiagnosticTestSpec, obs: Observation) -> Fraction:
    """Return P(O_t | E^c): 1 - specificity if O_t is '+', or specificity if '-'."""
    return test.false_positive_rate if obs is Observation.POSITIVE else test.specificity


def apply_bayes_step(
    step_index: int,
    prior: Fraction,
    application: TestApplication,
    cumulative_lr_prev: Fraction = Fraction(1, 1),
) -> BayesianStepResult:
    """Compute a single sequential transition from t - 1 to t using exact Fractions."""
    if step_index < 1:
        raise ValueError(f"step_index must be >= 1, got {step_index}")
    if not (Fraction(0) <= prior <= Fraction(1)):
        raise ValueError(f"Prior probability must be in [0, 1], got {prior!r}")

    test = application.test
    obs = application.observation

    prior_comp = Fraction(1) - prior
    p_obs_e = likelihood_given_disease(test, obs)
    p_obs_ec = likelihood_given_healthy(test, obs)

    joint_disease = p_obs_e * prior
    joint_healthy = p_obs_ec * prior_comp
    marginal_obs = joint_disease + joint_healthy
    if marginal_obs == 0:
        raise ZeroDivisionError("Marginal observation probability P(O_t) is zero.")

    posterior = joint_disease / marginal_obs
    posterior_comp = Fraction(1) - posterior

    if p_obs_ec == 0:
        raise ZeroDivisionError("Likelihood given healthy P(O_t | E^c) is zero.")
    step_lr = p_obs_e / p_obs_ec
    cumulative_lr = cumulative_lr_prev * step_lr

    if prior_comp == 0 or posterior_comp == 0:
        prior_odds = Fraction(0, 1)
        posterior_odds = Fraction(0, 1)
    else:
        prior_odds = prior / prior_comp
        posterior_odds = posterior / posterior_comp

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
        likelihood_ratio=step_lr,
        cumulative_likelihood_ratio=cumulative_lr,
        prior_odds=prior_odds,
        posterior_odds=posterior_odds,
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
    """Execute a sequential chain of diagnostic tests from t = 0 through t = n."""
    current_prior = scenario.prior
    cumulative_lr = Fraction(1, 1)
    steps: list[BayesianStepResult] = []

    for idx, app in enumerate(applications, start=1):
        step_result = apply_bayes_step(
            step_index=idx,
            prior=current_prior,
            application=app,
            cumulative_lr_prev=cumulative_lr,
        )
        steps.append(step_result)
        current_prior = step_result.posterior
        cumulative_lr = step_result.cumulative_likelihood_ratio

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
    """Build the default guided clinical sequence (1 -> 2 -> ... -> k)."""
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
    """Move a step up (direction = -1) or down (direction = +1) in flexible mode."""
    items = list(applications)
    target = index + direction
    if not (0 <= index < len(items)) or not (0 <= target < len(items)):
        return tuple(items)
    items[index], items[target] = items[target], items[index]
    return tuple(items)


def reverse_sequence(
    applications: Sequence[TestApplication],
) -> tuple[TestApplication, ...]:
    """Return the reversed test application sequence to demonstrate commutativity."""
    return tuple(reversed(applications))


def compare_across_scenarios(
    pathology: PathologyProfile,
    applications: Sequence[TestApplication],
    cohort_size: int = 10_000,
) -> tuple[SimulationTrace, ...]:
    """Evaluate the exact same diagnostic sequence across all t = 0 symptomatic strata."""
    return tuple(
        run_bayesian_sequence(
            pathology=pathology,
            scenario=scenario,
            applications=applications,
            cohort_size=cohort_size,
        )
        for scenario in pathology.scenarios
    )

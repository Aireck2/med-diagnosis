"""Natural frequency cohort decomposition over the finite sample space Omega.

Translates abstract rational probabilities into concrete expected and rounded
whole-person counts over a cohort |Omega| = cohort_size (default 10,000),
preserving exact partition conservation invariants:
    E + E^c == cohort_size
    TP_t + FN_t == E
    FP_t + TN_t == E^c
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from med_diagnosis.domain.models import (
    BayesianStepResult,
    NaturalFrequencyBreakdown,
)


def _round_complement_pair(exact_first: Fraction, total_int: int) -> tuple[int, int]:
    """Round exact_first to nearest integer and set complement so sum == total_int."""
    first_int = max(0, min(total_int, int(round(float(exact_first)))))
    second_int = total_int - first_int
    return first_int, second_int


def compute_natural_frequencies(
    initial_prior: Fraction,
    steps: Sequence[BayesianStepResult],
    cohort_size: int = 10_000,
    initial_label: str = "Etapa t = 0 (Estrato Inicial)",
) -> tuple[NaturalFrequencyBreakdown, ...]:
    """Build natural frequency breakdowns for t = 0 and each sequential test step t >= 1."""
    if cohort_size <= 0:
        raise ValueError(f"Cohort size must be positive, got {cohort_size}")
    if not (Fraction(0) <= initial_prior <= Fraction(1)):
        raise ValueError(f"Initial prior must be in [0, 1], got {initial_prior!r}")

    n_frac = Fraction(cohort_size, 1)
    exact_e0 = initial_prior * n_frac
    exact_ec0 = (Fraction(1) - initial_prior) * n_frac
    rounded_e0, rounded_ec0 = _round_complement_pair(exact_e0, cohort_size)

    breakdowns: list[NaturalFrequencyBreakdown] = [
        NaturalFrequencyBreakdown(
            step_index=0,
            stage_label=initial_label,
            cohort_size=cohort_size,
            entering_prior=initial_prior,
            exact_diseased=exact_e0,
            exact_healthy=exact_ec0,
            rounded_diseased=rounded_e0,
            rounded_healthy=rounded_ec0,
        )
    ]

    for step in steps:
        test = step.application.test
        obs = step.application.observation
        entering_p = step.prior
        exact_e = entering_p * n_frac
        exact_ec = (Fraction(1) - entering_p) * n_frac
        rounded_e, rounded_ec = _round_complement_pair(exact_e, cohort_size)

        exact_tp = exact_e * test.sensitivity
        exact_fn = exact_e * test.false_negative_rate
        exact_fp = exact_ec * test.false_positive_rate
        exact_tn = exact_ec * test.specificity

        rounded_tp, rounded_fn = _round_complement_pair(exact_tp, rounded_e)
        rounded_fp, rounded_tn = _round_complement_pair(exact_fp, rounded_ec)

        breakdowns.append(
            NaturalFrequencyBreakdown(
                step_index=step.step_index,
                stage_label=f"Etapa t = {step.step_index}: {test.name} ({obs.value})",
                cohort_size=cohort_size,
                entering_prior=entering_p,
                exact_diseased=exact_e,
                exact_healthy=exact_ec,
                rounded_diseased=rounded_e,
                rounded_healthy=rounded_ec,
                observation=obs,
                test_name=test.name,
                exact_true_positives=exact_tp,
                exact_false_negatives=exact_fn,
                exact_false_positives=exact_fp,
                exact_true_negatives=exact_tn,
                rounded_true_positives=rounded_tp,
                rounded_false_negatives=rounded_fn,
                rounded_false_positives=rounded_fp,
                rounded_true_negatives=rounded_tn,
            )
        )

    return tuple(breakdowns)

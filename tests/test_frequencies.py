"""Tests for natural frequency cohort decomposition over finite sample space Omega."""

from __future__ import annotations

from fractions import Fraction

import pytest

from med_diagnosis.data.database import load_default_repository
from med_diagnosis.domain.bayes import build_guided_sequence, run_bayesian_sequence
from med_diagnosis.domain.frequencies import compute_natural_frequencies
from med_diagnosis.domain.models import Observation


def test_natural_frequencies_conserve_population_and_match_bayes_posterior() -> None:
    repo = load_default_repository()
    tb = repo.get_pathology("tb")
    scenario = tb.scenarios[0]  # 0.15%
    seq = build_guided_sequence(
        tb.tests,
        observations=(Observation.POSITIVE, Observation.NEGATIVE, Observation.POSITIVE),
    )

    trace = run_bayesian_sequence(
        pathology=tb,
        scenario=scenario,
        applications=seq,
        cohort_size=10_000,
    )

    assert len(trace.frequencies) == 4  # t=0 plus 3 test stages

    t0 = trace.frequencies[0]
    assert t0.step_index == 0
    assert t0.rounded_diseased == 15
    assert t0.rounded_healthy == 9985
    assert t0.rounded_diseased + t0.rounded_healthy == 10_000

    for step, freq in zip(trace.steps, trace.frequencies[1:], strict=True):
        assert freq.rounded_diseased + freq.rounded_healthy == 10_000
        assert (
            freq.rounded_true_positives is not None
            and freq.rounded_false_negatives is not None
            and freq.rounded_false_positives is not None
            and freq.rounded_true_negatives is not None
        )
        assert freq.rounded_true_positives + freq.rounded_false_negatives == freq.rounded_diseased
        assert freq.rounded_false_positives + freq.rounded_true_negatives == freq.rounded_healthy
        assert freq.exact_posterior_from_counts == step.posterior


def test_compute_natural_frequencies_rejects_non_positive_cohort() -> None:
    with pytest.raises(ValueError, match="Cohort size"):
        compute_natural_frequencies(
            initial_prior=Fraction(15, 1000),
            steps=(),
            cohort_size=0,
        )

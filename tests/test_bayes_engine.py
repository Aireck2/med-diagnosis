"""Tests for the exact rational Sequential Bayesian Inference Engine."""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest

from med_diagnosis.data.database import load_default_repository
from med_diagnosis.domain.bayes import (
    apply_bayes_step,
    build_guided_sequence,
    compare_across_scenarios,
    move_application,
    reverse_sequence,
    run_bayesian_sequence,
)
from med_diagnosis.domain.models import Observation, TestApplication


def test_partition_and_law_of_total_probability_hold_at_every_stage() -> None:
    repo = load_default_repository()
    covid = repo.get_pathology("covid19")
    scenario = covid.scenarios[1]  # Sintomático IRA (12.0%)
    sequence = build_guided_sequence(
        covid.tests,
        observations=(Observation.POSITIVE, Observation.NEGATIVE, Observation.POSITIVE),
    )

    trace = run_bayesian_sequence(
        pathology=covid,
        scenario=scenario,
        applications=sequence,
    )

    assert trace.initial_prior + scenario.complement == Fraction(1, 1)
    for step in trace.steps:
        assert step.prior + step.prior_complement == Fraction(1, 1)
        assert step.posterior + step.posterior_complement == Fraction(1, 1)
        assert step.joint_disease == step.likelihood_disease * step.prior
        assert step.joint_healthy == step.likelihood_healthy * step.prior_complement
        assert step.marginal_observation_prob == step.joint_disease + step.joint_healthy
        assert step.posterior == step.joint_disease / step.marginal_observation_prob


def test_commutativity_order_invariance_under_conditional_independence() -> None:
    repo = load_default_repository()
    for pathology in repo.list_pathologies():
        scenario = pathology.scenarios[0]
        base_apps = [
            TestApplication(pathology.tests[0], Observation.POSITIVE),
            TestApplication(pathology.tests[1], Observation.NEGATIVE),
            TestApplication(pathology.tests[2], Observation.POSITIVE),
        ]
        reference_trace = run_bayesian_sequence(
            pathology=pathology,
            scenario=scenario,
            applications=base_apps,
        )

        for perm in itertools.permutations(base_apps):
            perm_trace = run_bayesian_sequence(
                pathology=pathology,
                scenario=scenario,
                applications=perm,
            )
            assert perm_trace.final_posterior == reference_trace.final_posterior
            assert (
                perm_trace.cumulative_likelihood_ratio
                == reference_trace.cumulative_likelihood_ratio
            )


def test_odds_and_cumulative_likelihood_ratio_equivalence() -> None:
    repo = load_default_repository()
    dengue = repo.get_pathology("dengue")
    scenario = dengue.scenarios[1]  # Brote estacional 22.0%
    apps = build_guided_sequence(
        dengue.tests,
        observations=(Observation.POSITIVE, Observation.POSITIVE, Observation.NEGATIVE),
    )
    trace = run_bayesian_sequence(
        pathology=dengue,
        scenario=scenario,
        applications=apps,
    )

    initial_odds = scenario.initial_odds
    for step in trace.steps:
        expected_posterior_odds = initial_odds * step.cumulative_likelihood_ratio
        assert step.posterior_odds == expected_posterior_odds
        assert step.posterior / step.posterior_complement == expected_posterior_odds


def test_golden_tb_base_rate_fallacy_and_stepped_confirmation() -> None:
    repo = load_default_repository()
    tb = repo.get_pathology("tb")
    general_pop = tb.scenarios[0]  # 0.15% = 15/10000
    all_positive = build_guided_sequence(
        tb.tests,
        observations=(Observation.POSITIVE, Observation.POSITIVE, Observation.POSITIVE),
    )

    trace = run_bayesian_sequence(
        pathology=tb,
        scenario=general_pop,
        applications=all_positive,
    )

    # Step 1: Chest X-Ray (+) -> 0.87 * 0.0015 / (0.87 * 0.0015 + 0.11 * 0.9985)
    # = 1305 / (1305 + 109835) = 1305 / 111140 = 261 / 22228 (~1.1742%)
    assert trace.steps[0].posterior == Fraction(261, 22228)
    assert pytest.approx(float(trace.steps[0].posterior), rel=1e-6) == 0.011741947

    # Step 2: Sputum AFB (+) -> Sens 65%, Spec 98% (FP 2%) -> 16965 / 60899 (~27.8576%)
    assert trace.steps[1].posterior == Fraction(16965, 60899)
    assert pytest.approx(float(trace.steps[1].posterior), rel=1e-6) == 0.27857600

    # Step 3: GeneXpert MTB/RIF (+) -> Sens 89%, Spec 99% (FP 1%) -> 1509885 / 1553819 (~97.1725%)
    assert trace.steps[2].posterior == Fraction(1509885, 1553819)
    assert pytest.approx(float(trace.steps[2].posterior), rel=1e-6) == 0.97172515


def test_golden_covid19_asymptomatic_vs_symptomatic_and_discordant() -> None:
    repo = load_default_repository()
    covid = repo.get_pathology("covid19")
    asymptomatic = covid.scenarios[0]  # 1.5%
    antigen_pos = [TestApplication(covid.tests[0], Observation.POSITIVE)]

    trace_asym = run_bayesian_sequence(
        pathology=covid,
        scenario=asymptomatic,
        applications=antigen_pos,
    )
    # 0.73 * 0.015 / (0.73 * 0.015 + 0.009 * 0.985) = 10950 / 19815 = 730 / 1321 (~55.261%)
    assert trace_asym.final_posterior == Fraction(730, 1321)

    comparisons = compare_across_scenarios(pathology=covid, applications=antigen_pos)
    assert len(comparisons) == 3
    posteriors = [t.final_posterior for t in comparisons]
    assert posteriors[0] < posteriors[1] < posteriors[2]


def test_flexible_sequence_reordering_and_reversal_helpers() -> None:
    repo = load_default_repository()
    covid = repo.get_pathology("covid19")
    seq = build_guided_sequence(covid.tests)

    moved_down = move_application(seq, index=0, direction=1)
    assert [a.test.name for a in moved_down] == [
        covid.tests[1].name,
        covid.tests[0].name,
        covid.tests[2].name,
    ]

    reversed_seq = reverse_sequence(seq)
    assert [a.test.name for a in reversed_seq] == [
        covid.tests[2].name,
        covid.tests[1].name,
        covid.tests[0].name,
    ]


def test_apply_bayes_step_validates_prior_bounds() -> None:
    repo = load_default_repository()
    covid = repo.get_pathology("covid19")
    app = TestApplication(covid.tests[0], Observation.POSITIVE)

    with pytest.raises(ValueError, match="Prior probability"):
        apply_bayes_step(1, Fraction(12, 10), app, Fraction(1, 1))

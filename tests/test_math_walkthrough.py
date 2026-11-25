"""Tests for step-by-step LaTeX mathematical walkthroughs, formatting, charts, and presets."""

from __future__ import annotations

from fractions import Fraction

from med_diagnosis.data.database import load_default_repository
from med_diagnosis.domain.bayes import (
    build_guided_sequence,
    compare_across_scenarios,
    reverse_sequence,
    run_bayesian_sequence,
)
from med_diagnosis.domain.models import Observation
from med_diagnosis.ui.charts import (
    build_bayesian_evolution_chart,
    build_multi_stratum_comparison_chart,
    build_natural_frequencies_chart,
)
from med_diagnosis.ui.formatting import (
    format_fraction,
    format_integer,
    format_likelihood_ratio,
    format_percentage,
    format_signed_percentage_delta,
)
from med_diagnosis.ui.math_walkthrough import (
    build_initial_stratum_walkthrough,
    build_step_walkthroughs,
)
from med_diagnosis.ui.presets import list_clinical_presets


def test_formatting_helpers_produce_consistent_output() -> None:
    assert format_percentage(Fraction(15, 1000)) == "1.50 %"
    assert format_percentage(Fraction(15, 10000)) == "0.15 %"
    assert format_fraction(Fraction(730, 1321)) == "730 / 1321"
    assert format_likelihood_ratio(Fraction(730, 9)) == "81.11"
    assert format_integer(10_000) == "10 000"
    assert format_signed_percentage_delta(Fraction(5, 100)) == "+5.00 pp"
    assert format_signed_percentage_delta(Fraction(-3, 100)) == "-3.00 pp"


def test_math_walkthrough_includes_t0_stratum_and_every_selected_test() -> None:
    repo = load_default_repository()
    covid = repo.get_pathology("covid19")
    scenario = covid.scenarios[1]  # Sintomático IRA 12.0%
    apps = build_guided_sequence(
        covid.tests,
        observations=(Observation.POSITIVE, Observation.NEGATIVE, Observation.POSITIVE),
    )
    trace = run_bayesian_sequence(pathology=covid, scenario=scenario, applications=apps)

    t0_card = build_initial_stratum_walkthrough(trace)
    assert "Sintomático IRA" in t0_card.title
    assert "12.00 \\%" in t0_card.partition_latex
    assert "3}{25}" in t0_card.partition_latex
    assert "1 200" in t0_card.natural_frequency_summary

    step_cards = build_step_walkthroughs(trace)
    assert len(step_cards) == 3
    assert "Prueba Rápida de Antígenos" in step_cards[0].title
    assert "P(E \\mid +_1)" in step_cards[0].bayes_substitution_latex
    assert "P(-_2 \\mid E)" in step_cards[1].likelihood_latex
    assert "RT-PCR en Tiempo Real" in step_cards[2].title


def test_charts_build_valid_plotly_figures_without_shannon_entropy() -> None:
    repo = load_default_repository()
    tb = repo.get_pathology("tb")
    scenario = tb.scenarios[0]
    apps = build_guided_sequence(tb.tests)
    trace = run_bayesian_sequence(pathology=tb, scenario=scenario, applications=apps)
    permuted = run_bayesian_sequence(
        pathology=tb,
        scenario=scenario,
        applications=reverse_sequence(apps),
    )

    fig_evol = build_bayesian_evolution_chart(trace, permuted_trace=permuted)
    assert len(fig_evol.data) == 3  # P_t(E), P_t(E^c), and permuted P_t(E)

    multi_traces = compare_across_scenarios(pathology=tb, applications=apps)
    fig_multi = build_multi_stratum_comparison_chart(multi_traces)
    assert len(fig_multi.data) == 3  # 3 strata for TB

    fig_freq = build_natural_frequencies_chart(trace.frequencies[1])
    assert len(fig_freq.data) >= 2


def test_presets_resolve_against_database() -> None:
    repo = load_default_repository()
    presets = list_clinical_presets()
    assert len(presets) >= 4
    for preset in presets:
        pathology = repo.get_pathology(preset.pathology_slug)
        stratum_names = {s.stratum_name for s in pathology.scenarios}
        assert preset.stratum_name in stratum_names

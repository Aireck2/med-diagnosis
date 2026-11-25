"""Immutable domain models for sequential Bayesian diagnostic inference.

All probabilities and likelihood ratios are represented as exact rational
numbers (`fractions.Fraction`) over the finite boolean partition
Omega = E U E^c, with E cap E^c = empty set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction


class Observation(StrEnum):
    """Binary outcome produced by a diagnostic test at discrete stage t >= 1."""

    POSITIVE = "+"
    NEGATIVE = "-"

    @property
    def label_es(self) -> str:
        """Return neutral clinical Spanish label for UI display."""
        return "Positivo (+)" if self is Observation.POSITIVE else "Negativo (-)"


@dataclass(frozen=True)
class Citation:
    """Official epidemiological or clinical reference backing model parameters."""

    cite_key: str
    short_label: str
    full_reference: str
    pub_year: int
    doi_or_url: str = ""


@dataclass(frozen=True)
class PrevalenceScenario:
    """Initial clinical/symptomatic stratum at t = 0 defining P_0(E)."""

    scenario_id: int
    pathology_slug: str
    stratum_name: str
    clinical_description: str
    prior: Fraction
    display_order: int

    def __post_init__(self) -> None:
        if not (Fraction(0) < self.prior < Fraction(1)):
            raise ValueError(
                f"Prior probability P_0(E) must be strictly in (0, 1), got {self.prior!r}"
            )

    @property
    def complement(self) -> Fraction:
        """Return healthy complement probability P_0(E^c) = 1 - P_0(E)."""
        return Fraction(1) - self.prior

    @property
    def initial_odds(self) -> Fraction:
        """Return initial prior odds P_0(E) / P_0(E^c)."""
        return self.prior / self.complement


@dataclass(frozen=True)
class DiagnosticTestSpec:
    """Clinical performance specification of a diagnostic test."""

    test_id: int
    pathology_slug: str
    default_step_order: int
    name: str
    sensitivity: Fraction
    specificity: Fraction
    biological_mechanism: str

    def __post_init__(self) -> None:
        if not (Fraction(0) < self.sensitivity <= Fraction(1)):
            raise ValueError(f"Sensitivity P(+|E) must be in (0, 1], got {self.sensitivity!r}")
        if not (Fraction(0) < self.specificity <= Fraction(1)):
            raise ValueError(f"Specificity P(-|E^c) must be in (0, 1], got {self.specificity!r}")

    @property
    def false_positive_rate(self) -> Fraction:
        """Return P(+|E^c) = 1 - Specificity."""
        return Fraction(1) - self.specificity

    @property
    def false_negative_rate(self) -> Fraction:
        """Return P(-|E) = 1 - Sensitivity."""
        return Fraction(1) - self.sensitivity

    @property
    def positive_likelihood_ratio(self) -> Fraction | None:
        """Return LR(+) = P(+|E) / P(+|E^c), or None if specificity == 1."""
        fpr = self.false_positive_rate
        return None if fpr == 0 else self.sensitivity / fpr

    @property
    def negative_likelihood_ratio(self) -> Fraction:
        """Return LR(-) = P(-|E) / P(-|E^c)."""
        return self.false_negative_rate / self.specificity


@dataclass(frozen=True)
class PathologyProfile:
    """Complete epidemiological profile for a target pathology."""

    slug: str
    name: str
    target_population: str
    clinical_context: str
    scenarios: tuple[PrevalenceScenario, ...]
    tests: tuple[DiagnosticTestSpec, ...]
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class TestApplication:
    """A single test applied at stage t with a binary observation (+ or -)."""

    __test__ = False

    test: DiagnosticTestSpec
    observation: Observation


@dataclass(frozen=True)
class NaturalFrequencyBreakdown:
    """Natural frequency partition over a finite cohort |Omega| = cohort_size."""

    step_index: int
    stage_label: str
    cohort_size: int
    entering_prior: Fraction
    exact_diseased: Fraction
    exact_healthy: Fraction
    rounded_diseased: int
    rounded_healthy: int
    observation: Observation | None = None
    test_name: str | None = None
    exact_true_positives: Fraction | None = None
    exact_false_negatives: Fraction | None = None
    exact_false_positives: Fraction | None = None
    exact_true_negatives: Fraction | None = None
    rounded_true_positives: int | None = None
    rounded_false_negatives: int | None = None
    rounded_false_positives: int | None = None
    rounded_true_negatives: int | None = None

    @property
    def exact_posterior_from_counts(self) -> Fraction:
        """Compute posterior directly from exact subpopulation counts."""
        if self.observation is None:
            return self.entering_prior
        if self.observation is Observation.POSITIVE:
            assert self.exact_true_positives is not None
            assert self.exact_false_positives is not None
            total_pos = self.exact_true_positives + self.exact_false_positives
            return self.exact_true_positives / total_pos
        assert self.exact_false_negatives is not None
        assert self.exact_true_negatives is not None
        total_neg = self.exact_false_negatives + self.exact_true_negatives
        return self.exact_false_negatives / total_neg


@dataclass(frozen=True)
class BayesianStepResult:
    """Complete state of a single transition from t - 1 to t under Bayes' Theorem."""

    step_index: int
    application: TestApplication
    prior: Fraction
    prior_complement: Fraction
    likelihood_disease: Fraction
    likelihood_healthy: Fraction
    joint_disease: Fraction
    joint_healthy: Fraction
    marginal_observation_prob: Fraction
    likelihood_ratio: Fraction
    cumulative_likelihood_ratio: Fraction
    prior_odds: Fraction
    posterior_odds: Fraction
    posterior: Fraction
    posterior_complement: Fraction
    belief_delta: Fraction


@dataclass(frozen=True)
class SimulationTrace:
    """Complete sequential trajectory from t = 0 through t = n."""

    pathology_slug: str
    pathology_name: str
    scenario: PrevalenceScenario
    steps: tuple[BayesianStepResult, ...]
    frequencies: tuple[NaturalFrequencyBreakdown, ...]
    cohort_size: int

    @property
    def initial_prior(self) -> Fraction:
        """Return P_0(E) at t = 0."""
        return self.scenario.prior

    @property
    def final_posterior(self) -> Fraction:
        """Return P_n(E) after the last applied test (or P_0(E) if no tests)."""
        return self.steps[-1].posterior if self.steps else self.scenario.prior

    @property
    def final_complement(self) -> Fraction:
        """Return P_n(E^c) = 1 - P_n(E)."""
        return Fraction(1) - self.final_posterior

    @property
    def total_belief_shift(self) -> Fraction:
        """Return net change P_n(E) - P_0(E)."""
        return self.final_posterior - self.initial_prior

    @property
    def cumulative_likelihood_ratio(self) -> Fraction:
        """Return product of likelihood ratios across all applied steps."""
        return self.steps[-1].cumulative_likelihood_ratio if self.steps else Fraction(1, 1)

    @property
    def probabilities_by_stage(self) -> tuple[Fraction, ...]:
        """Return tuple (P_0(E), P_1(E), ..., P_n(E))."""
        return (self.scenario.prior, *(s.posterior for s in self.steps))

    @property
    def complements_by_stage(self) -> tuple[Fraction, ...]:
        """Return tuple (P_0(E^c), P_1(E^c), ..., P_n(E^c))."""
        return tuple(Fraction(1) - p for p in self.probabilities_by_stage)


@dataclass(frozen=True)
class ParameterMatrixRow:
    """Flat summary row matching Table 1 of MateDiscreta.pdf."""

    pathology_slug: str
    pathology_name: str
    target_population: str
    default_step_order: int
    test_name: str
    sensitivity: Fraction
    specificity: Fraction
    false_positive_rate: Fraction
    biological_mechanism: str
    priors_summary: str
    citations_summary: str

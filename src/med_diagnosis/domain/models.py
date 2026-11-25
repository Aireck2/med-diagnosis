"""Modelos inmutables de dominio para la inferencia diagnóstica bayesiana secuencial.

Todas las probabilidades se representan como números racionales exactos
(`fractions.Fraction`) sobre la partición booleana finita
Omega = E U E^c, con E cap E^c = conjunto vacío.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction


class Observation(StrEnum):
    """Resultado binario producido por una prueba diagnóstica en la etapa discreta t >= 1."""

    POSITIVE = "+"
    NEGATIVE = "-"

    @property
    def label_es(self) -> str:
        """Retorna la etiqueta clínica en español neutro para mostrar en la interfaz."""
        return "Positivo (+)" if self is Observation.POSITIVE else "Negativo (-)"


@dataclass(frozen=True)
class Citation:
    """Referencia epidemiológica o clínica oficial que respalda los parámetros del modelo."""

    cite_key: str
    short_label: str
    full_reference: str
    pub_year: int
    doi_or_url: str = ""


@dataclass(frozen=True)
class PrevalenceScenario:
    """Estrato clínico o sintomático inicial en t = 0 que define P_0(E)."""

    scenario_id: int
    pathology_slug: str
    stratum_name: str
    clinical_description: str
    prior: Fraction
    display_order: int

    def __post_init__(self) -> None:
        # Validar que la probabilidad previa P_0(E) pertenezca estrictamente a (0, 1)
        if not (Fraction(0) < self.prior < Fraction(1)):
            raise ValueError(
                f"Prior probability P_0(E) must be strictly in (0, 1), got {self.prior!r}"
            )

    @property
    def complement(self) -> Fraction:
        """Retorna la probabilidad del complemento sano P_0(E^c) = 1 - P_0(E)."""
        return Fraction(1) - self.prior


@dataclass(frozen=True)
class DiagnosticTestSpec:
    """Especificación de rendimiento clínico de una prueba diagnóstica."""

    test_id: int
    pathology_slug: str
    default_step_order: int
    name: str
    sensitivity: Fraction
    specificity: Fraction
    biological_mechanism: str

    def __post_init__(self) -> None:
        # Validar que sensibilidad y especificidad estén dentro de (0, 1]
        if not (Fraction(0) < self.sensitivity <= Fraction(1)):
            raise ValueError(f"Sensitivity P(+|E) must be in (0, 1], got {self.sensitivity!r}")
        if not (Fraction(0) < self.specificity <= Fraction(1)):
            raise ValueError(f"Specificity P(-|E^c) must be in (0, 1], got {self.specificity!r}")

    @property
    def false_positive_rate(self) -> Fraction:
        """Retorna la tasa de falsos positivos P(+|E^c) = 1 - Especificidad."""
        return Fraction(1) - self.specificity

    @property
    def false_negative_rate(self) -> Fraction:
        """Retorna la tasa de falsos negativos P(-|E) = 1 - Sensibilidad."""
        return Fraction(1) - self.sensitivity


@dataclass(frozen=True)
class PathologyProfile:
    """Perfil epidemiológico completo de una patología de estudio."""

    slug: str
    name: str
    target_population: str
    clinical_context: str
    scenarios: tuple[PrevalenceScenario, ...]
    tests: tuple[DiagnosticTestSpec, ...]
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class TestApplication:
    """Aplicación individual de una prueba en la etapa t con observación binaria (+ o -)."""

    __test__ = False

    test: DiagnosticTestSpec
    observation: Observation


@dataclass(frozen=True)
class NaturalFrequencyBreakdown:
    """Partición de frecuencias naturales sobre una cohorte finita |Omega| = cohort_size."""

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
        """Calcula la probabilidad posterior directamente del cociente exacto de subpoblaciones."""
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
    """Estado completo de una transición individual de t - 1 a t bajo el Teorema de Bayes."""

    step_index: int
    application: TestApplication
    prior: Fraction
    prior_complement: Fraction
    likelihood_disease: Fraction
    likelihood_healthy: Fraction
    joint_disease: Fraction
    joint_healthy: Fraction
    marginal_observation_prob: Fraction
    posterior: Fraction
    posterior_complement: Fraction
    belief_delta: Fraction


@dataclass(frozen=True)
class SimulationTrace:
    """Trayectoria secuencial completa desde t = 0 hasta t = n."""

    pathology_slug: str
    pathology_name: str
    scenario: PrevalenceScenario
    steps: tuple[BayesianStepResult, ...]
    frequencies: tuple[NaturalFrequencyBreakdown, ...]
    cohort_size: int

    @property
    def initial_prior(self) -> Fraction:
        """Retorna P_0(E) en la etapa inicial t = 0."""
        return self.scenario.prior

    @property
    def final_posterior(self) -> Fraction:
        """Retorna P_n(E) tras la última prueba aplicada (o P_0(E) si no hay pruebas)."""
        return self.steps[-1].posterior if self.steps else self.scenario.prior

    @property
    def final_complement(self) -> Fraction:
        """Retorna el complemento sano final P_n(E^c) = 1 - P_n(E)."""
        return Fraction(1) - self.final_posterior

    @property
    def total_belief_shift(self) -> Fraction:
        """Retorna el cambio neto acumulado P_n(E) - P_0(E)."""
        return self.final_posterior - self.initial_prior

    @property
    def probabilities_by_stage(self) -> tuple[Fraction, ...]:
        """Retorna la tupla ordenada (P_0(E), P_1(E), ..., P_n(E))."""
        return (self.scenario.prior, *(s.posterior for s in self.steps))

    @property
    def complements_by_stage(self) -> tuple[Fraction, ...]:
        """Retorna la tupla ordenada (P_0(E^c), P_1(E^c), ..., P_n(E^c))."""
        return tuple(Fraction(1) - p for p in self.probabilities_by_stage)


@dataclass(frozen=True)
class ParameterMatrixRow:
    """Fila resumen plana equivalente a la Tabla 1 de MateDiscreta.pdf."""

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

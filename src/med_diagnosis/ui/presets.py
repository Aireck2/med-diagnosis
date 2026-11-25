"""Casos clínicos demostrativos alineados con MateDiscreta.pdf Sección 2.2."""

from __future__ import annotations

from dataclasses import dataclass

from med_diagnosis.domain.models import Observation, PathologyProfile, TestApplication


@dataclass(frozen=True)
class ClinicalPreset:
    """Escenario clínico preconfigurado que demuestra un fenómeno clave del Teorema de Bayes."""

    key: str
    title: str
    pedagogical_focus: str
    pathology_slug: str
    stratum_name: str
    step_orders_and_observations: tuple[tuple[int, Observation], ...]

    def build_applications(
        self,
        pathology: PathologyProfile,
    ) -> tuple[TestApplication, ...]:
        """Resuelve el orden de las pruebas contra el catálogo diagnóstico de la patología."""
        tests_by_order = {t.default_step_order: t for t in pathology.tests}
        return tuple(
            TestApplication(test=tests_by_order[step_order], observation=obs)
            for step_order, obs in self.step_orders_and_observations
        )


CLINICAL_PRESETS: tuple[ClinicalPreset, ...] = (
    ClinicalPreset(
        key="tb_low_prevalence_single_vs_chain",
        title="1. Sesgo de Tasa Base (TB Población general 0.15 % — Protocolo completo [+, +, +])",
        pedagogical_focus=(
            "Demuestra por qué una primera prueba positiva aislada (Radiografía de Tórax +) en "
            "una población de muy baja prevalencia (0.15 %) apenas alcanza P₁(E) = 1.17 % debido "
            "al volumen de falsos positivos (1 098 sanos vs. 13 enfermos por cada 10 000 "
            "evaluados), y cómo encadenar Baciloscopía (+) y GeneXpert (+) eleva la certeza "
            "diagnóstica a P₃(E) = 97.17 %."
        ),
        pathology_slug="tb",
        stratum_name="Población general",
        step_orders_and_observations=(
            (1, Observation.POSITIVE),
            (2, Observation.POSITIVE),
            (3, Observation.POSITIVE),
        ),
    ),
    ClinicalPreset(
        key="covid19_asymptomatic_screening_confirmation",
        title="2. Tamizaje Asintomático y Confirmación Molecular (COVID-19 1.5 % — [+, +])",
        pedagogical_focus=(
            "Evidencia que en tamizaje asintomático (P₀ = 1.5 %), una Prueba Rápida de Antígenos "
            "positiva deja la probabilidad posterior en una zona de alta ambigüedad "
            "(P₁ = 55.26 %), requiriendo confirmación con RT-PCR en Tiempo Real (+) para "
            "alcanzar P₂ = 99.57 %."
        ),
        pathology_slug="covid19",
        stratum_name="Asintomático",
        step_orders_and_observations=(
            (1, Observation.POSITIVE),
            (3, Observation.POSITIVE),
        ),
    ),
    ClinicalPreset(
        key="covid19_symptomatic_discordant",
        title=(
            "3. Secuencia con Resultados Discordantes (COVID-19 Sintomático IRA 12.0 % — [+, -, +])"
        ),
        pedagogical_focus=(
            "Modela un paciente con Infección Respiratoria Aguda (P₀ = 12.0 %) que obtiene "
            "Antígeno (+), seguido de Serología IgM/IgG (-) y desempate molecular con RT-PCR (+), "
            "mostrando cómo el Teorema de Bayes pondera evidencia contradictoria."
        ),
        pathology_slug="covid19",
        stratum_name="Sintomático IRA",
        step_orders_and_observations=(
            (1, Observation.POSITIVE),
            (2, Observation.NEGATIVE),
            (3, Observation.POSITIVE),
        ),
    ),
    ClinicalPreset(
        key="dengue_outbreak_high_prevalence",
        title="4. Alta Prevalencia en Brote Estacional (Dengue 22.0 % — [+, +])",
        pedagogical_focus=(
            "Contrasta cómo una prevalencia a priori elevada en zona con brote estacional activo "
            "(P₀ = 22.0 %) permite que una Prueba Rápida Antígeno NS1 (+) alcance de inmediato "
            "P₁ = 91.47 %, y con ELISA IgM (+) llegue a P₂ = 99.13 %."
        ),
        pathology_slug="dengue",
        stratum_name="Brote estacional",
        step_orders_and_observations=(
            (1, Observation.POSITIVE),
            (2, Observation.POSITIVE),
        ),
    ),
)


def list_clinical_presets() -> tuple[ClinicalPreset, ...]:
    """Retorna todos los casos clínicos demostrativos preconfigurados."""
    return CLINICAL_PRESETS

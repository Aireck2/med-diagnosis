"""Utilidades de presentación, generadores LaTeX, casos demostrativos y gráficos Plotly."""

from med_diagnosis.ui.charts import (
    build_bayesian_evolution_chart,
    build_multi_stratum_comparison_chart,
    build_natural_frequencies_chart,
)
from med_diagnosis.ui.formatting import (
    format_fraction,
    format_integer,
    format_percentage,
    format_signed_percentage_delta,
)
from med_diagnosis.ui.math_walkthrough import (
    InitialStratumWalkthrough,
    StepMathWalkthrough,
    build_initial_stratum_walkthrough,
    build_step_walkthroughs,
)
from med_diagnosis.ui.presets import ClinicalPreset, list_clinical_presets

__all__ = [
    "ClinicalPreset",
    "InitialStratumWalkthrough",
    "StepMathWalkthrough",
    "build_bayesian_evolution_chart",
    "build_initial_stratum_walkthrough",
    "build_multi_stratum_comparison_chart",
    "build_natural_frequencies_chart",
    "build_step_walkthroughs",
    "format_fraction",
    "format_integer",
    "format_percentage",
    "format_signed_percentage_delta",
    "list_clinical_presets",
]

"""Catálogo didáctico con parámetros sintéticos de aspecto plausible."""

from diagnosis_simulator.domain import Condition, DiagnosisCatalog, Symptom

# Estos valores se inventaron para una demostración coherente: no representan
# prevalencias, tasas clínicas, ni estimaciones tomadas de una población real.
SAMPLE_CATALOG = DiagnosisCatalog(
    symptoms=(
        Symptom("cough", "Tos", "Tos reciente o persistente."),
        Symptom("fever", "Fiebre", "Sensación febril o temperatura elevada."),
        Symptom("nasal_congestion", "Congestión nasal", "Nariz tapada o con secreción."),
        Symptom("muscle_ache", "Dolor muscular", "Dolor muscular generalizado."),
        Symptom("loss_of_smell", "Pérdida del olfato", "Disminución nueva del olfato."),
        Symptom("sneezing", "Estornudos", "Estornudos frecuentes."),
    ),
    conditions=(
        Condition(
            "common_cold",
            "Resfriado común (simulado)",
            0.40,
            {
                "cough": 0.55,
                "fever": 0.12,
                "nasal_congestion": 0.78,
                "muscle_ache": 0.20,
                "loss_of_smell": 0.08,
                "sneezing": 0.62,
            },
        ),
        Condition(
            "influenza_like",
            "Síndrome gripal (simulado)",
            0.25,
            {
                "cough": 0.72,
                "fever": 0.78,
                "nasal_congestion": 0.40,
                "muscle_ache": 0.76,
                "loss_of_smell": 0.08,
                "sneezing": 0.22,
            },
        ),
        Condition(
            "covid_like",
            "Cuadro tipo COVID-19 (simulado)",
            0.20,
            {
                "cough": 0.65,
                "fever": 0.48,
                "nasal_congestion": 0.38,
                "muscle_ache": 0.55,
                "loss_of_smell": 0.32,
                "sneezing": 0.28,
            },
        ),
        Condition(
            "seasonal_allergy",
            "Alergia estacional (simulada)",
            0.15,
            {
                "cough": 0.20,
                "fever": 0.02,
                "nasal_congestion": 0.72,
                "muscle_ache": 0.05,
                "loss_of_smell": 0.06,
                "sneezing": 0.82,
            },
        ),
    ),
)


def get_sample_catalog() -> DiagnosisCatalog:
    """Devuelve el catálogo sintético inicial del simulador."""
    return SAMPLE_CATALOG

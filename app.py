"""Punto de entrada principal en Streamlit para el Simulador de Inferencia Bayesiana Secuencial.

Fiel a MateDiscreta.pdf (Secciones 1.1-1.4, Tabla 1, Sección 2),
enfocado 100 % en el Teorema de Bayes, aritmética racional exacta con Fraction,
demostraciones LaTeX paso a paso desde t = 0 hasta t = n, comparativas
multi-estrato y encadenamiento de pruebas en modo guiado + flexible.
"""

from __future__ import annotations

import streamlit as st

from med_diagnosis.data.database import (
    SqliteEpidemiologyRepository,
    load_default_repository,
)
from med_diagnosis.domain.bayes import (
    compare_across_scenarios,
    move_application,
    reverse_sequence,
    run_bayesian_sequence,
)
from med_diagnosis.domain.models import (
    Observation,
    PathologyProfile,
    TestApplication,
)
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
    build_initial_stratum_walkthrough,
    build_step_walkthroughs,
)
from med_diagnosis.ui.presets import list_clinical_presets


@st.cache_resource
def get_repository() -> SqliteEpidemiologyRepository:
    """Carga el repositorio epidemiológico SQLite de solo lectura una sola vez por proceso."""
    return load_default_repository()


def _init_sequence_state(pathology: PathologyProfile) -> None:
    """Inicializa o reinicia el estado de la secuencia en sesión cuando cambia la patología."""
    current_slug = st.session_state.get("active_pathology_slug")
    if current_slug != pathology.slug or "sequence_items" not in st.session_state:
        st.session_state["active_pathology_slug"] = pathology.slug
        st.session_state["sequence_items"] = [
            (t.default_step_order, Observation.POSITIVE.value) for t in pathology.tests
        ]


def _resolve_applications(
    pathology: PathologyProfile,
    raw_items: list[tuple[int, str]],
) -> tuple[TestApplication, ...]:
    """Convierte las tuplas (step_order, obs_value) de sesión en objetos TestApplication."""
    tests_by_order = {t.default_step_order: t for t in pathology.tests}
    resolved: list[TestApplication] = []
    for step_order, obs_val in raw_items:
        if step_order in tests_by_order:
            obs = (
                Observation.POSITIVE
                if obs_val == Observation.POSITIVE.value
                else Observation.NEGATIVE
            )
            resolved.append(TestApplication(test=tests_by_order[step_order], observation=obs))
    return tuple(resolved)


def main() -> None:
    """Renderiza el Simulador de Inferencia Bayesiana Secuencial interactivo."""
    st.set_page_config(
        page_title="Simulador de Inferencia Bayesiana Secuencial",
        page_icon="🧬",
        layout="wide",
    )

    repo = get_repository()
    pathologies = repo.list_pathologies()
    pathologies_by_slug = {p.slug: p for p in pathologies}
    presets = list_clinical_presets()

    st.title("Simulador de Inferencia Bayesiana Secuencial en Diagnóstico Médico")
    st.caption(
        "Modelado matemático discreto (Ω = E ∪ Eᶜ), actualización iterativa del "
        "Teorema de Bayes y frecuencias naturales sobre base de datos epidemiológica "
        "oficial (CDC Perú, MINSA, OMS, OPS, Cochrane)."
    )

    st.warning(
        "**Alcance académico y pedagógico (MateDiscreta.pdf §1.4):** Este simulador es una "
        "herramienta de modelado matemático, análisis computacional y apoyo pedagógico. "
        "**No constituye un dispositivo médico ni software de soporte clínico**, no emite "
        "diagnósticos definitivos y no debe utilizarse para tomar decisiones médicas "
        "sobre pacientes reales."
    )

    with st.sidebar:
        st.header("Configuración del Escenario")

        preset_options = ["Configuración manual / personalizada"] + [p.title for p in presets]
        selected_preset_title = st.selectbox(
            "Casos demostrativos rápidos (Objetivo 2.2)",
            options=preset_options,
            index=0,
        )

        active_preset = next(
            (p for p in presets if p.title == selected_preset_title),
            None,
        )

        if active_preset is not None:
            last_applied = st.session_state.get("last_applied_preset")
            if last_applied != active_preset.key:
                st.session_state["last_applied_preset"] = active_preset.key
                st.session_state["active_pathology_slug"] = active_preset.pathology_slug
                st.session_state["selected_stratum_name"] = active_preset.stratum_name
                st.session_state["sequence_items"] = [
                    (order, obs.value) for order, obs in active_preset.step_orders_and_observations
                ]
            st.info(f"**Enfoque del caso:** {active_preset.pedagogical_focus}")
        else:
            st.session_state["last_applied_preset"] = None

        slug_list = [p.slug for p in pathologies]
        default_slug = st.session_state.get("active_pathology_slug", slug_list[0])
        if default_slug not in slug_list:
            default_slug = slug_list[0]

        selected_slug = st.selectbox(
            "1. Patología de estudio",
            options=slug_list,
            index=slug_list.index(default_slug),
            format_func=lambda s: pathologies_by_slug[s].name,
            disabled=(active_preset is not None),
        )
        if active_preset is not None:
            selected_slug = active_preset.pathology_slug
        pathology = pathologies_by_slug[selected_slug]
        _init_sequence_state(pathology)

        stratum_names = [s.stratum_name for s in pathology.scenarios]
        default_stratum = st.session_state.get("selected_stratum_name", stratum_names[0])
        if default_stratum not in stratum_names:
            default_stratum = stratum_names[0]

        selected_stratum_name = st.selectbox(
            "2. Estrato sintomático / epidemiológico (t = 0)",
            options=stratum_names,
            index=stratum_names.index(default_stratum),
            format_func=lambda name: next(
                f"{s.stratum_name} — P₀(E) = {format_percentage(s.prior)}"
                for s in pathology.scenarios
                if s.stratum_name == name
            ),
            disabled=(active_preset is not None),
        )
        if active_preset is not None:
            selected_stratum_name = active_preset.stratum_name
        st.session_state["selected_stratum_name"] = selected_stratum_name
        scenario = next(s for s in pathology.scenarios if s.stratum_name == selected_stratum_name)
        st.caption(f"**Criterio clínico (t = 0):** {scenario.clinical_description}")

        cohort_size = st.selectbox(
            "3. Tamaño de cohorte |Ω| (Frecuencias Naturales)",
            options=[1_000, 10_000, 100_000],
            index=1,
            format_func=lambda n: f"{format_integer(n)} personas",
        )

        st.divider()
        st.markdown(
            f"**Población objetivo:** {pathology.target_population}\n\n"
            f"**Fuentes oficiales:** "
            f"{'; '.join(c.short_label for c in pathology.citations)}"
        )

    st.subheader("Protocolo de Diagnóstico Secuencial (Modo Guiado + Flexible)")
    col_info, col_reset, col_add = st.columns([3, 1.4, 1.6])
    with col_info:
        st.markdown(
            "Inicia con el protocolo escalonado sugerido (`1 → 2 → 3`) de la **Tabla 1** "
            "o personaliza el orden, activa/desactiva etapas y alterna resultados "
            "`Positivo (+)` / `Negativo (-)`."
        )
    with col_reset:
        if st.button("Restaurar protocolo 1 → 2 → 3", width="stretch"):
            st.session_state["sequence_items"] = [
                (t.default_step_order, Observation.POSITIVE.value) for t in pathology.tests
            ]
            st.rerun()
    with col_add:
        if len(st.session_state["sequence_items"]) < len(pathology.tests):
            used_orders = {order for order, _ in st.session_state["sequence_items"]}
            next_test = next(
                (t for t in pathology.tests if t.default_step_order not in used_orders),
                pathology.tests[0],
            )
            if st.button(f"+ Añadir etapa ({next_test.name})", width="stretch"):
                st.session_state["sequence_items"].append(
                    (next_test.default_step_order, Observation.POSITIVE.value)
                )
                st.rerun()

    raw_items: list[tuple[int, str]] = list(st.session_state["sequence_items"])
    tests_by_order = {t.default_step_order: t for t in pathology.tests}
    test_orders = [t.default_step_order for t in pathology.tests]

    updated_items: list[tuple[int, str]] = []
    action_to_run: tuple[str, int] | None = None

    for idx, (step_order, obs_val) in enumerate(raw_items):
        c_stage, c_test, c_obs, c_params, c_up, c_down, c_del = st.columns(
            [0.9, 2.6, 1.7, 2.4, 0.5, 0.5, 0.5]
        )
        with c_stage:
            st.markdown(f"**Etapa t = {idx + 1}**")
        with c_test:
            chosen_order = st.selectbox(
                f"Prueba etapa {idx + 1} ({pathology.slug})",
                options=test_orders,
                index=test_orders.index(step_order) if step_order in test_orders else 0,
                format_func=lambda o: f"{o}. {tests_by_order[o].name}",
                label_visibility="collapsed",
            )
        with c_obs:
            chosen_obs = st.selectbox(
                f"Resultado etapa {idx + 1} ({pathology.slug})",
                options=[Observation.POSITIVE.value, Observation.NEGATIVE.value],
                index=0 if obs_val == Observation.POSITIVE.value else 1,
                format_func=lambda v: "Positivo (+)" if v == "+" else "Negativo (-)",
                label_visibility="collapsed",
            )
        chosen_spec = tests_by_order[chosen_order]
        with c_params:
            st.caption(
                f"Sens P(+|E): **{format_percentage(chosen_spec.sensitivity)}** · "
                f"Espec P(-|Eᶜ): **{format_percentage(chosen_spec.specificity)}** · "
                f"FP P(+|Eᶜ): **{format_percentage(chosen_spec.false_positive_rate)}**"
            )
        with c_up:
            if st.button(
                "↑",
                key=f"up_{idx}",
                disabled=(idx == 0),
                help="Subir etapa en la secuencia",
            ):
                action_to_run = ("up", idx)
        with c_down:
            if st.button(
                "↓",
                key=f"down_{idx}",
                disabled=(idx == len(raw_items) - 1),
                help="Bajar etapa en la secuencia",
            ):
                action_to_run = ("down", idx)
        with c_del:
            if st.button(
                "✕",
                key=f"del_{idx}",
                disabled=(len(raw_items) <= 1),
                help="Quitar esta etapa",
            ):
                action_to_run = ("del", idx)

        updated_items.append((chosen_order, chosen_obs))

    st.session_state["sequence_items"] = updated_items

    if action_to_run is not None:
        act, index = action_to_run
        apps_now = _resolve_applications(pathology, updated_items)
        if act == "up":
            moved = move_application(apps_now, index, -1)
            st.session_state["sequence_items"] = [
                (a.test.default_step_order, a.observation.value) for a in moved
            ]
        elif act == "down":
            moved = move_application(apps_now, index, 1)
            st.session_state["sequence_items"] = [
                (a.test.default_step_order, a.observation.value) for a in moved
            ]
        elif act == "del" and len(updated_items) > 1:
            updated_items.pop(index)
            st.session_state["sequence_items"] = updated_items
        st.rerun()

    applications = _resolve_applications(pathology, st.session_state["sequence_items"])
    trace = run_bayesian_sequence(
        pathology=pathology,
        scenario=scenario,
        applications=applications,
        cohort_size=cohort_size,
    )

    show_commutativity = False
    if len(applications) >= 2:
        show_commutativity = st.checkbox(
            "Superponer secuencia en orden invertido para verificar conmutatividad "
            "bayesiana bajo independencia condicional (P(E | O₁, …, Oₙ) es invariante al orden)",
            value=False,
        )

    permuted_trace = (
        run_bayesian_sequence(
            pathology=pathology,
            scenario=scenario,
            applications=reverse_sequence(applications),
            cohort_size=cohort_size,
        )
        if show_commutativity
        else None
    )

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        label=f"Prevalencia Inicial P₀(E) [{scenario.stratum_name}]",
        value=format_percentage(trace.initial_prior),
        help=f"Fracción exacta en t = 0: {format_fraction(trace.initial_prior)}",
    )
    n_steps = len(trace.steps)
    final_frac = format_fraction(trace.final_posterior)
    k2.metric(
        label=f"Probabilidad Posterior Final P_{n_steps}(E)",
        value=format_percentage(trace.final_posterior),
        delta=format_signed_percentage_delta(trace.total_belief_shift),
        help=f"Fracción exacta tras {n_steps} etapa(s): {final_frac}",
    )
    k3.metric(
        label=f"Complemento Sano P_{n_steps}(Eᶜ) = 1 − P(E)",
        value=format_percentage(trace.final_complement),
        help=f"Fracción exacta: {format_fraction(trace.final_complement)}",
    )
    k4.metric(
        label="Etapas Evaluadas en Secuencia (n)",
        value=f"{n_steps} prueba(s)",
        help=(
            "Cohorte evaluada bajo independencia condicional: "
            f"|Ω| = {format_integer(cohort_size)} personas."
        ),
    )

    if permuted_trace is not None:
        perm_frac = format_fraction(permuted_trace.final_posterior)
        perm_pct = format_percentage(permuted_trace.final_posterior)
        direct_pct = format_percentage(trace.final_posterior)
        st.success(
            f"**Verificación exacta de conmutatividad (aritmética racional `Fraction`):** "
            f"Secuencia directa P_{n_steps}(E) = `{final_frac}` ({direct_pct}) == "
            f"Secuencia invertida P_{n_steps}(E) = `{perm_frac}` ({perm_pct})."
        )

    tab_bayes, tab_strata, tab_freq, tab_db = st.tabs(
        [
            "1. Evolución y Demostración Paso a Paso (t = 0 … n)",
            "2. Comparativa entre Estratos Sintomáticos (t = 0)",
            "3. Frecuencias Naturales (Ω = E ∪ Eᶜ)",
            "4. Base de Datos Epidemiológica y Marco Formal",
        ]
    )

    with tab_bayes:
        fig_evol = build_bayesian_evolution_chart(trace, permuted_trace=permuted_trace)
        st.plotly_chart(fig_evol, width="stretch")

        st.markdown("### Demostración Algebraica Paso a Paso del Teorema de Bayes")
        st.caption(
            "Cada tarjeta desarrolla las ecuaciones exactas desde el estrato sintomático "
            "inicial (t = 0) hasta cada prueba diagnóstica encadenada (t ≥ 1)."
        )

        t0_walk = build_initial_stratum_walkthrough(trace)
        with st.expander(t0_walk.title, expanded=True):
            st.markdown(f"**Contexto clínico del estrato:** {t0_walk.clinical_description}")
            st.markdown(
                "**Partición booleana del espacio muestral finito ($\\Omega = E \\cup E^c$):**"
            )
            st.latex(t0_walk.partition_latex)
            st.info(t0_walk.natural_frequency_summary)

        step_walks = build_step_walkthroughs(trace)
        for sw in step_walks:
            with st.expander(sw.title, expanded=True):
                st.caption(f"**Mecanismo biológico (§1.3.4):** {sw.biological_mechanism}")
                st.markdown(
                    "**1. Probabilidad a priori heredada de la etapa previa ($P_{t-1}(E)$):**"
                )
                st.latex(sw.prior_inheritance_latex)
                st.markdown(
                    "**2. Verosimilitudes condicionales de la prueba "
                    "($P(O_t \\mid E)$ y $P(O_t \\mid E^c)$):**"
                )
                st.latex(sw.likelihood_latex)
                st.markdown("**3. Ley de la Probabilidad Total ($P(O_t)$):**")
                st.latex(sw.total_probability_latex)
                st.markdown(
                    "**4. Regla de recurrencia del Teorema de Bayes ($P_t(E) = P(E \\mid O_t)$):**"
                )
                st.latex(sw.bayes_substitution_latex)
                st.caption(sw.natural_frequency_check)

    with tab_strata:
        st.markdown(
            "### Impacto del Estrato Sintomático/Epidemiológico Inicial ($t = 0$) "
            "— Sesgo de Tasa Base"
        )
        st.markdown(
            "Esta vista ejecuta **exactamente la misma secuencia de pruebas diagnósticas** "
            "sobre todos los estratos clínicos de la patología registrada en la Tabla 1, "
            "mostrando cómo la prevalencia previa $P_0(E)$ condiciona toda la trayectoria."
        )
        multi_traces = compare_across_scenarios(
            pathology=pathology,
            applications=applications,
            cohort_size=cohort_size,
        )
        fig_multi = build_multi_stratum_comparison_chart(multi_traces)
        st.plotly_chart(fig_multi, width="stretch")

        comparison_rows: list[dict[str, str]] = []
        for mt in multi_traces:
            p0_pct = format_percentage(mt.initial_prior)
            p0_frc = format_fraction(mt.initial_prior)
            row: dict[str, str] = {
                "Estrato Sintomático (t = 0)": mt.scenario.stratum_name,
                "P₀(E) Inicial": f"{p0_pct} ({p0_frc})",
            }
            for step in mt.steps:
                col_key = (
                    f"t={step.step_index}: {step.application.test.name} "
                    f"({step.application.observation.value})"
                )
                row[col_key] = (
                    f"{format_percentage(step.posterior)} ({format_fraction(step.posterior)})"
                )
            row["Cambio Neto (ΔP)"] = format_signed_percentage_delta(mt.total_belief_shift)
            comparison_rows.append(row)
        st.table(comparison_rows)

    with tab_freq:
        st.markdown("### Visualización de Frecuencias Naturales sobre $|\\Omega| = N$")
        st.markdown(
            "Las frecuencias naturales descomponen la cohorte finita en conteos concretos "
            "de personas enfermas ($E$) y sanas ($E^c$), haciendo visible por qué una tasa "
            "pequeña de falsos positivos aplicada sobre una población sana masiva supera a "
            "los verdaderos positivos en tamizajes de baja prevalencia."
        )

        stage_options = list(range(len(trace.frequencies)))
        selected_stage_idx = st.selectbox(
            "Seleccionar etapa para inspeccionar el desglose poblacional",
            options=stage_options,
            index=len(stage_options) - 1,
            format_func=lambda idx: trace.frequencies[idx].stage_label,
        )
        active_freq = trace.frequencies[selected_stage_idx]
        fig_freq = build_natural_frequencies_chart(active_freq)
        st.plotly_chart(fig_freq, width="stretch")

        freq_rows: list[dict[str, str]] = []
        for fb in trace.frequencies:
            freq_rows.append(
                {
                    "Etapa": fb.stage_label,
                    "Cohorte |Ω|": format_integer(fb.cohort_size),
                    "Enfermos (E)": format_integer(fb.rounded_diseased),
                    "Sanos (Eᶜ)": format_integer(fb.rounded_healthy),
                    "Verdaderos Pos. (VP)": (
                        "—"
                        if fb.rounded_true_positives is None
                        else format_integer(fb.rounded_true_positives)
                    ),
                    "Falsos Pos. (FP)": (
                        "—"
                        if fb.rounded_false_positives is None
                        else format_integer(fb.rounded_false_positives)
                    ),
                    "Falsos Neg. (FN)": (
                        "—"
                        if fb.rounded_false_negatives is None
                        else format_integer(fb.rounded_false_negatives)
                    ),
                    "Verdaderos Neg. (VN)": (
                        "—"
                        if fb.rounded_true_negatives is None
                        else format_integer(fb.rounded_true_negatives)
                    ),
                    "Pₜ(E) por cociente exacto": (
                        f"{format_percentage(fb.exact_posterior_from_counts)} "
                        f"({format_fraction(fb.exact_posterior_from_counts)})"
                    ),
                }
            )
        st.table(freq_rows)

    with tab_db:
        st.markdown("### Matriz Epidemiológica Oficial (Base de Datos SQLite — Tabla 1)")
        matrix = repo.list_parameter_matrix()
        matrix_table = [
            {
                "Patología": r.pathology_name,
                "Población Objetivo": r.target_population,
                "Prevalencias Iniciales P₀(E)": r.priors_summary,
                "Orden": str(r.default_step_order),
                "Prueba Diagnóstica": r.test_name,
                "Sensibilidad P(+|E)": format_percentage(r.sensitivity, decimals=1),
                "Especificidad P(-|Eᶜ)": format_percentage(r.specificity, decimals=1),
                "Falsos Pos. P(+|Eᶜ)": format_percentage(r.false_positive_rate, decimals=1),
                "Fuentes": r.citations_summary,
            }
            for r in matrix
        ]
        st.table(matrix_table)

        col_model, col_limits = st.columns(2)
        with col_model:
            st.markdown("#### Formulación como Modelo Discreto (§1.3)")
            st.markdown(
                "1. **Espacio muestral finito ($\\Omega$) y partición disjunta:** "
                "$E \\cup E^c = \\Omega$ y $E \\cap E^c = \\emptyset$, con "
                "$P_0(E) + P_0(E^c) = 1$.\n"
                "2. **Sucesión discreta de etapas y observaciones:** pasos "
                "$t \\in \\{0, 1, \\dots, n\\}$ con resultados binarios "
                "$O_t \\in \\{+_t, -_t\\}$.\n"
                "3. **Regla de recurrencia bayesiana:**"
            )
            st.latex(
                r"P_t(E) = P(E \mid O_t) = \frac{P(O_t \mid E)\cdot P_{t-1}(E)}"
                r"{P(O_t \mid E)\cdot P_{t-1}(E) + P(O_t \mid E^c)\cdot (1 - P_{t-1}(E))}"
            )
            st.markdown(
                "4. **Supuesto de independencia condicional:** "
                "$P(O_1, \\dots, O_n \\mid E) = \\prod_{t=1}^n P(O_t \\mid E)$ y "
                "$P(O_1, \\dots, O_n \\mid E^c) = \\prod_{t=1}^n P(O_t \\mid E^c)$, "
                "justificado al encadenar pruebas con mecanismos biológicos diferenciados."
            )

        with col_limits:
            st.markdown("#### Alcance y Limitaciones (§1.4)")
            st.markdown(
                "- **No constituye dispositivo médico ni software de soporte clínico:** "
                "no reemplaza el criterio médico profesional ni debe usarse sobre "
                "pacientes reales.\n"
                "- **Idealización de independencia condicional:** en escenarios clínicos "
                "reales puede existir dependencia residual entre pruebas que el modelo "
                "base no parametriza con matrices de covarianza.\n"
                "- **Parámetros poblacionales estáticos por escenario:** los valores "
                "precargados representan estimaciones puntuales reportadas en guías "
                "oficiales y revisiones sistemáticas."
            )
            st.markdown("#### Referencias Bibliográficas Oficiales")
            for cite in repo.list_all_citations():
                ref_line = f"- **{cite.short_label}:** {cite.full_reference}"
                if cite.doi_or_url:
                    ref_line += f" [{cite.doi_or_url}]({cite.doi_or_url})"
                st.markdown(ref_line)


if __name__ == "__main__":
    main()

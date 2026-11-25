# Simulador de Inferencia Bayesiana Secuencial (`med-diagnosis`)

Aplicación web interactiva desarrollada en **Python 3.11+** y **Streamlit** que modela, demuestra algebraicamente y visualiza la actualización iterativa del **Teorema de Bayes** y las **frecuencias naturales** en protocolos escalonados de diagnóstico médico, respaldada por una base de datos epidemiológica relacional (**SQLite**) extraída de fuentes oficiales para **COVID-19**, **Tuberculosis Pulmonar (TB)** y **Dengue (DENV 1–4)**.

> [!WARNING]
> **Alcance académico y pedagógico:** Este sistema está concebido exclusivamente como una herramienta de modelado matemático discreto, análisis computacional y apoyo didáctico. **No constituye un dispositivo médico ni software de soporte clínico**, no emite diagnósticos definitivos y no debe utilizarse para tomar decisiones médicas sobre pacientes reales.

---

## Inicio Rápido

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar el paquete con dependencias de aplicación y desarrollo
pip install -e ".[app,dev]"

# 3. Ejecutar la aplicación interactiva en Streamlit
streamlit run app.py
```

Para ejecutar la suite completa de pruebas automatizadas y verificaciones de calidad:

```bash
pytest --cov=med_diagnosis --cov-report=term-missing
ruff check src tests app.py
ruff format --check src tests app.py
```

---

## Marco Matemático y Lógico-Computacional

Sobre un conjunto universo finito de individuos evaluados $\Omega$, se define la partición booleana exacta $\Omega = E \cup E^c$ con $E \cap E^c = \emptyset$, donde $E$ representa la presencia de la patología y $E^c$ su ausencia ($P_0(E) + P_0(E^c) = 1$).

1. **Etapa base ($t = 0$ — Estrato sintomático/epidemiológico):** El perfil clínico inicial seleccionado de la Tabla 1 fija la probabilidad a priori $P_0(E)$ y su complemento sano $P_0(E^c) = 1 - P_0(E)$.
2. **Sucesión discreta de pruebas ($t \in \{1, 2, \dots, n\}$):** En cada paso $t$, la aplicación de una prueba diagnóstica $T_t$ produce una observación binaria $O_t \in \{+_t, -_t\}$.
3. **Regla de recurrencia bayesiana:** La probabilidad posterior $P_{t-1}(E)$ se hereda como nueva probabilidad a priori y se actualiza mediante el Teorema de Bayes y la Ley de la Probabilidad Total:

$$P_t(E) = P(E \mid O_t) = \frac{P(O_t \mid E)\cdot P_{t-1}(E)}{P(O_t \mid E)\cdot P_{t-1}(E) + P(O_t \mid E^c)\cdot (1 - P_{t-1}(E))}$$

4. **Independencia condicional y conmutatividad:** Al encadenar pruebas que actúan sobre mecanismos biológicos diferenciados (antígenos, anticuerpos, radiografía, amplificación molecular), se asume independencia condicional dado el estado real del paciente. Toda la capa de dominio opera con **aritmética racional exacta (`fractions.Fraction`)**, garantizando sin error de punto flotante que $P_t(E) + P_t(E^c) = 1$ y que el orden de las pruebas sea conmutativo.

---

## Matriz Epidemiológica Oficial (Tabla 1)

| Patología | Estratos Sintomáticos / Epidemiológicos ($t = 0$) | Secuencia Protocolar Base ($T_1 \to T_2 \to T_3$) | Sensibilidad $P(+\mid E)$ | Especificidad $P(-\mid E^c)$ | Falsos Pos. $P(+\mid E^c)$ | Fuentes Oficiales |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **COVID-19 (SARS-CoV-2)** | Asintomático: `1.5%`<br>Sintomático IRA: `12.0%`<br>Contacto estrecho: `35.0%` | 1. Prueba Rápida de Antígenos<br>2. Prueba Serológica (IgM/IgG)<br>3. RT-PCR en Tiempo Real | `73.0%`<br>`68.0%`<br>`95.0%` | `99.1%`<br>`96.5%`<br>`99.5%` | `0.9%`<br>`3.5%`<br>`0.5%` | CDC Perú (2024a); Dinnes et al. (2022); OMS (2021) |
| **Tuberculosis Pulmonar (TB)** | Población general: `0.15%`<br>Sintomático Resp.: `4.5%`<br>Contacto TB activo: `15.0%` | 1. Radiografía de Tórax<br>2. Baciloscopía Esputo (AFB)<br>3. GeneXpert MTB/RIF | `87.0%`<br>`65.0%`<br>`89.0%` | `89.0%`<br>`98.0%`<br>`99.0%` | `11.0%`<br>`2.0%`<br>`1.0%` | MINSA (2023); OMS (2024) |
| **Dengue (DENV 1–4)** | Fuera de brote: `3.0%`<br>Brote estacional: `22.0%` | 1. Prueba Rápida Antígeno NS1<br>2. ELISA IgM Dengue<br>3. RT-PCR Multiplex DENV | `76.0%`<br>`85.0%`<br>`94.0%` | `98.0%`<br>`92.0%`<br>`99.5%` | `2.0%`<br>`8.0%`<br>`0.5%` | CDC Perú (2024b); OPS (2022) |

---

## Arquitectura del Proyecto

```text
.
├── app.py                                   # Punto de entrada principal de Streamlit
├── pyproject.toml                           # Configuración PEP 621, pytest y ruff
├── requirements.txt                         # Dependencias para Streamlit Community Cloud (-e .)
├── src/med_diagnosis/
│   ├── domain/                              # Núcleo matemático puro (100 % stdlib + Fraction)
│   │   ├── models.py                        # Entidades inmutables y partición booleana
│   │   ├── bayes.py                         # Recurrencia bayesiana, modo guiado+flexible y comparador
│   │   └── frequencies.py                   # Desglose de frecuencias naturales sobre |Ω| = N
│   ├── data/                                # Persistencia epidemiológica SQLite (3FN)
│   │   ├── schema.sql                       # Esquema DDL con claves foráneas y restricciones CHECK
│   │   ├── epidemiology_seed.json           # Semilla declarativa de patologías, estratos, pruebas y citas
│   │   ├── epidemiology.db                  # Base SQLite generada determinísticamente
│   │   └── database.py                      # Constructor y repositorio SqliteEpidemiologyRepository
│   └── ui/                                  # Presentación, demostrador LaTeX y gráficos Plotly
│       ├── formatting.py                    # Formateadores racionales, porcentuales y de conteos
│       ├── math_walkthrough.py              # Demostración algebraica paso a paso desde t = 0 hasta t = n
│       ├── presets.py                       # Casos demostrativos precargados
│       └── charts.py                        # Curvas de evolución, comparativa multi-estrato y frecuencias
└── tests/                                   # Suite de pruebas unitarias, AST, SQLite y AppTest
```

---

## Despliegue en Streamlit Community Cloud

> [!TIP]
> El archivo `requirements.txt` incluye `-e .` para instalar el paquete `med_diagnosis` desde `src/` automáticamente en el contenedor de Streamlit Community Cloud, y la base de datos SQLite funciona sin requerir servidores externos ni variables de entorno.

1. Subir la rama al repositorio público de GitHub.
2. Ingresar a [share.streamlit.io](https://share.streamlit.io/) y seleccionar **Create app**.
3. Elegir el repositorio, la rama correspondiente y especificar `app.py` como **Main file path**.
4. Pulsar **Deploy**.

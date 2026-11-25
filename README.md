# Simulador de Inferencia Bayesiana Secuencial (`med-diagnosis`)

Aplicación web interactiva en **Python 3.11+** y **Streamlit** para modelar, demostrar algebraicamente y visualizar la actualización iterativa del **Teorema de Bayes** y las **frecuencias naturales** en protocolos escalonados de diagnóstico médico, respaldada por una base de datos epidemiológica relacional (**SQLite**) extraída de fuentes oficiales para **COVID-19**, **Tuberculosis Pulmonar (TB)** y **Dengue (DENV 1–4)**.

> [!WARNING]
> **Alcance académico y pedagógico:** Esta herramienta está concebida exclusivamente para el modelado matemático discreto, el análisis computacional y la enseñanza del sesgo de la tasa base (*base rate fallacy*). **No constituye un dispositivo médico ni software de soporte clínico**, no emite diagnósticos definitivos y no debe utilizarse para tomar decisiones sobre pacientes reales.

---

## Capacidades principales

| Capacidad | Descripción |
| :--- | :--- |
| **Motor bayesiano racional exacto** | Opera íntegramente con `fractions.Fraction` sin errores de punto flotante, garantizando $P_t(E) + P_t(E^c) = 1$ e invarianza exacta al orden bajo independencia condicional. |
| **Modo guiado + flexible** | Inicia con la secuencia clínica sugerida (`1 → 2 → 3`) de cada patología y permite alternar resultados (`+` / `-`), añadir, quitar o reordenar etapas y superponer la secuencia invertida para comprobar la conmutatividad. |
| **Demostración algebraica paso a paso** | Despliega en LaTeX la partición booleana inicial desde el estrato sintomático ($t = 0$) y los cuatro pasos puros de Bayes en cada prueba encadenada ($t \ge 1$). |
| **Comparativa multi-estrato y frecuencias naturales** | Contrasta simultáneamente todos los estratos sintomáticos de la patología activa y traduce las probabilidades a conteos enteros sobre cohortes de `1 000`, `10 000` o `100 000` personas ($VP, FP, FN, VN$). |
| **Base de datos SQLite 3FN auditada** | Almacena de forma offline y tolerante a solo lectura las 3 patologías, 8 estratos, 9 pruebas diagnósticas y 7 referencias oficiales. |

---

## Inicio rápido

### Requisitos previos

- **Python 3.11** o superior (`>= 3.11`)
- `pip` y `venv`

### Instalación y ejecución local

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar el paquete con dependencias de aplicación y desarrollo
pip install -e ".[app,dev]"

# 3. Iniciar la aplicación interactiva en Streamlit
streamlit run app.py
```

La interfaz quedará disponible en `http://localhost:8501`.

### Pruebas y calidad de código

```bash
# Suite completa (dominio, SQLite, fronteras AST, LaTeX y AppTest) con cobertura
pytest --cov=med_diagnosis --cov-report=term-missing

# Linter y verificación de formato
ruff check src tests app.py
ruff format --check src tests app.py
```

---

## Marco matemático y modelo discreto

Sobre un conjunto universo finito de individuos evaluados $\Omega$, se define la partición booleana exacta $\Omega = E \cup E^c$ con $E \cap E^c = \emptyset$, donde $E$ representa la presencia de la patología y $E^c$ su ausencia ($P_0(E) + P_0(E^c) = 1$).

1. **Etapa base ($t = 0$ — Estrato sintomático/epidemiológico):** El perfil clínico inicial seleccionado de la Tabla 1 fija la probabilidad a priori $P_0(E)$ y su complemento sano $P_0(E^c) = 1 - P_0(E)$.
2. **Sucesión discreta de pruebas ($t \in \{1, 2, \dots, n\}$):** Cada prueba diagnóstica $T_t$ aplicada produce una observación binaria $O_t \in \{+_t, -_t\}$.
3. **Regla de recurrencia bayesiana:** La probabilidad posterior $P_{t-1}(E)$ se hereda como probabilidad a priori de la etapa siguiente y se actualiza mediante la Ley de la Probabilidad Total y el Teorema de Bayes:

$$P_t(E) = P(E \mid O_t) = \frac{P(O_t \mid E)\cdot P_{t-1}(E)}{P(O_t \mid E)\cdot P_{t-1}(E) + P(O_t \mid E^c)\cdot (1 - P_{t-1}(E))}$$

4. **Independencia condicional:** Al encadenar pruebas que actúan sobre mecanismos biológicos diferenciados (antígenos, anticuerpos, radiografía y amplificación molecular), se asume independencia condicional dado el estado real del paciente, por lo que $P(O_1, \dots, O_n \mid E) = \prod_{t=1}^n P(O_t \mid E)$.

---

## Matriz epidemiológica oficial

Los parámetros precargados en SQLite provienen de la Tabla 1 de [MateDiscreta.pdf](MateDiscreta.pdf):

| Patología | Estratos Sintomáticos / Epidemiológicos ($t = 0$) | Secuencia Protocolar Base ($T_1 \to T_2 \to T_3$) | Sensibilidad $P(+\mid E)$ | Especificidad $P(-\mid E^c)$ | Falsos Pos. $P(+\mid E^c)$ | Fuentes Oficiales |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **COVID-19 (SARS-CoV-2)** | Asintomático: `1.5%`<br>Sintomático IRA: `12.0%`<br>Contacto estrecho: `35.0%` | 1. Prueba Rápida de Antígenos<br>2. Prueba Serológica (IgM/IgG)<br>3. RT-PCR en Tiempo Real | `73.0%`<br>`68.0%`<br>`95.0%` | `99.1%`<br>`96.5%`<br>`99.5%` | `0.9%`<br>`3.5%`<br>`0.5%` | CDC Perú (2024a); Dinnes et al. (2022); OMS (2021) |
| **Tuberculosis Pulmonar (TB)** | Población general: `0.15%`<br>Sintomático Resp.: `4.5%`<br>Contacto TB activo: `15.0%` | 1. Radiografía de Tórax<br>2. Baciloscopía Esputo (AFB)<br>3. GeneXpert MTB/RIF | `87.0%`<br>`65.0%`<br>`89.0%` | `89.0%`<br>`98.0%`<br>`99.0%` | `11.0%`<br>`2.0%`<br>`1.0%` | MINSA (2023); OMS (2024) |
| **Dengue (DENV 1–4)** | Fuera de brote: `3.0%`<br>Brote estacional: `22.0%` | 1. Prueba Rápida Antígeno NS1<br>2. ELISA IgM Dengue<br>3. RT-PCR Multiplex DENV | `76.0%`<br>`85.0%`<br>`94.0%` | `98.0%`<br>`92.0%`<br>`99.5%` | `2.0%`<br>`8.0%`<br>`0.5%` | CDC Perú (2024b); OPS (2022) |

---

## Arquitectura del proyecto

```text
.
├── app.py                                   # Punto de entrada principal de Streamlit
├── pyproject.toml                           # Configuración PEP 621, pytest y ruff
├── requirements.txt                         # Dependencias para Streamlit Community Cloud (-e .)
├── INFORME.md                               # Informe técnico completo (análisis, diagramas, código y pruebas)
├── docs/
│   ├── GUIA_DE_USO_Y_EXPLICACION.md         # Guía paso a paso orientada a exposiciones sin base previa en Bayes
│   └── diagrams/                            # Diagramas de flujo y modelo UML en HTML y SVG autónomo
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

> [!NOTE]
> **Documentación complementaria:**
> - [docs/GUIA_DE_USO_Y_EXPLICACION.md](docs/GUIA_DE_USO_Y_EXPLICACION.md) explica el funcionamiento del simulador paso a paso desde cero para público no técnico.
> - [INFORME.md](INFORME.md) reúne la especificación, el análisis E-P-S, los diagramas vectoriales (`docs/diagrams/*.svg`) y la matriz de verificación analítica.

---

## Despliegue en Streamlit Community Cloud

> [!TIP]
> El archivo `requirements.txt` incluye `-e .` para instalar automáticamente el paquete `med_diagnosis` desde `src/`, y el repositorio SQLite cuenta con respaldo determinista en memoria si el entorno en la nube opera en modo de solo lectura.

1. Subir la rama al repositorio público en GitHub.
2. Ingresar a [share.streamlit.io](https://share.streamlit.io/) y seleccionar **Create app**.
3. Seleccionar el repositorio, la rama correspondiente y establecer `app.py` como **Main file path**.
4. Pulsar **Deploy**.

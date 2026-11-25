# Guía de uso y explicación del simulador (desde cero y sin saber Bayes)

Este documento explica qué problema resuelve el proyecto, de dónde sale cada dato clínico y cómo usar la aplicación paso a paso para explicársela a cualquier persona, incluso si nunca estudió probabilidad ni escuchó hablar del Teorema de Bayes.

---

## 1. La idea central en un minuto (sin fórmulas)

Imagina que una prueba médica tiene una precisión del **99 %** en personas sanas (solo se equivoca en el 1 % de los sanos). Si te haces esa prueba y sale **positiva**, la intuición dice: *"Tengo un 99 % de probabilidad de estar enfermo"*.

Esa intuición suele ser falsa, y en medicina ese error se llama **sesgo de la tasa base**.

Piénsalo con personas reales en lugar de porcentajes. Supongamos que evaluamos a **10 000 personas** de la población general para detectar **Tuberculosis Pulmonar** en Lima, donde la prevalencia en personas sin síntomas es del **0.15 %**:

1. De esas 10 000 personas, solo **15 están realmente enfermas** y **9 985 están sanas**.
2. Les hacemos una **Radiografía de Tórax**, que detecta al 87 % de los enfermos pero da un 11 % de falsas alarmas en personas sanas.
3. Entre los 15 enfermos, la radiografía detecta a **13 personas** (verdaderos positivos).
4. Entre los 9 985 sanos, ese 11 % de margen de error genera **1 098 falsas alarmas** (falsos positivos).

Si sumas a todos los que recibieron un resultado positivo en la radiografía, tienes `13 + 1 098 = 1 111` personas con resultado positivo, pero solo **13** tienen tuberculosis. La probabilidad real de estar enfermo tras esa primera prueba positiva es apenas `13 / 1 111 = 1.17 %`.

**¿Cómo lo resuelven los hospitales?**
No diagnostican con una sola prueba. Aplican un **protocolo secuencial**: a quienes dieron positivo en la primera prueba les hacen una segunda prueba distinta (Baciloscopía de esputo, que sube la probabilidad al **27.86 %**) y luego una tercera prueba molecular (GeneXpert, que filtra casi todas las falsas alarmas restantes y lleva la certeza al **97.17 %**).

**Qué hace este simulador:**
Toma exactamente ese proceso médico escalonado y calcula cómo cambia la probabilidad de estar enfermo después de cada prueba, mostrando tanto las ecuaciones exactas como el conteo de personas reales.

---

## 2. De dónde se obtienen los datos epidemiológicos y clínicos

Ningún número del simulador es inventado. Todos los parámetros provienen de la **Tabla 1** del estudio del proyecto ([MateDiscreta.pdf](file:///Users/hd/Projects/github.com/Aireck2/med-diagnosis/MateDiscreta.pdf)), que consolida reportes oficiales de vigilancia epidemiológica en el Perú y guías clínicas internacionales.

### 2.1 Qué mide cada parámetro

En cada enfermedad se usan tres datos básicos:

* **Prevalencia inicial $P_0(E)$ (etapa $t = 0$):** qué porcentaje de personas tiene la enfermedad *antes* de hacer la primera prueba, según su cuadro de síntomas o riesgo epidemiológico.
* **Sensibilidad $P(+\mid E)$:** qué porcentaje de las personas **enfermas** da positivo en la prueba (capacidad de detectar a los enfermos).
* **Especificidad $P(-\mid E^c)$:** qué porcentaje de las personas **sanas** da negativo en la prueba (capacidad de descartar a los sanos). El complemento (`100 % - Especificidad`) es la tasa de **falsos positivos**.

### 2.2 Fuentes oficiales por patología

| Patología | De dónde sale la prevalencia inicial ($t = 0$) | De dónde salen la sensibilidad y especificidad de las pruebas |
| :--- | :--- | :--- |
| **COVID-19 (SARS-CoV-2)** | **CDC Perú (2024a):** Sala situacional COVID-19 y boletín epidemiológico nacional (tasas en tamizaje asintomático `1.5 %`, pacientes con infección respiratoria aguda `12.0 %` y contactos estrechos `35.0 %`). | **OMS (2021)** y revisión sistemática de la **Colaboración Cochrane (Dinnes et al., 2022)** para prueba rápida de antígenos, serología IgM/IgG y RT-PCR en tiempo real. |
| **Tuberculosis Pulmonar (TB)** | **MINSA (2023):** Norma Técnica de Salud NTS N.° 200-MINSA/DGIESP-2023 para Lima Metropolitana (población general `0.15 %`, sintomáticos respiratorios con tos > 15 días `4.5 %` y contactos domiciliarios `15.0 %`). | **MINSA (2023)** y **OMS (2024)** (*WHO consolidated guidelines on tuberculosis, Module 3*) para radiografía de tórax, baciloscopía de esputo (AFB) y prueba molecular GeneXpert MTB/RIF. |
| **Dengue (DENV 1-4)** | **CDC Perú (2024b):** Sala situacional de dengue en el Perú para pacientes febriles en zonas con el vector *Aedes aegypti* (fuera de temporada de brote `3.0 %` y durante brote estacional activo `22.0 %`). | **OPS (2022):** Directrices para el diagnóstico clínico y tratamiento del dengue, chikunguña y zika para antígeno NS1, ELISA IgM y RT-PCR multiplex. |

### 2.3 Cómo están guardados los datos dentro del proyecto

Para que cualquier persona pueda auditar los datos y la aplicación funcione sin conexión a internet ni servidores externos, la capa de datos (`src/med_diagnosis/data/`) se organiza en tres archivos:

1. [epidemiology_seed.json](file:///Users/hd/Projects/github.com/Aireck2/med-diagnosis/src/med_diagnosis/data/epidemiology_seed.json): archivo JSON legible donde están declaradas las 3 patologías, los 8 estratos sintomáticos iniciales, las 9 pruebas diagnósticas (como fracciones exactas, por ejemplo `730 / 1000` para `73.0 %`) y las 7 referencias bibliográficas completas.
2. [schema.sql](file:///Users/hd/Projects/github.com/Aireck2/med-diagnosis/src/med_diagnosis/data/schema.sql): estructura relacional SQLite en Tercera Forma Normal (tablas `pathologies`, `prevalence_scenarios`, `diagnostic_tests`, `citations` y `pathology_citations`) con reglas de integridad que impiden cargar probabilidades fuera del rango válido.
3. [epidemiology.db](file:///Users/hd/Projects/github.com/Aireck2/med-diagnosis/src/med_diagnosis/data/epidemiology.db) y [database.py](file:///Users/hd/Projects/github.com/Aireck2/med-diagnosis/src/med_diagnosis/data/database.py): base de datos SQLite que la aplicación consulta en modo de solo lectura al iniciar.

---

## 3. Cómo explicar el Teorema de Bayes usando el simulador

Cuando alguien pregunta *"¿Pero qué cuenta hace exactamente el Teorema de Bayes?"*, puedes explicarlo en tres pasos muy simples que coinciden con lo que muestra la pantalla:

1. **Paso base ($t = 0$, antes de las pruebas):**
   Elegimos el perfil del paciente (por ejemplo, si tiene síntomas o no). Ese perfil nos da el porcentaje inicial de enfermos $P_0(E)$ y el porcentaje de sanos $P_0(E^c) = 1 - P_0(E)$.
2. **Cada vez que aplicamos una prueba ($t = 1, 2, 3$):**
   * Si la prueba sale **Positiva (`+`)**, el simulador calcula qué proporción de todos los resultados positivos corresponde a personas realmente enfermas:
     $$\text{Probabilidad actualizada} = \frac{\text{Enfermos que dan positivo}}{\text{Enfermos que dan positivo} + \text{Sanos que dan falso positivo}}$$
   * Si la prueba sale **Negativa (`-`)**, calcula qué proporción de todos los resultados negativos corresponde a enfermos que la prueba no detectó (falsos negativos):
     $$\text{Probabilidad actualizada} = \frac{\text{Enfermos que dan falso negativo}}{\text{Enfermos que dan falso negativo} + \text{Sanos que dan negativo correcto}}$$
3. **Encadenamiento secuencial:**
   El resultado final de la Prueba 1 se convierte automáticamente en el punto de partida (la nueva probabilidad previa) de la Prueba 2, y el resultado de la Prueba 2 pasa a ser el punto de partida de la Prueba 3.

> [!TIP]
> **Por qué el simulador usa fracciones exactas (`Fraction`):** En [bayes.py](file:///Users/hd/Projects/github.com/Aireck2/med-diagnosis/src/med_diagnosis/domain/bayes.py), Python no redondea decimales durante los cálculos intermedios. Opera con fracciones racionales exactas (como `730 / 1321`). Gracias a eso, si cambias el orden de las pruebas, el resultado final coincide de manera exacta sin errores de redondeo.

---

## 4. Paso a paso para usar y demostrar la aplicación

### Paso 1: Iniciar el simulador

Desde la terminal, en la carpeta raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[app,dev]"
streamlit run app.py
```

El navegador abrirá automáticamente la interfaz en `http://localhost:8501`.

### Paso 2: Configurar el escenario en la barra lateral izquierda

En la columna izquierda tienes dos formas de empezar:

* **Opción A (ideal para exposiciones): "Casos demostrativos rápidos"**
  Despliega el primer menú y elige uno de los 4 escenarios pedagógicos preconfigurados:
  1. *Sesgo de Tasa Base en Tuberculosis (0.15 %)*: muestra cómo una radiografía positiva sola da `1.17 %` y cómo las 3 pruebas positivas llevan al `97.17 %`.
  2. *Tamizaje Asintomático en COVID-19 (1.5 %)*: muestra por qué un antígeno positivo en una persona sin síntomas apenas da `55.26 %` (casi como lanzar una moneda) hasta que se confirma con RT-PCR (`99.57 %`).
  3. *Secuencia con Resultados Discordantes en COVID-19 (12.0 %)*: muestra qué pasa cuando el antígeno da `+`, la prueba serológica contradice con `-` y la RT-PCR desempata con `+`.
  4. *Alta Prevalencia en Brote de Dengue (22.0 %)*: muestra cómo, en pleno brote epidémico, la primera prueba rápida NS1 positiva ya salta al `91.48 %`.

* **Opción B: "Configuración manual / personalizada"**
  1. Elige la **Patología de estudio** (COVID-19, Tuberculosis o Dengue).
  2. Elige el **Estrato sintomático / epidemiológico ($t = 0$)**. Debajo verás el criterio clínico oficial que define a ese grupo.
  3. Elige el **Tamaño de cohorte $|\Omega|$** (`1 000`, `10 000` o `100 000` personas) para los gráficos de conteo de personas.

### Paso 3: Manejar el protocolo de pruebas (Modo Guiado + Flexible)

En la sección central **"Protocolo de Diagnóstico Secuencial"**:

* Por defecto aparecen cargadas las 3 pruebas en el orden clínico recomendado (`1 → 2 → 3`).
* En cada fila puedes cambiar qué prueba se aplica, cambiar su resultado entre `Positivo (+)` y `Negativo (-)`, mover una prueba hacia arriba (`↑`) o hacia abajo (`↓`), o quitarla (`✕`) para ver qué pasaría si solo se hicieran una o dos pruebas.
* El botón **"Restaurar protocolo 1 → 2 → 3"** devuelve la secuencia original en un clic.
* Si marcas la casilla **"Superponer secuencia en orden invertido"**, el gráfico dibuja una segunda línea punteada aplicando las mismas pruebas al revés. Verás que el camino intermedio cambia, pero ambas curvas terminan exactamente en el mismo porcentaje final.

### Paso 4: Qué mostrar en cada una de las 4 pestañas

Debajo de las 4 tarjetas de resumen (Prevalencia Inicial, Probabilidad Posterior Final, Complemento Sano y Etapas Evaluadas en Secuencia), tienes cuatro pestañas:

#### Pestaña 1: Evolución y Demostración Paso a Paso ($t = 0 \dots n$)
* **Arriba:** el gráfico interactivo de líneas muestra cómo sube la probabilidad de enfermedad $P_t(E)$ (línea verde) y cómo baja la probabilidad de estar sano $P_t(E^c)$ (línea gris punteada) en cada etapa.
* **Abajo (Demostración Algebraica):** tienes tarjetas desplegables para **$t = 0$ (el estrato sintomático elegido)** y para **cada prueba $t = 1, 2, 3$**. Cada tarjeta muestra la fórmula matemática y reemplaza cada letra por la fracción exacta y el porcentaje de esa prueba.

#### Pestaña 2: Comparativa entre Estratos Sintomáticos ($t = 0$)
* Ejecuta la secuencia de pruebas que tengas armada sobre **todos los perfiles sintomáticos de esa enfermedad al mismo tiempo**.
* Es la forma más rápida de explicar por qué los síntomas importan: en un solo gráfico se ve cómo la misma prueba de COVID-19 da resultados muy distintos si la persona es *Asintomática (1.5 %)*, *Sintomática IRA (12.0 %)* o *Contacto estrecho (35.0 %)*.

#### Pestaña 3: Frecuencias Naturales ($\Omega = E \cup E^c$)
* **Es la pestaña clave para explicar el proyecto a alguien que no sabe matemática.**
* En lugar de hablar de probabilidades abstractas, muestra barras con cantidad de personas reales (por ejemplo, sobre `10 000` personas): cuántos enfermos reales dieron positivo (*Verdaderos Positivos*) frente a cuántos sanos dieron positivo por error (*Falsos Positivos*).

#### Pestaña 4: Base de Datos Epidemiológica y Marco Formal
* Muestra la tabla completa leída directamente desde SQLite (**Tabla 1**), las ecuaciones del modelo discreto, las tres limitaciones metodológicas del estudio y la lista completa de referencias bibliográficas con sus enlaces oficiales.

---

## 5. Preguntas típicas en una presentación y cómo responderlas

**1. "¿Por qué los síntomas se eligen al principio ($t = 0$) y no como una prueba más dentro de la cadena?"**
Porque dos síntomas del mismo paciente (como fiebre y tos) comparten el mismo proceso inflamatorio y están correlacionados entre sí; si los multiplicaras como pruebas separadas violarías el supuesto matemático de **independencia condicional**. En cambio, las guías del CDC Perú y del MINSA ya miden la prevalencia real de la enfermedad dentro de cada grupo sintomático (por ejemplo, `4.5 %` en sintomáticos respiratorios de TB), lo que da un punto de partida $t = 0$ validado epidemiológicamente, y luego se encadenan pruebas de laboratorio que sí usan mecanismos biológicos independientes (rayos X, microscopía y ADN).

**2. "¿Por qué al invertir el orden de las pruebas cambia la curva del medio pero el resultado final da exactamente igual?"**
Porque bajo el supuesto de independencia condicional (§1.3.4), la probabilidad conjunta de obtener una secuencia de resultados $(O_1, \dots, O_n)$ es simplemente la multiplicación de las probabilidades de cada prueba individual: $\prod_{t=1}^n P(O_t \mid E)$ en enfermos y $\prod_{t=1}^n P(O_t \mid E^c)$ en sanos. Como la multiplicación de fracciones es conmutativa, hacer primero Antígeno y después PCR lleva exactamente al mismo valor final que hacer primero PCR y después Antígeno, aunque en la práctica clínica los hospitales empiezan por la prueba más rápida y accesible.

**3. "¿Este programa sirve para diagnosticar a un paciente real en una clínica?"**
No. El simulador incluye una advertencia permanente indicando que es una herramienta académica de modelado matemático y apoyo pedagógico. Trabaja con promedios poblacionales estáticos de literatura oficial y no reemplaza el juicio clínico de un médico.

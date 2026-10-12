# Especificación: Simulador educativo de inferencia bayesiana médica

## Supuestos acordados

- Aplicación web local en Python con Streamlit.
- Uso educativo; no se utilizará para diagnosticar ni tratar pacientes.
- Datos de ejemplo sintéticos, con valores plausibles e identificados visiblemente como simulados. No se presentarán como datos epidemiológicos validados.
- El repositorio está vacío; se puede definir su estructura desde cero.
- La inferencia usará una tabla explícita de probabilidades previas y probabilidades condicionales para las condiciones y síntomas de ejemplo.

## Objetivo

Construir un simulador interactivo que permita seleccionar presencia o ausencia de síntomas y observar cómo cambian las probabilidades posteriores de varias condiciones ilustrativas. La interfaz explicará el cálculo de Bayes y la ley de probabilidad total, y mostrará la entropía de Shannon antes y después de incorporar la evidencia.

### Criterios de aceptación

- El usuario puede marcar un síntoma como presente, ausente o no evaluado; los síntomas no evaluados no afectan el cálculo.
- Para cada condición, el motor calcula la probabilidad conjunta de la evidencia bajo el supuesto de independencia condicional de síntomas dado el diagnóstico.
- El denominador se calcula por probabilidad total, sumando sobre todas las condiciones, y las posteriores resultantes suman 1 dentro de tolerancia numérica.
- La aplicación muestra priors, probabilidades de evidencia, posteriores, entropía previa y posterior, y una comparación visual antes/después.
- La selección inicial (sin síntomas evaluados) reproduce los priors y no introduce evidencia artificial.
- Entradas inválidas o distribuciones inconsistentes se rechazan con errores claros; la interfaz no falla silenciosamente.
- Cada pantalla indica que los datos son sintéticos y que el resultado no constituye consejo ni diagnóstico médico.

## Modelo probabilístico

Sea `D` una condición del conjunto de condiciones ilustrativas, y `E` la evidencia de síntomas evaluados, cada uno con estado presente o ausente. Se define un prior `P(D)` y una probabilidad condicional `P(S_i presente | D)` por síntoma y condición. Para un síntoma ausente se usa `1 - P(S_i presente | D)`.

Con independencia condicional de los síntomas dado `D`:

```text
P(E | D) = producto_i P(E_i | D)
P(E) = suma_d P(E | d) * P(d)                  # ley de probabilidad total
P(D | E) = P(E | D) * P(D) / P(E)              # teorema de Bayes
H(D) = - suma_d P(d) * log2(P(d))              # bits
```

Cuando no se ha evaluado ningún síntoma, se usa `P(E|D)=1` para todas las condiciones y la posterior debe coincidir con el prior. Si `P(E)` es cero o numéricamente indistinguible de cero, el motor debe informar que la evidencia no es compatible con el modelo, sin dividir por cero.

La independencia condicional es una simplificación pedagógica: los síntomas clínicos pueden estar correlacionados. Las probabilidades de ejemplo serán sintéticas y plausibles, pero no deben describirse como estimaciones clínicas, ni sus resultados como aptos para decisiones médicas.

## Stack tecnológico

- Python 3.11 o superior.
- Streamlit para la interfaz interactiva.
- pytest para pruebas unitarias y de integración ligera.
- Dependencias de ejecución limitadas a Streamlit; gráficos con componentes integrados de Streamlit cuando sean suficientes.

## Comandos

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
python -m pytest
```

En Windows, activar el entorno virtual con `.venv\\Scripts\\activate`.

## Estructura del proyecto

```text
app.py                         # Punto de entrada y composición de la página Streamlit
src/diagnosis_simulator/
  __init__.py
  domain.py                    # Modelos y validación del catálogo probabilístico
  inference.py                 # Evidencia, probabilidad total y posterior
  entropy.py                   # Entropía de Shannon y unidades
  sample_data.py               # Catálogo de ejemplo sintético, documentado
  presentation.py              # Renderizado de controles, resultados y gráficos
tests/
  test_domain.py
  test_inference.py
  test_entropy.py
  test_sample_data.py
pyproject.toml                 # Metadatos, dependencias y configuración de pytest
README.md                      # Instalación, uso, límites del modelo y aviso educativo
```

La capa de dominio no dependerá de Streamlit. El motor será una función pura que reciba el catálogo y la evidencia y devuelva resultados estructurados, facilitando la validación analítica independiente.

## Estilo de código

Se usarán nombres descriptivos en inglés para identificadores Python, anotaciones de tipo, funciones pequeñas y docstrings en español para interfaces públicas. Los comentarios estarán en español y explicarán decisiones o pasos probabilísticos clave, sin repetir lo obvio en cada línea.

```python
def posterior_probabilities(
    priors: dict[str, float], likelihoods: dict[str, float]
) -> dict[str, float]:
    """Aplica Bayes y normaliza las condiciones mediante probabilidad total."""
    evidence_probability = sum(
        priors[condition] * likelihoods[condition]
        for condition in priors
    )
    if evidence_probability <= 0:
        raise ValueError("La evidencia seleccionada tiene probabilidad cero en el modelo.")
    return {
        condition: priors[condition] * likelihoods[condition] / evidence_probability
        for condition in priors
    }
```

## Estrategia de pruebas

- Pruebas unitarias del dominio: probabilidades dentro de `[0, 1]`, priors normalizados, catálogo completo y errores para datos inválidos.
- Pruebas unitarias de inferencia: resultado sin evidencia igual a los priors; evidencia presente y ausente usa la verosimilitud correcta; denominador por probabilidad total; posteriores normalizadas; y manejo de evidencia de probabilidad cero.
- Pruebas de entropía: distribución uniforme, distribución determinista, resultado en bits y validación de distribuciones.
- Casos de ejemplo contrastados con cálculos analíticos independientes escritos en las pruebas, incluidos al menos dos síntomas y una ausencia.
- `python -m pytest` como verificación local requerida antes de considerar terminada la implementación.

## Límites

- **Siempre:** Mantener la inferencia independiente de la interfaz; validar el catálogo y entradas; mostrar el aviso educativo y el origen sintético de los datos; comentar las decisiones matemáticas clave; contrastar casos con cálculos analíticos.
- **Pedir confirmación antes de:** Incorporar datos personales o clínicos reales; añadir almacenamiento, cuentas de usuario, llamadas a servicios externos, telemetría o despliegue público; cambiar el alcance a soporte de decisiones clínicas.
- **Nunca:** Presentar datos simulados como mediciones reales; declarar la salida como diagnóstico o recomendación terapéutica; ocultar las limitaciones del supuesto de independencia condicional; registrar información identificable de pacientes.

## Éxito

La implementación estará completa cuando la aplicación arranque con el comando indicado, permita ajustar síntomas sin recargar errores, presente probabilidades y entropía consistentes con las fórmulas documentadas, y todas las pruebas analíticas pasen. La interfaz deberá dejar claro en todo momento que el catálogo es sintético y el simulador no es una herramienta clínica.

## Preguntas abiertas

- Catálogo inicial propuesto: un conjunto pequeño de condiciones respiratorias ilustrativas y síntomas comunes. Si no se especifica otro conjunto, se escogerá un catálogo reducido que facilite explicar el modelo y evite sugerir cobertura médica exhaustiva.
- El realismo se limitará a patrones plausibles para una demostración. No se afirmará calibración epidemiológica sin una fuente de datos validada y autorización para incorporarla.

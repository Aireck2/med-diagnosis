# Simulador educativo de inferencia bayesiana médica

Aplicación local e interactiva para explorar cómo los síntomas modifican una distribución de probabilidades sobre condiciones ilustrativas. Usa Python, Streamlit, el teorema de Bayes y la entropía de Shannon.

[English](README.md) · [Español](README.es.md)

> [!WARNING]
> Este proyecto es solo educativo. Las condiciones y probabilidades son ejemplos sintéticos, no datos epidemiológicos ni clínicos validados. El resultado no es un diagnóstico médico ni una recomendación de tratamiento. No lo uses para tomar decisiones de salud.

## Funciones

- Marca cada síntoma como **Presente**, **Ausente** o **No evaluado**. Los síntomas no evaluados se excluyen.
- Compara las probabilidades previas y posteriores de cuatro condiciones respiratorias ilustrativas.
- Consulta la probabilidad de la evidencia seleccionada, la verosimilitud de cada condición y su contribución a la probabilidad total.
- Compara la entropía de Shannon previa y posterior en bits.
- Explora un modelo bayesiano explicado y un catálogo sintético compacto.

## Requisitos

- Python 3.11 o superior
- `pip`

## Instalación y ejecución

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

En Windows, activa el entorno con `.venv\Scripts\activate` y luego ejecuta los comandos de instalación y arranque anteriores.

## Modelo

Para una condición `D` y la evidencia `E` de los síntomas evaluados, el simulador supone que los síntomas son condicionalmente independientes dado `D`:

```text
P(E|D) = producto_i P(E_i|D)
P(E)   = suma_d P(E|d) * P(d)             # ley de probabilidad total
P(D|E) = P(E|D) * P(D) / P(E)            # teorema de Bayes
H(D)   = -suma_d P(d) * log2(P(d))        # entropía de Shannon, en bits
```

Para un síntoma ausente se usa el complemento de su probabilidad de presencia. Si no se evalúa ningún síntoma, la posterior coincide con la previa. El cálculo usa logaritmos y log-sum-exp para mantener la estabilidad cuando las verosimilitudes son muy pequeñas.

## Limitaciones

La independencia condicional es un supuesto simplificador; en la realidad, los síntomas pueden estar correlacionados. Los priors y las probabilidades condicionales del catálogo de ejemplo se inventaron para la demostración: no son estimaciones derivadas de historias clínicas, estudios ni poblaciones. Las condiciones de ejemplo no constituyen una cobertura médica exhaustiva.

## Estructura del proyecto

```text
app.py                              # Punto de entrada de Streamlit
src/diagnosis_simulator/
  domain.py                         # Tipos y validación del catálogo
  inference.py                      # Verosimilitudes y probabilidades posteriores
  entropy.py                        # Entropía de Shannon
  sample_data.py                    # Catálogo sintético de ejemplo
  presentation.py                   # Controles y resultados de Streamlit
tests/                              # Pruebas unitarias con casos analíticos
README.md                           # Documentación principal (inglés)
README.es.md                        # Documentación en español
```

## Ejecutar las verificaciones

```bash
python -m pytest
```

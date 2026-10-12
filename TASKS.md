# Tareas de implementación

Ordenadas por dependencias; cada tarea tiene una verificación propia y acota los archivos que modifica.

- [x] **Tarea 1: Crear estructura y configuración del paquete**
  - Aceptación: paquete importable `diagnosis_simulator`, metadatos para Python 3.11+, Streamlit en ejecución y pytest en desarrollo.
  - Verificar: `python -m pip install -e ".[dev]"` y `python -c "import diagnosis_simulator"`.
  - Archivos: `pyproject.toml`, `src/diagnosis_simulator/__init__.py`, `.gitignore`.

- [x] **Tarea 2: Definir modelos y validaciones del dominio**
  - Aceptación: estructuras tipadas para catálogo/evidencia y validaciones de rangos, integridad y normalización de priors; docstrings en español.
  - Verificar: `python -m pytest tests/test_domain.py` cubre catálogos válidos e inválidos.
  - Archivos: `src/diagnosis_simulator/domain.py`, `tests/test_domain.py`.

- [x] **Tarea 3: Implementar inferencia bayesiana estable**
  - Aceptación: combina síntomas presentes y ausentes con log-verosimilitudes, calcula denominador por probabilidad total y posterior normalizada; evidencia vacía retorna priors y el caso imposible produce error explícito.
  - Verificar: `python -m pytest tests/test_inference.py` contrasta casos analíticos independientes, incluye ausencia, evidencia múltiple y extremos numéricos.
  - Archivos: `src/diagnosis_simulator/inference.py`, `tests/test_inference.py`.

- [x] **Tarea 4: Implementar entropía de Shannon**
  - Aceptación: calcula bits, incluyendo distribuciones uniformes, deterministas y términos de probabilidad cero; valida entrada.
  - Verificar: `python -m pytest tests/test_entropy.py` compara valores con resultados analíticos.
  - Archivos: `src/diagnosis_simulator/entropy.py`, `tests/test_entropy.py`.

- [x] **Tarea 5: Crear y validar catálogo sintético**
  - Aceptación: catálogo respiratorio compacto, completo y normalizado; condiciones y parámetros claramente etiquetados como sintéticos.
  - Verificar: `python -m pytest tests/test_sample_data.py` valida la estructura y límites sin presentarlos como datos clínicos.
  - Archivos: `src/diagnosis_simulator/sample_data.py`, `tests/test_sample_data.py`.

- [x] **Tarea 6: Construir la interfaz Streamlit**
  - Aceptación: controles de presente/ausente/no evaluado, tabla de resultados, comparación gráfica prior/posterior, entropía inicial/final, explicación y aviso educativo visibles; sin fórmulas duplicadas en la presentación.
  - Verificar: `streamlit run app.py` inicia y su endpoint de salud respondió `ok`; `python -m pytest` sigue pasando.
  - Archivos: `app.py`, `src/diagnosis_simulator/presentation.py`.

- [x] **Tarea 7: Documentar el uso y ejecutar verificación final**
  - Aceptación: README en español con instalación, ejecución, modelo, supuestos y límites; suite completa aprobada y aplicación inicia.
  - Verificar: `python -m pytest` y `streamlit run app.py`.
  - Archivos: `README.md`.

## Convenciones de implementación

- Comentarios y docstrings del código en español; identificadores Python descriptivos en inglés y anotaciones de tipo.
- En cada paso probabilístico importante se documentará la decisión matemática correspondiente.
- No agregar persistencia, autenticación, telemetría ni dependencias de visualización adicionales.

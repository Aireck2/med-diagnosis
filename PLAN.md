# Plan técnico: Simulador educativo bayesiano

Este plan implementa la especificación aprobada en `SPEC.md`. No incluye datos personales, fuentes externas ni funciones de apoyo a decisiones clínicas.

## Componentes y dependencias

1. **Dominio (`domain.py`)**
   - Definir modelos tipados para condiciones, síntomas, evidencia y catálogo.
   - Validar que los priors sumen 1 dentro de una tolerancia, que las probabilidades estén en `[0, 1]` y que el catálogo tenga parámetros completos.
   - No depende de Streamlit.

2. **Motor probabilístico (`inference.py`)**
   - Recibir catálogo validado y estados de síntomas (`present`, `absent`, `unknown`).
   - Omitir los síntomas `unknown`; para los evaluados, multiplicar sus probabilidades condicionales dada cada condición.
   - Calcular `P(E)` mediante suma ponderada de verosimilitudes y priors; detectar denominador cero; normalizar posteriores.
   - Devolver un resultado tipado con prior, verosimilitud, contribución al denominador, posterior y probabilidad de evidencia.

3. **Entropía (`entropy.py`)**
   - Implementar entropía de Shannon en bits sobre distribuciones validadas.
   - Devolver cero para distribuciones deterministas; definir la contribución de términos con probabilidad cero como cero.

4. **Catálogo sintético (`sample_data.py`)**
   - Declarar un catálogo compacto de condiciones respiratorias y síntomas comunes, con priors y probabilidades condicionales explícitos.
   - Documentar que son parámetros didácticos sintéticos, no prevalencias ni sensibilidad/especificidad clínicas validadas.
   - Mantener los datos separados del motor para permitir futuras sustituciones auditables.

5. **Interfaz (`presentation.py`, `app.py`)**
   - Construir controles dinámicos para los estados presente, ausente y no evaluado.
   - Renderizar explicación breve del modelo y aviso educativo visible.
   - Mostrar tabla de prior/posterior, probabilidad de evidencia y entropías antes/después.
   - Mostrar gráfico comparativo previo/posterior y actualizarlo al cambiar cualquier selección.
   - Mantener el punto de entrada `app.py` delgado; la presentación llamará al dominio y al motor, sin duplicar fórmulas.

6. **Empaquetado y documentación (`pyproject.toml`, `README.md`)**
   - Declarar Python mínimo, Streamlit y pytest como dependencia de desarrollo.
   - Documentar instalación, ejecución, supuestos, interpretación y limitaciones en español.

## Orden de implementación

1. Crear estructura del paquete y metadatos del proyecto.
2. Implementar tipos y validación del dominio; agregar pruebas de límites y errores.
3. Implementar cálculo de evidencia, probabilidad total y posterior; validar con ejemplos analíticos manuales.
4. Implementar entropía y sus casos límite; contrastar distribuciones conocidas.
5. Crear catálogo sintético y probar integridad y normalización.
6. Integrar controles y resultados en Streamlit, incluyendo el aviso y gráfico comparativo.
7. Escribir README y ejecutar la suite completa y una comprobación de arranque/importación de Streamlit.

## Riesgos y mitigaciones

- **Síntomas correlacionados:** el producto de verosimilitudes puede sobrevalorar evidencia repetida. Mostrar el supuesto en la interfaz y el README; mantener el simulador como demostración educativa.
- **Parámetros con apariencia clínica:** valores plausibles podrían confundirse con estimaciones reales. Etiquetar el catálogo como sintético en datos, interfaz y documentación; no usar lenguaje de sensibilidad o prevalencia validada.
- **Probabilidad de evidencia extremadamente pequeña:** las multiplicaciones pueden sufrir subdesbordamiento. Usar sumas de log-verosimilitudes para combinar síntomas y estabilizar la normalización (log-sum-exp), o detectar explícitamente el límite numérico; incluir un caso extremo en pruebas.
- **Selecciones sin evidencia:** no mostrar cambios artificiales. Garantizar que posterior sea igual al prior y que la entropía no cambie.
- **Estado interactivo incoherente:** usar valores por defecto válidos y regenerar resultados a partir del estado actual de controles en cada renderizado.

## Puntos de verificación

- **V1 – Dominio:** validación del catálogo acepta el ejemplo válido y rechaza probabilidades inválidas.
- **V2 – Inferencia:** un caso de una condición y un síntoma concuerda con cálculo analítico; ausencia usa el complemento; evidencia vacía devuelve priors.
- **V3 – Incertidumbre:** entropía uniforme y determinista coincide con valores analíticos; las comparaciones antes/después se calculan sobre distribuciones correctas.
- **V4 – Integración:** la app inicia, muestra advertencias, acepta estados de síntomas y representa cambios de resultados sin excepciones.
- **V5 – Entrega:** README contiene comandos reproducibles y suite completa aprobada.

## Decisiones por confirmar en esta revisión

- Aceptar el enfoque de log-verosimilitudes para resistencia a subdesbordamiento numérico.
- Aceptar el catálogo compacto de condiciones respiratorias como conjunto inicial.

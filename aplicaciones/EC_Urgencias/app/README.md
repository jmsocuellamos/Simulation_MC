# Simulador de la red de colas de Urgencias (Streamlit)

Aplicación para configurar, simular (Monte Carlo) y comparar escenarios del Servicio de Urgencias
del caso de estudio *01 Red de Colas Hospitalaria*, y generar un informe Word con los resultados.

## Instalación y ejecución

```bash
cd app_urgencias
python -m venv .venv && source .venv/bin/activate      # opcional (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre en el navegador (por defecto en http://localhost:8501).

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `motor.py` | Gemelo digital SimPy (el mismo del cuaderno) y configuraciones por defecto de la Línea base y de los escenarios A, B y C, más `combinar_configs` |
| `kpis.py` | KPIs, SLAs, Monte Carlo (en serie o en paralelo), comparación pareada, ranking y textos automáticos |
| `validacion.py` | Comprobaciones de coherencia de cada escenario y de las franjas horarias |
| `graficos.py` | Gráficos matplotlib/seaborn |
| `informe.py` | Generación del informe `.docx` |

## Uso

1. **Parámetros comunes**: son los mismos para todos los escenarios: llegadas por franja horaria (tasa y
   reparto por perfil), reembolsos, costes variables, penalizaciones, tramos de margen exigido y umbrales de SLA.
2. **Escenarios**: cada escenario tiene sus propios recursos y capacidades (servidores, camas, Box, UCI),
   modo de gestión de la UCI, tiempos de servicio, probabilidades de derivación y costes fijos (anuales o únicos).
   Se pueden crear escenarios nuevos copiando uno existente, combinar varios escenarios sobre una base,
   renombrarlos, eliminarlos o restablecerlos. La pestaña *Validación* avisa de cualquier incoherencia:
   filas de derivación que no suman 100 %, tiempos que faltan, estaciones sin servidor…
3. **Barra lateral**: número de réplicas, días por réplica, semilla de inicialización, días de calentamiento y
   de cierre descartados, ejecución en paralelo, escenarios que se simulan y escenario de referencia. La réplica *i*
   de todos los escenarios usa la semilla *s + i* (números aleatorios comunes), lo que permite hacer comparaciones
   pareadas.
4. **Resultados**:
   - síntesis automática y tabla comparativa con ▲/▼ cuando el cambio frente a la referencia es significativo;
   - SLAs, ranking y resultado económico;
   - gráficos;
   - detalle por escenario;
   - descarga de las réplicas en Excel.
5. **Informe**: genera y descarga un `.docx` que contiene:
   - el resumen ejecutivo;
   - los parámetros comunes;
   - los parámetros de cada escenario y sus diferencias con la referencia;
   - las tablas y gráficas comparativas;
   - un capítulo de detalle por escenario.

La configuración completa se puede **guardar y cargar en JSON** desde la barra lateral, para repetir un estudio.

## Notas

- Las estaciones cuyo nombre empieza por `Obs_` se tratan como observación: no tienen recurso propio y el
  paciente retiene la sala o el Box que ya ocupaba.
- Los perfiles de paciente son fijos: Verde, Amarillo, Naranja y Rojo.
- Con 50 réplicas de 180 días por escenario, cada escenario tarda algunos minutos. La ejecución en paralelo
  reparte las réplicas entre los núcleos del ordenador; si el entorno no lo permite, continúa en serie.

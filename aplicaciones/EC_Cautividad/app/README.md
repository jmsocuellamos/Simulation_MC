# Simulador del centro de cría en cautividad y reintroducción (Streamlit)

Aplicación para configurar, simular (Monte Carlo) y comparar escenarios del caso de estudio
*02 Centro de Cría en Cautividad y Reintroducción*, y para generar un informe Word con los resultados.
Todos los tiempos se expresan en **días**.

## Instalación y ejecución

```bash
cd app_conservacion
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `motor.py` | Gemelo digital SimPy (el mismo que el del cuaderno), configuraciones de la Línea base y de los escenarios A (IA genómica), B (Rescate rápido) y C (UCI compartida), y `combinar_configs` |
| `kpis.py` | KPIs operativos, de recursos, biológicos y económicos; SLAs; Monte Carlo; comparación pareada; ranking y textos automáticos |
| `validacion.py` | Comprobaciones de coherencia de los escenarios y de las estaciones del año |
| `graficos.py` | Gráficos |
| `informe.py` | Generación del informe `.docx` |

## Uso

1. **Parámetros comunes**:
   - ingresos por estación del año: tasa en individuos/día y reparto por perfil, con periodos definidos por días del año;
   - subvención por desenlace (Liberado o Cautividad permanente) y penalización por fallecimiento, según perfil;
   - costes por procedimiento y por día de estancia;
   - margen exigido y umbrales de los seis SLAs.
2. **Escenarios**: cada escenario tiene sus propios parámetros:
   - servidores, jaulas y Recinto de Recuperación, con sus capacidades, y la estación en la que cada perfil ocupa el Recinto;
   - modo de gestión de la UCI Veterinaria (bolsas reservadas o bolsa compartida con prioridad), plazas y límite de espera;
   - tiempos de servicio, probabilidades de derivación y costes fijos (anuales o únicos).

   Se pueden crear escenarios nuevos copiando otro, combinar varios sobre una base, renombrarlos,
   eliminarlos o restablecerlos.
3. **Barra lateral**: réplicas, días por réplica (por defecto 3.650 = 10 años), semilla, días de calentamiento
   (365) y de cierre (730) descartados, escenarios a simular y escenario de referencia.
4. **Resultados**:
   - síntesis automática;
   - tabla comparativa con ▲/▼ cuando el cambio frente a la referencia es significativo;
   - SLAs, ranking y resultado económico;
   - gráficos;
   - detalle por escenario;
   - descarga de las réplicas en Excel.
5. **Informe**: un `.docx` con todas las tablas y gráficas de los escenarios y sus comparativas.

La configuración completa se guarda y se carga en JSON desde la barra lateral.

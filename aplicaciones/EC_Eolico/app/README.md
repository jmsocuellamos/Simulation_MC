# Simulador del taller de mantenimiento y reacondicionamiento eólico (Streamlit)

Aplicación para configurar, simular (Monte Carlo) y comparar escenarios del caso de estudio
*03 Taller Eólico*, y para generar un informe Word con los resultados.
Todos los tiempos se expresan en **días**.

## Instalación y ejecución

```bash
cd app_taller
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `motor.py` | Gemelo digital SimPy (el mismo que el del cuaderno), configuraciones de la Línea base y de los escenarios A (IA predictiva), B (Respuesta rápida) y C (Contención compartida), y `combinar_configs` |
| `kpis.py` | KPIs operativos, de recursos, de fiabilidad y económicos; SLAs; Monte Carlo; comparación pareada; ranking y textos automáticos |
| `validacion.py` | Comprobaciones de coherencia de los escenarios y de las estaciones del año |
| `graficos.py` | Gráficos |
| `informe.py` | Generación del informe `.docx` |

## Uso

1. **Parámetros comunes**:
   - ingresos por estación del año: tasa en órdenes/día y reparto por perfil, con periodos definidos por días del año;
   - tarifa por desenlace (Operativo o Baja definitiva) y penalización por pérdida total, según perfil;
   - costes por procedimiento y por día de estancia;
   - margen exigido y umbrales de los seis SLAs.
2. **Escenarios**: cada escenario tiene sus propios parámetros:
   - servidores, bahías y Bahía Urgente, con sus capacidades, y la estación en la que cada perfil ocupa la Bahía Urgente;
   - modo de gestión de la Contención de Emergencia (bolsas reservadas o bolsa compartida con prioridad), plazas y límite de espera;
   - tiempos de servicio, probabilidades de derivación y costes fijos (anuales o únicos).

   Se pueden crear escenarios nuevos copiando otro, combinar varios sobre una base, renombrarlos,
   eliminarlos o restablecerlos.
3. **Barra lateral**: réplicas, días por réplica (por defecto 1.825 = 5 años), semilla, días de calentamiento
   (60) y de cierre (90) descartados, escenarios a simular y escenario de referencia.
4. **Resultados**:
   - síntesis automática;
   - tabla comparativa con ▲/▼ cuando el cambio frente a la referencia es significativo;
   - SLAs, ranking y resultado económico;
   - gráficos;
   - detalle por escenario;
   - descarga de las réplicas en Excel.
5. **Informe**: un `.docx` con todas las tablas y gráficas de los escenarios y sus comparativas.

La configuración completa se guarda y se carga en JSON desde la barra lateral.

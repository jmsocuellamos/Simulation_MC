# Simulador de la consultora de reestructuración de deuda (Streamlit)

Aplicación para configurar, simular (Monte Carlo) y comparar escenarios del caso de estudio
*04 Reestructuración de deuda*, y para generar un informe Word con los resultados.
Todos los tiempos se expresan en **días**.

## Instalación y ejecución

```bash
cd app_consultora
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `motor.py` | Gemelo digital SimPy (el mismo que el del cuaderno), configuraciones de la Línea base y de los escenarios A (Analítica predictiva), B (Intervención temprana) y C (Turnos compartidos), y `combinar_configs` |
| `kpis.py` | KPIs operativos, de recursos, de recuperación financiera y económicos; SLAs; Monte Carlo; comparación pareada; ranking y textos automáticos |
| `validacion.py` | Comprobaciones de coherencia de los escenarios y de los periodos del año |
| `graficos.py` | Gráficos |
| `informe.py` | Generación del informe `.docx` |

## Uso

1. **Parámetros comunes**:
   - ingresos por periodo del año: tasa en expedientes/día y reparto por perfil, con periodos definidos por días del año;
   - honorarios por desenlace (Regularizado o Liquidación ordenada), coste de oportunidad por insolvencia total y penalización por demora del riesgo concursal;
   - costes por procedimiento y por día de estancia;
   - margen exigido y umbrales de los seis SLAs.
2. **Escenarios**: cada escenario tiene sus propios parámetros:
   - servidores, mesas y Mesa Prioritaria, con sus capacidades, y la estación en la que cada perfil ocupa la Mesa Prioritaria;
   - modo de gestión de la Intervención Judicial (bolsas reservadas o bolsa compartida con prioridad), plazas y límite de espera;
   - tiempos de servicio, probabilidades de derivación y costes fijos (anuales o únicos).

   Se pueden crear escenarios nuevos copiando otro, combinar varios sobre una base, renombrarlos,
   eliminarlos o restablecerlos.
3. **Barra lateral**: réplicas, días por réplica (por defecto 3.650 = 10 años), semilla, días de calentamiento
   (180) y de cierre (365) descartados, escenarios a simular y escenario de referencia.
4. **Resultados**:
   - síntesis automática;
   - tabla comparativa con ▲/▼ cuando el cambio frente a la referencia es significativo;
   - SLAs, ranking y resultado económico;
   - gráficos;
   - detalle por escenario;
   - descarga de las réplicas en Excel.
5. **Informe**: un `.docx` con todas las tablas y gráficas de los escenarios y sus comparativas.

La configuración completa se guarda y se carga en JSON desde la barra lateral.

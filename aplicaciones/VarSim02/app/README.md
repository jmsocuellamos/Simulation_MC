# Riesgo y retorno de planes de inversión (Streamlit)

Aplicación para comparar planes de inversión cuyo retorno R es una variable discreta (valores vᵢ con probabilidades
pᵢ) generada por el **método de la transformada inversa**. Incluye la teoría exacta (retorno esperado, riesgo σ,
probabilidad de pérdida, percentiles), la simulación Monte Carlo con réplicas, la comparación de planes (escenarios)
con objetivos y el gráfico de riesgo frente a retorno, la sensibilidad del estimador frente a nsim y un informe Word.

Basado en el Ejemplo 2 de «B-70. Distribuciones derivadas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS
(CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_inversion
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `inversion.py` | Teoría, transformada inversa, simulación, objetivos, sensibilidad y textos automáticos |
| `graficos.py` | Gráficos (escalera F(i), distribución y función de distribución del retorno, simulaciones, réplicas, riesgo-retorno, comparativas, sensibilidad) |
| `informe.py` | Informe `.docx` (las tablas ajustan el ancho de sus columnas para caber siempre en la página) |

## Uso

1. **Parámetros comunes**: inversión inicial I₀ y objetivos (rentabilidad esperada mínima, probabilidad de pérdida
   máxima, riesgo σ máximo y resultado neto mínimo en el peor 5 %; cada uno se puede desactivar).
2. **Planes**: cada plan es una tabla de tramos (retorno y probabilidad, hasta 8; se normalizan si no suman 100 %) y
   una descripción. Se pueden copiar, renombrar y eliminar. La vista previa teórica incluye el mecanismo de la
   transformada inversa con una U de ejemplo (deslizador o 🎲 U aleatorio).
3. **Barra lateral**: simulaciones por réplica (nsim), réplicas, semilla y planes a simular; guardar y cargar la
   configuración en JSON. Todos los planes usan las mismas semillas (números aleatorios comunes).
4. **Resultados**: síntesis y recomendación; riesgo frente a retorno; comparación y objetivos; detalle por plan
   separando lo **teórico** (📐) de lo **simulado** (🎲): KPIs, réplicas, primeras simulaciones, tramos, distribución
   y función de distribución del retorno, calculadora de probabilidades y percentiles; descarga en Excel.
5. **Sensibilidad**: barrido logarítmico de nsim con varias réplicas por punto: convergencia hacia el retorno teórico
   y σ entre réplicas frente a la predicción σ(R)/√nsim.
6. **Informe**: un `.docx` con todas las tablas y gráficas, incluida (opcionalmente) la sensibilidad.

## Modelo

* F(i) = p₁ + … + pᵢ ; con U ~ Unif(0, 1), R = vᵢ para el menor i tal que U ≤ F(i).
* E(R) = Σ pᵢ·vᵢ ; σ(R) = √(Σ pᵢ·(vᵢ − E(R))²) ; P(pérdida) = P(R < I₀) ; rentabilidad = (E(R) − I₀)/I₀.

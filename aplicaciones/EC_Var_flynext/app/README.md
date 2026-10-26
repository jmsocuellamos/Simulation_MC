# EC1 · Optimización de costes en «Fly-Next» (Streamlit)

Aplicación para resolver el estudio de caso **EC1 «Fly-Next»** (generación de variables aleatorias):

* **Bloque A · coste de una entrega.** Método de composición: la ruta T (Urbana, Regional, Premium) se elige por
  transformada inversa y, condicionadas a ella, se simulan la distancia D (Poisson o normal), el consumo G (uniforme)
  y el retraso M (exponencial). Y = D·G·P_fuel + M·C_labor + peaje(D).
* **Bloque B · coste diario con franjas.** N_h ~ Poisson(λ_franja) pedidos por hora; el tráfico multiplica G y M
  según la franja y la ruta; Y_con e Y_sin se calculan con las mismas variables base (simulación pareada) y
  Δ = Y_con − Y_sin es el sobrecoste por tráfico.

Cada resultado se da **teórico** (📐: momentos exactos, distribución de Y por integración numérica, coste diario como
suma compuesta de Poisson) y **simulado** (🎲: macro-réplicas con IC95 %), con los objetivos del encargo:

| Objetivo | Dónde |
| --- | --- |
| O1.1–O1.3 · E(Y), P(Y > 150), percentil 90, análisis por ruta y peso en la cola | Resultados → Bloque A |
| O2.1 · descomposición de Var(Y) por términos | Resultados → Bloque A |
| O2.2 · sensibilidad de Var(Y) a P_fuel, C_labor y K (elasticidades) | Sensibilidad → parámetros económicos |
| O3.1 · mezcla de rutas más cercana a la actual con un 10 % menos de coste | Optimización |
| O3.2–O3.4 · peaje proporcional, K −20 % (ahorro anual), cambio de proveedor de combustible (A/B pareado) | Escenarios y Resultados → Ahorros |
| O4.1–O4.3 · coste por franja, coste y sobrecoste diario, percentil 95 y CVaR 5 % | Resultados → Bloque B |
| O5.1–O5.3 · desplazar urbanos de Punta a Valle, externalizar Premium, equilibrio Punta/Llano | Escenarios, Resultados → Bloque B y Ahorros |
| O6.1–O6.3 · capacidad horaria, P(coste diario > presupuesto), límite de capacidad y pedidos rechazados | Resultados → Bloque B |
| O7.1 · recomendación y cuadro de mando | Resultados → Síntesis e Informe |
| O8.1 y O8.4 · pedidos diarios Poisson(Σλ) y ganancia del diseño pareado | Resultados → Bloque B |

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_flynext
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit (parámetros comunes, escenarios, resultados, optimización, sensibilidad e informe) |
| `flynext.py` | Modelo: teoría exacta, simulación por composición con números aleatorios comunes, objetivos, ahorros, optimización y sensibilidad |
| `graficos.py` | Gráficos (mecanismo, distribución de Y, rutas, descomposición, coste diario, franjas, perfil horario, comparativas, ahorros, optimización, sensibilidad) |
| `informe.py` | Informe `.docx` organizado por objetivos (las tablas ajustan el ancho de sus columnas a la página) |

## Uso

1. **Parámetros comunes**: distribuciones condicionadas por ruta, tasas λ y factores de tráfico por franja, umbral de
   entrega cara (150 €), entregas al año (50.000), presupuesto diario (30.000 €), capacidad horaria opcional,
   reducción objetivo de O3.1 y objetivos del cuadro de mando.
2. **Escenarios**: mezcla de rutas, P_fuel, C_labor, peaje fijo o proporcional, consumo del proveedor, desplazamiento
   de urbanos de Punta a Valle, reducción del retraso en Punta, externalización de Premium y coste de implantación. La
   vista previa teórica incluye el mecanismo de una entrega (U_T, U_D, U_G, U_M con deslizadores o 🎲 al azar).
3. **Barra lateral**: entregas y réplicas del Bloque A, días y réplicas del Bloque B, semilla, escenario base para los
   ahorros y escenarios a simular; guardar y cargar la configuración en JSON.
4. **Resultados, Optimización, Sensibilidad e Informe**: ver la tabla de objetivos anterior.

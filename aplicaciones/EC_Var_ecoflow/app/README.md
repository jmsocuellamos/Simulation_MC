# EC4 · Gestión de riesgos en la micro-red de «Eco-Flow Renewables» (Streamlit)

Aplicación para resolver el estudio de caso **EC4 «Eco-Flow»** (variables aleatorias y simulación). Energía en MWh y
dinero en €.

* **Bloque A · un día.** Método de composición: el escenario climático T (Estival, Frontal, Monzónico, Anticiclónico)
  se elige por transformada inversa y, condicionadas a él, se simulan la solar G_s ~ Tri(a, m, b), la eólica
  G_w ~ Weibull(k, λ), la hidráulica G_h ~ LN(μ, σ) y la demanda D ~ N(μ, σ). Con B = G − D,
  Y = P_exp·B⁺ − P_pen·B⁻ − C_fijo − c·G.
* **Bloque B · batería.** Capacidad Cap y eficiencia η; se recorre día a día durante 365 días: carga con el superávit,
  cubre el déficit con E·η y el resto se compra en el mercado. Con y sin batería se aplican sobre la misma base
  climática (diseño pareado). Las estimaciones son medias **por día**; los años son solo macro-réplicas para el IC95 %.

Cada resultado se da **teórico** (📐: distribución exacta de G por convolución FFT de las tres fuentes e integración en
D, de la que salen E, Var, su descomposición, P(B < 0), VaR y CVaR; la batería como cadena de Markov en el estado de
carga, de la que salen la ganancia, la carga media día a día, P(compra) y el LOLE) y **simulado** (🎲: macro-réplicas
con IC95 %), con los objetivos del encargo:

| Objetivo | Dónde |
| --- | --- |
| O1.1–O1.3 · E(Y), P(B < 0), VaR 95 %, CVaR 5 %, CV por escenario | Resultados → Bloque A |
| O2.1 · descomposición de Var(Y) en mercado P·B y coste variable | Resultados → Bloque A |
| O2.2 · elasticidades de Var(Y) e índices de Sobol | Sensibilidad → parámetros y Sobol |
| O3.1, O3.2, O3.4 · spot volátil, más solar, contrato a plazo, respuesta de demanda | Escenarios y Resultados → Diferencias |
| O3.3 · solar adicional para E(Y) ≥ 0 | Sensibilidad → solar |
| O4.1–O4.3 · beneficio con y sin batería, ganancia pareada, P(compra), CVaR | Resultados → Bloque B |
| O5.1–O5.3 · capacidad, eficiencia y ganancia por escenario climático | Batería y Resultados → Bloque B |
| O6.1–O6.3 · capacidad óptima con amortización, racha de 20 días Anticiclónicos, LOLE | Batería y Resultados → Bloque B |
| O7.1 · recomendación y cuadro de mando | Resultados → Síntesis e Informe |
| O8.1–O8.6 · contraste, carga inicial, gestión activa, información meteorológica, servicios de red, batería frente a solar | Ampliación (O8) |

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_ecoflow
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit (parámetros comunes, escenarios, resultados, batería, sensibilidad, ampliación e informe) |
| `ecoflow.py` | Modelo: teoría exacta en rejilla, cadena de Markov de la batería, simulación con números aleatorios comunes, objetivos, diferencias, barridos, sensibilidad y ampliación |
| `graficos.py` | Gráficos (mecanismo, distribución de Y, escenarios, descomposición, estado de carga, ganancia por escenario, comparativas, barridos, racha, sensibilidad) |
| `informe.py` | Informe `.docx` organizado por objetivos (las tablas ajustan el ancho de sus columnas a la página) |

## Supuestos propios (editables)

* η de referencia = 0,98 (el enunciado da 0,90 en el texto, 0,98 en la tabla y 0,95 en el código; O5.2 toma 0,98).
* Spot volátil (O3.1): lognormal de media 80 €/MWh y CV del 50 %.
* Ampliación solar (O3.2): +30 % de capacidad; contrato a plazo a 60 €/MWh con una prima de 150 €/día.
* Respuesta de demanda (O3.4): −10 % de D si el déficit supera 20 MWh, con un descuento de 20 € por MWh no consumido.
* Amortización de la batería (O6.1): 1.800 € por MWh y año. Coste de la solar (O8.6): 9.000 € por MWh/día y año.
* Gestión activa (O8.3): precio de exportación lognormal de media P_exp y CV del 30 %.

## Uso

1. **Parámetros comunes**: distribuciones de solar, eólica, hidráulica y demanda por escenario y objetivos.
2. **Escenarios**: probabilidades del clima, precios, costes, palancas (spot volátil, ampliación solar, contrato,
   respuesta de demanda) y batería (capacidad, eficiencia, carga inicial, amortización, tarifa de servicios de red);
   copiar, renombrar y eliminar. La vista previa teórica incluye el mecanismo de un día (U_T, U_s, U_w, U_h, U_D con
   deslizadores o 🎲).
3. **Barra lateral**: días y réplicas del Bloque A, años del Bloque B, semilla, escenario base y escenarios a simular;
   guardar y cargar la configuración en JSON o restablecer los valores por defecto.
4. **Resultados, Batería, Sensibilidad, Ampliación e Informe**: ver la tabla de objetivos anterior.

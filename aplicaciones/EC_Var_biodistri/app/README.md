# EC3 · Rentabilidad de proyectos de calidad en «Bio-Distri S.A.» (Streamlit)

Aplicación para resolver el estudio de caso **EC3 «Bio-Distri»** (generación de variables aleatorias). Todas las
cantidades están en miles de € (k€).

* **Bloque A · un proyecto.** Método de composición: el tipo T (IoT, IA, Empaque, Auditoría) se elige por
  transformada inversa y, condicionadas a él, se simulan la inversión I ~ LN(μ, σ), la reducción de merma
  R ~ Beta(α, β) y el ahorro S ~ Gamma(k, θ). Y = V_stock·R + S − I − Pen(R), con multa de 20 k€ si R < 10 %.
* **Bloque B · programa plurianual.** Cada centro mantiene su tipo H años; aprendizaje R_t = 1 − (1 − R⁰_t)·λ^(t−1) y,
  sin mantenimiento, degradación S_t = S⁰_t·δ^(t−1) (con mantenimiento, S_t = S⁰_t y se paga C_mant). VAN descontado a
  la tasa r; las dos políticas se aplican sobre la misma base de números aleatorios (diseño pareado).

Cada resultado se da **teórico** (📐: momentos exactos con la Beta truncada, la lognormal y la Gamma; distribución de Y
y del VAN por convolución FFT de las distribuciones exactas de cada término, de la que salen P(Y < 0), VaR y CVaR) y
**simulado** (🎲: macro-réplicas con IC95 %), con los objetivos del encargo:

| Objetivo | Dónde |
| --- | --- |
| O1.1–O1.3 · E(Y), P(Y < 0), VaR 95 %, CVaR 5 %, CV por tipo | Resultados → Bloque A |
| O2.1 · descomposición de Var(Y) en V·R, S, −I y penalización | Resultados → Bloque A |
| O2.2 · elasticidades de Var(Y) e índices de Sobol | Sensibilidad → parámetros y Sobol |
| O3.1, O3.2, O3.4 · penalización continua, cartera solo IoT + IA, garantía de resultados | Escenarios y Resultados → Diferencias |
| O3.3 · V_stock que necesita el Empaque | Sensibilidad → V_stock |
| O4.1–O4.3 · VAN con y sin mantenimiento, ganancia pareada, P(Δ > 0), VaR y CVaR | Resultados → Bloque B |
| O5.1–O5.3 · δ, C_mant y ganancia por tipo de proyecto | Mantenimiento y Resultados → Bloque B |
| O6.1–O6.3 · C_mant de indiferencia, racha de 3 años, horizonte de 10 o 15 años | Mantenimiento |
| O7.1 · recomendación y cuadro de mando | Resultados → Síntesis e Informe |
| O8.1–O8.6 · contraste, experiencia previa, política adaptativa, actualización bayesiana, subvención, sustitución | Ampliación (O8) y Mantenimiento |

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_biodistri
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit (parámetros comunes, escenarios, resultados, mantenimiento, sensibilidad, ampliación e informe) |
| `biodistri.py` | Modelo: teoría exacta, distribuciones en rejilla, simulación con números aleatorios comunes, objetivos, diferencias, barridos, sensibilidad y ampliación |
| `graficos.py` | Gráficos (mecanismo, trayectoria de un centro, distribución de Y y del VAN, tipos, descomposición, ganancia, comparativas, barridos, racha, sensibilidad) |
| `informe.py` | Informe `.docx` organizado por objetivos (las tablas ajustan el ancho de sus columnas a la página) |

## Supuestos propios (editables)

* Penalización continua (O3.1): 200·(0,10 − R) k€ si R < 0,10 (la multa máxima coincide con los 20 k€ del escalón).
* Cartera solo IoT + IA (O3.2): se mantiene su proporción relativa actual (58 % / 42 %).
* Política adaptativa (O8.3): se mantiene el año t si el ahorro que se observaría sin mantener cae por debajo del
  umbral; el mantenimiento restaura el ahorro y reinicia la degradación.
* Actualización bayesiana (O8.4): λ de cada centro ~ U(0,60; 0,95).
* Sustitución (O8.6): a mitad del horizonte se paga otra inversión, la curva de aprendizaje vuelve a t = 1 y el
  proyecto nuevo se mantiene.

## Uso

1. **Parámetros comunes**: distribuciones de I, R y S por tipo de proyecto y objetivos (con casilla para evaluarlos).
2. **Escenarios**: cartera de proyectos, V_stock, multa fija o proporcional, umbral, garantía, horizonte, λ, δ, C_mant,
   r, subvención, punto de partida de la curva y política elegida; copiar, renombrar y eliminar. La vista previa
   teórica incluye el mecanismo de un proyecto y su trayectoria en el programa (U_T, U_I, U_R, U_S con deslizadores o 🎲).
3. **Barra lateral**: proyectos y réplicas del Bloque A, centros y réplicas del Bloque B, semilla, escenario base y
   escenarios a simular; guardar y cargar la configuración en JSON o restablecer los valores por defecto.
4. **Resultados, Mantenimiento, Sensibilidad, Ampliación e Informe**: ver la tabla de objetivos anterior.

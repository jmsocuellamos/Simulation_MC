# EC2 · Farmacoeconomía de la terapia para la superbacteria «Staph-X» (Streamlit)

Aplicación para resolver el estudio de caso **EC2 «Staph-X»** (generación de variables aleatorias):

* **Bloque A · un paciente.** Método de composición: el fenotipo T (Rápido, Estándar, Persistente) se elige por
  transformada inversa y, condicionadas a él, se simulan la carga bacteriana L ~ Gamma(α_T; 1) y la depuración
  R ~ U(r₀, r₁). D = máx(0, ln(L + 1)/R + ε), ε ~ N(1; 0,2), e Y = C_fijo + C_día·D + C_UCI·I{D > 7}.
* **Bloque B · un brote de 90 días.** N_d ~ Poisson(λ_fase) ingresos al día (Inicio, Pico, Remisión) con la mezcla de
  fenotipos de la fase; solo B_UCI camas para los complicados de cada día y el resto a la clínica externa. Y_con e
  Y_sin se calculan con los mismos pacientes (simulación pareada) y Δ = Y_con − Y_sin es el sobrecoste por capacidad.

Cada resultado se da **teórico** (📐: discretización fina de Z = ln(L+1)/R con normal truncada, adelgazamiento de
Poisson y convoluciones exactas; el percentil 95 del brote, por aproximación normal) y **simulado** (🎲: macro-réplicas
con IC95 %), con los objetivos del encargo:

| Objetivo | Dónde |
| --- | --- |
| O1.1–O1.3 · E(Y), P(D > 7), percentil 95, CV por fenotipo | Resultados → Bloque A |
| O2.1 · descomposición de Var(Y) en estancia y salto por complicación | Resultados → Bloque A |
| O2.2 · sensibilidad global: elasticidades de Var(Y) e índices de Sobol | Sensibilidad → parámetros y Sobol |
| O3.1–O3.3 · genérico, nuevo fármaco con ICER, pago por resultados | Escenarios y Resultados → Ahorros e ICER |
| O3.4 · umbral de complicación de 5, 7 o 10 días (mismos pacientes) | Sensibilidad → umbral |
| O4.1–O4.3 · coste del brote con y sin capacidad, sobrecoste y %, desbordamientos, derivados, P95 | Resultados → Bloque B |
| O5.1–O5.3 · aislamiento (λ Pico 25 → 18), contrato por volumen, sobrecoste por fase | Escenarios, Resultados → Bloque B y Ahorros |
| O6.1–O6.3 · barrido de camas con coste de reserva, UCI llena dos días seguidos, 5 fijas frente a 3 + 4 de alquiler | Capacidad UCI y Resultados |
| O7.1 · tres recomendaciones con su métrica y cuadro de mando | Resultados → Síntesis e Informe |
| O8.1–O8.2 · ganancia del diseño pareado y modelo con ocupación de varios días | Capacidad UCI |
| O8.3, O8.4, O8.6 · camas + diagnóstico precoz con presupuesto, proceso de Cox, coste-beneficio del diagnóstico | Ampliación (O8) |
| O8.5 · cuota diaria máxima por cama alquilada en el Pico | Capacidad UCI |

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_staphx
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit (parámetros comunes, escenarios, resultados, capacidad UCI, sensibilidad, ampliación e informe) |
| `staphx.py` | Modelo: teoría exacta, simulación por composición con números aleatorios comunes, objetivos, ahorros, ICER, capacidad, sensibilidad y ampliación |
| `graficos.py` | Gráficos (mecanismo, distribución de Y, fenotipos, descomposición, brote, fases, camas, comparativas, sensibilidad, optimización) |
| `informe.py` | Informe `.docx` organizado por objetivos (las tablas ajustan el ancho de sus columnas a la página) |

## Supuestos propios (editables)

El enunciado no fija estos valores; la aplicación los propone y se pueden cambiar:

* Genérico (O3.1): carga bacteriana un 30 % mayor en todos los fenotipos.
* Coste de oportunidad de una cama reservada (O6.1, O6.3): 250 € por cama y día.
* Alquiler (O6.3b): 6.500 € por uso, sin cuota fija. Contrato por volumen (O5.2): 7.000 € desde el 16.º derivado.
* Aislamiento (O5.1): 50.000 € de implantación por brote. Diagnóstico precoz (O8.6): −30 % de carga del
  Persistente por 108.000 €; en O8.3 el coste es 12.000 €·(10·r)² y el presupuesto, 250.000 €.
* O8.2: cada complicado ocuparía la cama ⌈D − umbral⌉ días.

## Uso

1. **Parámetros comunes**: probabilidades y distribuciones por fenotipo, ruido ε, fases del brote (días, λ y mezcla),
   coste de oportunidad de las camas y objetivos (con casilla para evaluarlos o no).
2. **Escenarios**: costes, umbral, devolución por resultados, factores de carga y depuración, λ del Pico, camas
   propias y de alquiler, contrato por volumen y coste de implantación; copiar, renombrar y eliminar. La vista previa
   teórica incluye el mecanismo de un paciente (U_T, U_L, U_R, U_ε con deslizadores o 🎲 al azar).
3. **Barra lateral**: pacientes y réplicas del Bloque A, brotes del Bloque B, semilla, escenario base y escenarios a
   simular; guardar y cargar la configuración en JSON o restablecer los valores por defecto.
4. **Resultados, Capacidad UCI, Sensibilidad, Ampliación e Informe**: ver la tabla de objetivos anterior.

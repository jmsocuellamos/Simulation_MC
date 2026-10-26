# Simulador de colas M/M/s (Streamlit)

Aplicación para estudiar sistemas de colas **M/M/s con capacidad infinita**: teoría (distribución límite y
fórmulas de Erlang C), validación por simulación de eventos discretos (Monte Carlo), costes y
dimensionamiento del número de servidores, con comparación de varios escenarios e informe Word.

Basado en «D-190. Sistemas de Colas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS (CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_mms
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `mms.py` | Teoría M/M/s (Erlang C), simulación de eventos discretos, costes, barrido de s, objetivos y textos automáticos |
| `graficos.py` | Gráficos (distribución p_n, trayectoria N(t), diagrama de funcionamiento, costes, dimensionamiento, comparativas) |
| `informe.py` | Generación del informe `.docx` |

## Uso

1. **Parámetros comunes**: unidad de tiempo, coste de espera c_q, coste de servicio por defecto c_s, s máximo del
   dimensionamiento y objetivos de servicio (ρ máxima, Wq máxima, P(espera) máxima, Lq máxima; cada uno se puede
   desactivar).
2. **Escenarios**: cada escenario tiene λ, μ, s, una descripción y, opcionalmente, su propio coste de servicio. Se
   pueden copiar, renombrar y eliminar. La pestaña muestra al momento la vista previa teórica (estabilidad, KPIs,
   coste y distribución p_n).
3. **Barra lateral**: duración de cada réplica (en la unidad de tiempo del modelo), réplicas, semilla y escenarios a
   analizar; guardar y cargar la configuración en JSON. De cada réplica se
   descarta el primer 5 % como calentamiento.
4. **Resultados**:
   - síntesis automática y recomendación;
   - comparación de escenarios, objetivos de servicio y validación de la simulación;
   - detalle por escenario, separando siempre lo **teórico** (fórmulas) de lo **simulado** (Monte Carlo):
     resultados teóricos, resultados simulados (KPIs con IC95 %, trayectoria N(t), diagrama de funcionamiento de los primeros 60
     clientes), probabilidad de n clientes en el sistema (P(N = n), P(N ≤ n) y P(N > n), teórica y simulada, con
     calculadora de P(N = k), P(N ≤ k), P(N ≥ k) y P(N > k)) y costes (desglose CTS, CTSn, CTQ y barrido de s);
   - descarga de las réplicas, los barridos y las tablas de probabilidades en Excel.
5. **Informe**: un `.docx` con todas las tablas y gráficas de los escenarios y sus comparativas.

## Modelo

* P0 = [Σ_{n<s} aⁿ/n! + aˢ/(s!(1−ρ))]⁻¹, con a = λ/μ y ρ = a/s (régimen estacionario solo si ρ < 1).
* P(espera) = C(s, a) = aˢ/(s!(1−ρ))·P0 ; Lq = C(s, a)·ρ/(1−ρ) ; L = Lq + a ; W = L/λ ; Wq = Lq/λ.
* Costes por unidad de tiempo: CTS = (c_q + c_s)·(L − Lq) ; CTSn = c_s·(s − (L − Lq)) ; CTQ = λ·c_q·Wq ; CT = CTS + CTSn + CTQ.

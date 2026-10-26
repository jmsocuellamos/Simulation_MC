# Simulador de colas M/M/s/K (Streamlit)

Aplicación para estudiar sistemas de colas **M/M/s/K con capacidad finita K** (clientes en servicio + en cola):
teoría (distribución límite, probabilidad de bloqueo y tasa efectiva de entrada), validación por simulación de
eventos discretos con rechazo de clientes (Monte Carlo), costes y dimensionamiento del número de servidores, con
comparación de varios escenarios e informe Word.

Basado en «D-190. Sistemas de Colas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS (CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_mmsk
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `mmsk.py` | Teoría M/M/s/K, simulación de eventos discretos con bloqueo, costes, barrido de s, objetivos y textos automáticos |
| `graficos.py` | Gráficos (distribución p_n, trayectoria N(t), diagrama de funcionamiento con rechazos, costes, dimensionamiento, comparativas) |
| `informe.py` | Generación del informe `.docx` |

## Uso

1. **Parámetros comunes**: unidad de tiempo, coste de espera c_q, coste por servidor por defecto c_s, s máximo del
   dimensionamiento y objetivos de servicio (ρ̄ máxima, Wq máxima, P(bloqueo) máxima, Lq máxima; cada uno se puede
   desactivar).
2. **Escenarios**: cada escenario tiene λ, μ, s, K (≥ s), una descripción y, opcionalmente, su propio coste por
   servidor. Se pueden copiar, renombrar y eliminar. La pestaña muestra al momento la vista previa teórica (régimen,
   bloqueo, KPIs, coste y distribución p_n con su tabla).
3. **Barra lateral**: duración de cada réplica (en la unidad de tiempo del modelo), réplicas, semilla y escenarios a
   analizar; guardar y cargar la configuración en JSON. De cada réplica se descarta el primer 5 % como calentamiento.
4. **Resultados**:
   - síntesis automática y recomendación;
   - comparación de escenarios, objetivos de servicio y validación de la simulación;
   - detalle por escenario, separando siempre lo **teórico** (📐 fórmulas) de lo **simulado** (🎲 Monte Carlo):
     resultados teóricos, resultados simulados (KPIs con IC95 %, trayectoria N(t) con las líneas s y K, diagrama de
     funcionamiento de las primeras 60 llegadas con los rechazos marcados), probabilidad de n clientes en el sistema
     (n = 0 … K, teórica y simulada, con calculadora) y costes (desglose CTS, CTSn, CTQ y barrido de s con K fijo);
   - descarga de las réplicas, los barridos y las tablas de probabilidades en Excel.
5. **Informe**: un `.docx` con todas las tablas y gráficas de los escenarios y sus comparativas.

## Modelo

* a = λ/μ, ρ = a/s (carga nominal). Al ser finito, el sistema siempre tiene régimen estacionario; con ρ ≥ 1 está
  *saturado* y la capacidad K contiene la cola a costa de rechazar clientes.
* P0 = [Σ_{n<s} aⁿ/n! + aˢ/s!·(1 − ρ^{K−s+1})/(1 − ρ)]⁻¹ (si ρ = 1, el último factor es K − s + 1).
* p_n = P0·aⁿ/n! (n < s) ; p_n = P0·aⁿ/(s!·s^{n−s}) (s ≤ n ≤ K) ; P(bloqueo) = p_K.
* L = Σ n·p_n ; λ̄ = λ(1 − p_K) ; Lq = L − λ̄/μ ; W = L/λ̄ ; Wq = W − 1/μ ; ρ̄ = λ̄/(sμ).
* Costes por unidad de tiempo: CTS = (c_q + c_s)·(L − Lq) ; CTSn = c_s·(s − (L − Lq)) ; CTQ = λ·c_q·Wq (λ nominal) ;
  CT = CTS + CTSn + CTQ. El coste de los clientes rechazados no se incluye.

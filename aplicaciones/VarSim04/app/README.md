# Beneficio esperado por mixtura (Streamlit)

Aplicación para estudiar el beneficio por cliente de una tienda de hornos: cada cliente compra con probabilidad
Pr(compra) y, si compra, elige un modelo con unas probabilidades condicionadas. El beneficio es una **mixtura** que se
simula en dos pasos (indicador de compra y transformada inversa para el modelo). Incluye la teoría exacta, la
simulación Monte Carlo con réplicas, la comparación de estrategias comerciales (escenarios) con objetivos, la
sensibilidad del estimador frente a nsim y un informe Word.

Basado en el Ejemplo 4 de «B-70. Distribuciones derivadas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS
(CC BY-SA 4.0).

## Instalación y ejecución

```bash
cd app_mixtura
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

| Fichero | Contenido |
| --- | --- |
| `app.py` | Interfaz Streamlit |
| `mixtura.py` | Teoría de la mixtura, simulación en dos pasos, objetivos, sensibilidad y textos automáticos |
| `graficos.py` | Gráficos (mecanismo en dos pasos, desenlaces, descomposición de E(B), clientes simulados, réplicas, comparativas, sensibilidad) |
| `informe.py` | Informe `.docx` (las tablas ajustan el ancho de sus columnas para caber siempre en la página) |

## Uso

1. **Parámetros comunes**: clientes que entran por día y objetivos (beneficio neto mínimo por cliente, proporción
   mínima que compra, beneficio mínimo por comprador y compras de gama alta mínimas; cada uno se puede desactivar).
2. **Estrategias**: cada estrategia tiene Pr(compra), un coste comercial por cliente, una tabla de modelos (nombre,
   ganancia y probabilidad condicionada; hasta 8) y una descripción. Se pueden copiar, renombrar y eliminar. La vista
   previa teórica incluye el mecanismo de un cliente en dos pasos (U₁ y U₂ con deslizadores o 🎲 al azar).
3. **Barra lateral**: clientes por réplica (nsim), réplicas, semilla y estrategias a simular; guardar y cargar la
   configuración en JSON. Todas las estrategias usan las mismas semillas (números aleatorios comunes).
4. **Resultados**: síntesis y recomendación; comparación y objetivos; detalle por estrategia separando lo
   **teórico** (📐) de lo **simulado** (🎲): KPIs, réplicas, primeros clientes simulados, desenlaces de la mixtura,
   modelos entre compradores y descomposición de E(B); descarga en Excel.
5. **Sensibilidad**: barrido logarítmico de nsim para el beneficio medio de todos los clientes, el de los compradores
   o la proporción que compra, con la σ entre réplicas frente a la σ teórica del estimador.
6. **Informe**: un `.docx` con todas las tablas y gráficas, incluida (opcionalmente) la sensibilidad.

## Modelo

* T ~ Bernoulli(Pr(compra)) ; si T = 1, modelo i con probabilidad pᵢ (transformada inversa con U₂) y B = bᵢ ; si T = 0, B = 0.
* E(B | compra) = Σ pᵢ·bᵢ ; E(B) = Pr(compra)·E(B | compra) ; Var(B) = Pr(compra)·Σ pᵢ·bᵢ² − E(B)².
* Beneficio neto por cliente = E(B) − coste comercial ; diario = clientes por día × beneficio neto.

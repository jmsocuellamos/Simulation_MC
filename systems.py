import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gamma

def gof_distr(data):
    """
    Evalúa la bondad de ajuste de múltiples distribuciones utilizando
    el Test de Kolmogorov-Smirnov y estimación de parámetros por
    el MÉTODO DE LOS MOMENTOS (MoM).
    
    Distribuciones: Uniforme, Exponencial, Normal, Gamma, Erlang, Triangular, Weibull.
    """
    
    # 1. Preparación de datos y estadísticos básicos
    x = np.array(data)
    x = x[~np.isnan(x)] # Eliminar NaNs si existen
    n = len(x)
    
    # Nivel de significancia
    alpha = 0.05
    
    # Momentos Muestrales
    mu = np.mean(x)
    var = np.var(x, ddof=1) # Varianza muestral (n-1)
    std = np.std(x, ddof=1) # Desviación estándar
    x_min = np.min(x)
    x_max = np.max(x)
    
    results = []

    # ==============================================================================
    # 1. Distribución Uniforme
    # MoM: Rango = sqrt(12 * var). Centrada en la media.
    # ==============================================================================
    range_uni = np.sqrt(12 * var)
    uni_a = mu - (range_uni / 2)
    uni_scale = range_uni # scale = b - a
    
    d_uni, p_uni = stats.kstest(x, 'uniform', args=(uni_a, uni_scale))
    
    results.append({
        'Distribución': 'Uniforme',
        'Parámetros': f'Min={uni_a:.2f}, Range={uni_scale:.2f}',
        'KS Stat': d_uni,
        'P-Value': p_uni
    })

    # ==============================================================================
    # 2. Distribución Exponencial
    # MoM: Scale = Media. (Asumiendo loc=0, típica en tiempos de espera)
    # ==============================================================================
    exp_scale = mu
    d_exp, p_exp = stats.kstest(x, 'expon', args=(0, exp_scale))
    
    results.append({
        'Distribución': 'Exponencial',
        'Parámetros': f'Scale={exp_scale:.2f}',
        'KS Stat': d_exp,
        'P-Value': p_exp
    })

    # ==============================================================================
    # 3. Distribución Normal
    # MoM: Media = mu, Scale = std
    # ==============================================================================
    d_norm, p_norm = stats.kstest(x, 'norm', args=(mu, std))
    
    results.append({
        'Distribución': 'Normal',
        'Parámetros': f'Mu={mu:.2f}, Std={std:.2f}',
        'KS Stat': d_norm,
        'P-Value': p_norm
    })

    # ==============================================================================
    # 4. Distribución Gamma
    # MoM: alpha = mu^2 / var, scale = var / mu
    # ==============================================================================
    gam_scale = var / mu
    gam_a = (mu ** 2) / var
    
    d_gam, p_gam = stats.kstest(x, 'gamma', args=(gam_a, 0, gam_scale))
    
    results.append({
        'Distribución': 'Gamma',
        'Parámetros': f'Alpha={gam_a:.2f}, Beta={gam_scale:.2f}',
        'KS Stat': d_gam,
        'P-Value': p_gam
    })

    # ==============================================================================
    # 5. Distribución Erlang
    # MoM: Igual que Gamma pero k debe ser entero.
    # Ajustamos k redondeando y recalculamos scale para mantener la media.
    # ==============================================================================
    erl_k = max(1, round((mu ** 2) / var))
    erl_scale = mu / erl_k
    
    # Scipy no tiene 'erlang' explícita, usamos gamma con shape entero
    d_erl, p_erl = stats.kstest(x, 'gamma', args=(erl_k, 0, erl_scale))
    
    results.append({
        'Distribución': 'Erlang',
        'Parámetros': f'k={int(erl_k)}, Beta={erl_scale:.2f}',
        'KS Stat': d_erl,
        'P-Value': p_erl
    })

    # ==============================================================================
    # 6. Distribución Triangular
    # MoM Heurístico: Usamos min y max empíricos.
    # Despejamos la moda (c) usando la fórmula de la media: mu = (a+b+c)/3
    # ==============================================================================
    tri_loc = x_min # a
    tri_scale = x_max - x_min # b - a
    
    # Estimación de la moda (c real)
    mode_est = 3 * mu - x_min - x_max
    
    # Restricción: la moda debe estar dentro del rango [min, max]
    mode_est = max(x_min, min(x_max, mode_est))
    
    # Parámetro c para scipy (proporción 0-1)
    if tri_scale > 0:
        tri_c = (mode_est - tri_loc) / tri_scale
    else:
        tri_c = 0.5 # Caso degenerado (varianza 0)

    d_tri, p_tri = stats.kstest(x, 'triang', args=(tri_c, tri_loc, tri_scale))
    
    results.append({
        'Distribución': 'Triangular',
        'Parámetros': f'Min={tri_loc:.2f}, Mode={mode_est:.2f}, Max={x_max:.2f}',
        'KS Stat': d_tri,
        'P-Value': p_tri
    })

    # ==============================================================================
    # 7. Distribución Weibull
    # MoM Numérico: No hay solución cerrada para k (shape).
    # Ecuación a resolver: CV^2 = (std/mu)^2 = [Gamma(1+2/k) / Gamma(1+1/k)^2] - 1
    # ==============================================================================
    cv_sq = (std / mu) ** 2
    
    def weibull_eq(k):
        # Función objetivo para encontrar k
        if k <= 0: return 100.0
        return (gamma(1 + 2/k) / (gamma(1 + 1/k)**2)) - 1 - cv_sq

    # Resolver numéricamente para k (shape)
    try:
        wei_k = optimize.fsolve(weibull_eq, 1.0)[0] # Semilla inicial = 1 (Exponencial)
    except:
        wei_k = 1.0
        
    # Una vez tenemos k, obtenemos lambda (scale)
    wei_scale = mu / gamma(1 + 1/wei_k)
    
    d_wei, p_wei = stats.kstest(x, 'weibull_min', args=(wei_k, 0, wei_scale))
    
    results.append({
        'Distribución': 'Weibull',
        'Parámetros': f'Shape={wei_k:.2f}, Scale={wei_scale:.2f}',
        'KS Stat': d_wei,
        'P-Value': p_wei
    })

    # ==============================================================================
    # Consolidación de Resultados
    # ==============================================================================
    df_results = pd.DataFrame(results)
    
    # Añadir columna de decisión
    df_results['¿Ajuste Válido?'] = df_results['P-Value'].apply(
        lambda p: 'Sí' if p > alpha else 'No (Rechazado)'
    )
    
    # Ordenar por P-Value descendente (Mejor ajuste arriba)
    df_results = df_results.sort_values(by='P-Value', ascending=False).reset_index(drop=True)
    
    return df_results


def kpis_temporales(df, num_servidores, t_window):
    """
    Calcula KPIs temporales con redondeo a 3 decimales.
    
    Parámetros de entrada:
      - df: dataframe con columnas `Pieza_ID`, `Station`, `Tin`, `Tinit_service`, `Tout`, `Nsystem`, `Nqueue`, 
      `Estado_QC`, `Tservice`, `Tstation`, `Tqueue`, `Ttras`.
      - num_servidores: diccionario con el número de servidores por estación.
      - t_window: intervalo de tiempo (tin, tfin) donde se evalua el sistema.

    Devuelve:
      Dataframe con los KPIs calculados por estación y para el sistema global. Los KPis incluyen el valor medio, 
      desviación típica, e intervalo con los percentiles del 5% y del 95%
    """
    
    tin, tfin = t_window
    resultados = []
    
    # ==============================================================================
    # 1. ANÁLISIS POR ESTACIÓN
    # ==============================================================================
    
    # Filtramos salidas dentro de la ventana
    df_period_station = df[(df['Tout'] >= tin) & (df['Tout'] <= tfin)].copy()
    estaciones = sorted(df['Station'].unique())
    
    for estacion in estaciones:
        data_st = df_period_station[df_period_station['Station'] == estacion]
        
        if data_st.empty:
            continue
            
        # --- PERMANENCIA (Tstation) ---
        ts_values = data_st['Tstation']
        mean_ts = round(ts_values.mean(), 3)
        std_ts = round(ts_values.std(), 3)
        # Intervalo 5% - 95%
        ts_p05 = round(ts_values.quantile(0.05), 3)
        ts_p95 = round(ts_values.quantile(0.95), 3)
        ts_interval = f"[{ts_p05} - {ts_p95}]"
        
        # --- COLA (Tqueue) ---
        wq_values = data_st['Tqueue']
        mean_wq = round(wq_values.mean(), 3)
        std_wq = round(wq_values.std(), 3)
        # Intervalo 5% - 95%
        wq_p05 = round(wq_values.quantile(0.05), 3)
        wq_p95 = round(wq_values.quantile(0.95), 3)
        wq_interval = f"[{wq_p05} - {wq_p95}]"
        
        # --- EFICIENCIA (Ratio %) ---
        total_wq = wq_values.sum()
        total_ts = ts_values.sum()
        
        # Multiplicamos por 100 para porcentaje
        ratio_eff = (total_wq / total_ts) if total_ts > 0 else 0
        ratio_eff = round(ratio_eff, 3)
        
        resultados.append({
            'Nivel': estacion,
            'Tipo': 'Estación',
            'Media_Permanencia': mean_ts,
            'Std_Permanencia': std_ts,
            'Intervalo_Permanencia (5-95%)': ts_interval,
            'Media_Cola': mean_wq,
            'Std_Cola': std_wq,
            'Intervalo_Cola (5-95%)': wq_interval,
            'Eficiencia_Flujo_Cola': ratio_eff
        })

    # ==============================================================================
    # 2. ANÁLISIS DEL SISTEMA GLOBAL
    # ==============================================================================
    
    estacion_final = '06_Control_QC'
    
    # Identificar piezas terminadas en la ventana
    piezas_terminadas_ids = df[
        (df['Station'] == estacion_final) & 
        (df['Tout'] >= tin) & 
        (df['Tout'] <= tfin)
    ]['Pieza_ID'].unique()
    
    if len(piezas_terminadas_ids) > 0:
        df_system_full = df[df['Pieza_ID'].isin(piezas_terminadas_ids)].copy()
        
        # Agregamos por Pieza para tener los totales del sistema
        grouped = df_system_full.groupby('Pieza_ID').agg(
            Sum_Tstation=('Tstation', 'sum'),
            Sum_Ttras=('Ttras', 'sum'),
            Sum_Tqueue=('Tqueue', 'sum')
        )
        
        # Cálculo de tiempos totales por pieza
        sys_times = grouped['Sum_Tstation'] + grouped['Sum_Ttras'] # T_sys
        queue_times = grouped['Sum_Tqueue'] # W_q_sys
        
        # --- PERMANENCIA SISTEMA ---
        sys_mean_ts = round(sys_times.mean(), 3)
        sys_std_ts = round(sys_times.std(), 3)
        # Intervalo
        sys_p05 = round(sys_times.quantile(0.05), 3)
        sys_p95 = round(sys_times.quantile(0.95), 3)
        sys_interval = f"[{sys_p05} - {sys_p95}]"
        
        # --- COLA SISTEMA ---
        sys_mean_wq = round(queue_times.mean(), 3)
        sys_std_wq = round(queue_times.std(), 3)
        # Intervalo
        sys_wq_p05 = round(queue_times.quantile(0.05), 3)
        sys_wq_p95 = round(queue_times.quantile(0.95), 3)
        sys_wq_interval = f"[{sys_wq_p05} - {sys_wq_p95}]"
        
        # --- EFICIENCIA SISTEMA (%) ---
        total_sys_wq = queue_times.sum()
        total_sys_cycle = sys_times.sum()
        
        # Multiplicamos por 100 para porcentaje
        sys_ratio_eff = (total_sys_wq / total_sys_cycle) if total_sys_cycle > 0 else 0
        sys_ratio_eff = round(sys_ratio_eff, 3)
        
        resultados.append({
            'Nivel': 'Global_Sistema',
            'Tipo': 'Sistema',
            'Media_Permanencia': sys_mean_ts,
            'Std_Permanencia': sys_std_ts,
            'Intervalo_Permanencia (5-95%)': sys_interval,
            'Media_Cola': sys_mean_wq,
            'Std_Cola': sys_std_wq,
            'Intervalo_Cola (5-95%)': sys_wq_interval,
            'Eficiencia_Flujo_Cola': sys_ratio_eff
        })
    else:
        # Caso sin datos en la ventana
        resultados.append({
            'Nivel': 'Global_Sistema', 'Tipo': 'Sistema',
            'Media_Permanencia': 0.0, 'Std_Permanencia': 0.0, 
            'Intervalo_Permanencia (5-95%)': '[0 - 0]',
            'Media_Cola': 0.0, 'Std_Cola': 0.0, 
            'Intervalo_Cola (5-95%)': '[0 - 0]',
            'Eficiencia_Flujo_Cola': 0.0
        })

    # ==============================================================================
    # 3. OUTPUT
    # ==============================================================================
    
    df_kpis = pd.DataFrame(resultados)
    
    # Orden de columnas
    cols = [
        'Nivel', 'Tipo', 
        'Media_Permanencia', 'Std_Permanencia', 'Intervalo_Permanencia (5-95%)',
        'Media_Cola', 'Std_Cola', 'Intervalo_Cola (5-95%)',
        'Eficiencia_Flujo_Cola'
    ]
    df_kpis = df_kpis[cols]
    
    return df_kpis


import pandas as pd
import numpy as np

def kpis_eficiencia(df, num_servidores, t_window):
    """
    Calcula KPIs de eficiencia (Throughput y Utilización) por estación
    y añade una fila de resumen GLOBAL del sistema.
    """
    
    tin, tfin = t_window
    
    # PASO 1: Horizonte Temporal
    T_obs = tfin - tin
    
    if T_obs <= 0:
        raise ValueError("El intervalo de tiempo (tfin - tin) debe ser mayor a 0")

    # Filtramos datos dentro de la ventana
    df_window = df[(df['Tout'] >= tin) & (df['Tout'] <= tfin)].copy()
    
    estaciones = sorted(df['Station'].unique())
    resultados_temp = []
    
    # Acumuladores para el cálculo global
    global_work_sum = 0
    global_capacity_sum = 0
    
    # Recorremos cada estación
    for estacion in estaciones:
        data_st = df_window[df_window['Station'] == estacion]
        k_servidores = num_servidores.get(estacion, 1)
        
        # --- PASO 2 (Estación): Throughput ---
        N_out = len(data_st)
        th_val = N_out / T_obs
        
        # --- PASO 3 (Estación): Utilización ---
        work_s = data_st['Tservice'].sum() # Trabajo total realizado
        cap_s = T_obs * k_servidores        # Capacidad total disponible
        
        # Acumulamos para el global
        global_work_sum += work_s
        global_capacity_sum += cap_s
        
        if cap_s > 0:
            rho_s = work_s / cap_s
        else:
            rho_s = 0.0
            
        resultados_temp.append({
            'Estación': estacion,
            'N_out': N_out,
            'Throughput': th_val,
            'Utilización': rho_s,
            'Tipo': 'Estación' # Marcador para distinguir
        })

    # --- CÁLCULOS GLOBALES ---
    # 1. Throughput Global: Solo cuenta lo que sale de la última estación (06_Control_QC)
    estacion_final = '06_Control_QC'
    data_final_st = df_window[df_window['Station'] == estacion_final]
    N_out_sys = len(data_final_st)
    th_sys = N_out_sys / T_obs
    
    # 2. Utilización Global (Ponderada): Total Trabajo / Total Capacidad
    if global_capacity_sum > 0:
        rho_sys = global_work_sum / global_capacity_sum
    else:
        rho_sys = 0.0
        
    # Añadimos la fila global a la lista temporal (sin analizar cuello de botella aún)
    resultados_temp.append({
        'Estación': 'Global_Sistema',
        'N_out': N_out_sys,
        'Throughput': th_sys,
        'Utilización': rho_sys,
        'Tipo': 'Sistema'
    })

    # --- PASO 4: Análisis y Formato Final ---
    
    df_res = pd.DataFrame(resultados_temp)
    
    if not df_res.empty:
        # Identificar Bottleneck (SOLO entre estaciones, ignoramos la fila Global)
        subset_estaciones = df_res[df_res['Tipo'] == 'Estación']
        if not subset_estaciones.empty:
            max_rho = subset_estaciones['Utilización'].max()
        else:
            max_rho = 0

        final_rows = []
        for index, row in df_res.iterrows():
            rho = row['Utilización']
            th = row['Throughput']
            estacion_nombre = row['Estación']
            tipo = row['Tipo']
            
            # Lógica de Bottleneck y Estabilidad
            if tipo == 'Estación':
                es_bottleneck = (rho == max_rho) and (max_rho > 0)
                bottleneck_str = 'SÍ' if es_bottleneck else 'No'
                
                if rho >= 1.0:
                    estado = "Inestable (Saturado)"
                elif rho >= 0.85:
                    estado = "Alta Congestión"
                else:
                    estado = "Estable"
            else:
                # Lógica para la fila Global
                bottleneck_str = '-'
                estado = "Promedio Sistema"

            final_rows.append({
                'Estación': estacion_nombre,
                'Unidades_Procesadas': int(row['N_out']),
                'Throughput (Unid/tiempo)': round(th, 3),
                'Utilización (Rho)': round(rho, 3),
                'Es_Cuello_Botella': bottleneck_str,
                'Estado_Estabilidad': estado
            })
            
        df_final = pd.DataFrame(final_rows)
        
    else:
        df_final = pd.DataFrame(columns=[
            'Estación', 'Unidades_Procesadas', 'Throughput (Unid/tiempo)', 
            'Utilización (Rho)', 'Es_Cuello_Botella', 'Estado_Estabilidad'
        ])

    return df_final

import pandas as pd
import numpy as np

def kpis_inventario(df, num_servidores, t_window):
    """
    Calcula el WIP (Work In Process) aplicando la Ley de Little y lo compara
    con el WIP observado (Time-Weighted) para validar la estabilidad del sistema.
    """

    tin, tfin = t_window

    # ==============================================================================
    # PASO 1: Definición de Variables Previas (Horizonte Temporal)
    # ==============================================================================
    T_obs = tfin - tin

    if T_obs <= 0:
        raise ValueError("El intervalo de tiempo debe ser positivo.")

    resultados = []

    # ==============================================================================
    # PASO 2: Evaluación por Estación Individual (Nivel Micro)
    # ==============================================================================

    # Filtramos las piezas que FINALIZARON su paso por la estación en la ventana
    df_window = df[(df['Tout'] >= tin) & (df['Tout'] <= tfin)].copy()
    estaciones = sorted(df['Station'].unique())

    for estacion in estaciones:
        data_st = df_window[df_window['Station'] == estacion]

        # Si no hay datos, rellenamos con 0
        if data_st.empty:
            resultados.append({
                'Nivel': estacion,
                'WIP_Teorico (Little)': 0.0,
                'WIP_Observado (Time-Weighted)': 0.0,
                'Discrepancia': 0.0,
                'Estado_QA': 'Sin Datos'
            })
            continue

        # 1. Cálculo de Tasa Local (Lambda_s)
        # N_s: Cantidad de salidas en el periodo
        N_s = len(data_st)
        lambda_s = N_s / T_obs

        # 2. Obtención del Tiempo en Estación (E[W_s])
        # W_s incluye tiempo en cola + servicio (columna 'Tstation')
        mean_w_s = data_st['Tstation'].mean()

        # 3. Aplicación de la Ley (WIP Teórico)
        # L = Lambda * W
        wip_teorico = lambda_s * mean_w_s

        # 4. Cálculo del WIP Observado (Time-Weighted)
        # Matemáticamente equivale a la Suma de todos los tiempos de estancia / T_obs
        # Esto representa la integral del inventario en el tiempo dividida por el periodo
        total_time_spent = data_st['Tstation'].sum()
        wip_observado = total_time_spent / T_obs

        # Validación QA (Diferencia absoluta)
        diff = abs(wip_teorico - wip_observado)

        # Estado de consistencia (Tolerancia baja por precisión numérica)
        qa_status = "Consistente" if diff < 0.001 else "Discrepancia Numérica"

        resultados.append({
            'Nivel': estacion,
            'WIP_Teorico (Little)': round(wip_teorico, 3),
            'WIP_Observado (Time-Weighted)': round(wip_observado, 3),
            'Discrepancia': round(diff, 5), # Más decimales para ver errores finos
            'Estado_QA': qa_status
        })

    # ==============================================================================
    # PASO 3: Evaluación del Sistema Global (Nivel Macro)
    # ==============================================================================

    estacion_final = '06_Control_QC'

    # Identificamos piezas que terminaron todo el proceso en la ventana
    data_final_st = df_window[df_window['Station'] == estacion_final]
    piezas_terminadas_ids = data_final_st['Pieza_ID'].unique()

    if len(piezas_terminadas_ids) > 0:
        # 1. Cálculo de Tasa Global (Lambda_sys)
        N_sys = len(piezas_terminadas_ids)
        lambda_sys = N_sys / T_obs

        # Recuperamos la historia completa de estas piezas para calcular su Tiempo de Ciclo
        df_system_full = df[df['Pieza_ID'].isin(piezas_terminadas_ids)].copy()

        grouped = df_system_full.groupby('Pieza_ID').agg(
            Sum_Tstation=('Tstation', 'sum'),
            Sum_Ttras=('Ttras', 'sum')
        )

        # Tiempo de ciclo (T_sys) = Suma estancias + Suma traslados
        grouped['T_sys_total'] = grouped['Sum_Tstation'] + grouped['Sum_Ttras']

        # 2. Obtención del Tiempo de Ciclo Medio (E[W_sys])
        mean_w_sys = grouped['T_sys_total'].mean()

        # 3. Aplicación de la Ley (WIP Total Teórico)
        wip_sys_teorico = lambda_sys * mean_w_sys

        # 4. WIP Observado Global
        # Suma de todos los tiempos invertidos por las piezas que salieron / T_obs
        total_sys_time_spent = grouped['T_sys_total'].sum()
        wip_sys_observado = total_sys_time_spent / T_obs

        diff_sys = abs(wip_sys_teorico - wip_sys_observado)
        qa_status_sys = "Consistente (Estable)" if diff_sys < 0.01 else "Posible Inestabilidad"

        resultados.append({
            'Nivel': 'Global_Sistema',
            'WIP_Teorico (Little)': round(wip_sys_teorico, 3),
            'WIP_Observado (Time-Weighted)': round(wip_sys_observado, 3),
            'Discrepancia': round(diff_sys, 5),
            'Estado_QA': qa_status_sys
        })
    else:
        # Caso sin salidas globales
        resultados.append({
            'Nivel': 'Global_Sistema', 
            'WIP_Teorico (Little)': 0, 'WIP_Observado (Time-Weighted)': 0,
            'Discrepancia': 0, 'Estado_QA': 'Sin Salidas'
        })

    # ==============================================================================
    # PASO 4: Formato Final
    # ==============================================================================

    df_little = pd.DataFrame(resultados)

    # Reordenamos columnas
    cols = [
        'Nivel', 
        'WIP_Teorico (Little)', 'WIP_Observado (Time-Weighted)',
        'Discrepancia', 'Estado_QA'
    ]
    df_little = df_little[cols]

    return df_little

def kpis_calidad(df, num_servidores, t_window, costes_input):
    """
    Calcula KPIs de calidad y financieros utilizando una estructura de costes detallada.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataframe con las columnas del proceso.
    num_servidores : dict
        (Mantenido por consistencia de firma, no afecta cálculo financiero directo).
    t_window : tuple (tin, tfin)
        Ventana de observación.
    costes_input : dict
        Diccionario con claves: 'Materia_Prima', 'Coste_Rechazo' y 
        'Costes_Estacion' (este último mapea cada estación a sus costes).
        
    Retorna:
    --------
    pd.DataFrame
        KPIs de Calidad, Coste Hundido y CPU Real.
    """
    
    tin, tfin = t_window
    T_obs = tfin - tin
    
    # 1. Extracción de Parámetros de Coste
    C_mat = costes_input.get('Materia_Prima', 0.0)
    C_rechazo = costes_input.get('Coste_Rechazo', 0.0)
    dict_estaciones = costes_input.get('Costes_Estacion', {})
    
    # Preparamos mapas de mapeo para aplicar costes rápidamente al dataframe
    # Si una estación no está en el diccionario, asumimos coste 0 para evitar errores
    mapa_coste_hora = {k: v.get('Coste', 0.0) for k, v in dict_estaciones.items()}
    mapa_coste_proc = {k: v.get('Coste_Procesado', 0.0) for k, v in dict_estaciones.items()}
    
    # 2. Identificación de la producción en la ventana (Estación QC)
    qc_station = '06_Control_QC'
    df_qc = df[(df['Station'] == qc_station) & (df['Tout'] >= tin) & (df['Tout'] <= tfin)].copy()
    
    if df_qc.empty:
        return pd.DataFrame(columns=['KPI', 'Valor', 'Unidad'])

    # ==============================================================================
    # PASO 1 y 2: Métricas de Calidad (Yield y Throughput)
    # ==============================================================================
    
    N_total = len(df_qc)
    
    # Identificación de IDs
    ids_nok = df_qc[df_qc['Estado_QC'] == 'Rechazada']['Pieza_ID'].unique()
    ids_ok = df_qc[df_qc['Estado_QC'] != 'Rechazada']['Pieza_ID'].unique()
    
    N_nok = len(ids_nok)
    N_ok = len(ids_ok)
    
    # Yield
    yield_val = N_ok / N_total if N_total > 0 else 0.0
    
    # Throughput
    th_nom = N_total / T_obs
    th_eff = th_nom * yield_val
    
    # ==============================================================================
    # CÁLCULOS FINANCIEROS (Preparación de Datos)
    # ==============================================================================
    
    # Recuperamos el historial completo de TODAS las piezas que salieron en este periodo (OK y NOK)
    # para calcular el coste real acumulado de cada una.
    all_ids = np.concatenate([ids_ok, ids_nok])
    df_history = df[df['Pieza_ID'].isin(all_ids)].copy()
    
    # Mapeamos los costes al dataframe histórico basándonos en la estación
    # Coste Hora asignado a cada fila
    df_history['Rate_Hora'] = df_history['Station'].map(mapa_coste_hora).fillna(0)
    # Coste Fijo asignado a cada fila
    df_history['Rate_Fijo'] = df_history['Station'].map(mapa_coste_proc).fillna(0)
    
    # Calculamos el coste operativo de CADA paso individual
    # Coste_Paso = Coste_Fijo + (Tiempo_Servicio * Coste_Hora)
    df_history['Coste_Paso'] = df_history['Rate_Fijo'] + (df_history['Tservice'] * df_history['Rate_Hora'])
    
    # Agrupamos por pieza para tener el Coste de Procesado Acumulado (CP_i)
    costos_por_pieza = df_history.groupby('Pieza_ID')['Coste_Paso'].sum().reset_index()
    
    # Añadimos el Coste de Materia Prima a todas las piezas
    costos_por_pieza['Coste_Total_Pieza'] = costos_por_pieza['Coste_Paso'] + C_mat
    
    # ==============================================================================
    # PASO 3: Coste Hundido (Sunk Cost)
    # ==============================================================================
    
    sunk_cost_total = 0.0
    
    if N_nok > 0:
        # Filtramos solo los costes de las piezas NOK
        costos_nok = costos_por_pieza[costos_por_pieza['Pieza_ID'].isin(ids_nok)]
        
        # Sumamos el coste de producción acumulado de las malas
        valor_agregado_perdido = costos_nok['Coste_Total_Pieza'].sum()
        
        # Añadimos la gestión de residuos
        coste_gestion_residuos = N_nok * C_rechazo
        
        sunk_cost_total = valor_agregado_perdido + coste_gestion_residuos

    avg_sunk_cost = (sunk_cost_total / N_nok) if N_nok > 0 else 0.0

    # ==============================================================================
    # PASO 4: Coste Unitario Real (CPU)
    # ==============================================================================
    
    # Coste Total del Sistema (C_sys)
    # 1. Suma de costes de producción de TODAS las piezas (OK + NOK)
    total_production_cost = costos_por_pieza['Coste_Total_Pieza'].sum()
    
    # 2. Costes extra de calidad (solo las rechazadas generan coste de residuo)
    total_waste_cost = N_nok * C_rechazo
    
    c_sys = total_production_cost + total_waste_cost
    
    # CPU Real = Todo el dinero gastado / Solo las piezas que puedo vender
    if N_ok > 0:
        cpu_real = c_sys / N_ok
    else:
        cpu_real = c_sys # Infinito/Total pérdida
    
    # ==============================================================================
    # RESULTADOS
    # ==============================================================================
    
    resultados = [
        {'KPI': 'Piezas Totales (Salidas)', 'Valor': N_total, 'Unidad': 'Unidades'},
        {'KPI': 'Piezas aceptadas', 'Valor': N_ok, 'Unidad': 'Unidades'},
        {'KPI': 'Piezas rechazadas', 'Valor': N_nok, 'Unidad': 'Unidades'},
        {'KPI': 'Rendimiento (Yield)', 'Valor': round(yield_val * 100, 2), 'Unidad': '%'},
        {'KPI': 'Throughput Efectivo', 'Valor': round(th_eff, 3), 'Unidad': 'Pieza/ud_tiempo'},
        {'KPI': 'Coste Total', 'Valor': round(sunk_cost_total, 2), 'Unidad': 'Euros'},
        {'KPI': 'Pérdida media por rechazo', 'Valor': round(avg_sunk_cost, 2), 'Unidad': 'Euros/pz'},
        {'KPI': 'Coste Total Operativo', 'Valor': round(c_sys, 2), 'Unidad': 'Euros'},
        {'KPI': 'Coste Unitario', 'Valor': round(cpu_real, 2), 'Unidad': 'Euros/pz'}
    ]
    
    return pd.DataFrame(resultados)

def kpis_financieros(df, num_servidores, t_window, costes_input):
    """
    Calcula el Coste Unitario Total (CT_u) desglosado en Coste Variable y 
    Coste Fijo asignado por capacidad.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataframe de la simulación (se usa para calcular N_prod).
    num_servidores : dict
        Diccionario con la cantidad de servidores por estación (k_i).
    t_window : tuple (tin, tfin)
        Horizonte temporal de la simulación.
    costes_input : dict
        Diccionario de costes con estructura:
        {
            'Materia_Prima': float,
            'Costes_Estacion': {
                'Nombre_Estacion': {'Coste_Hora': float, 'Coste_Procesado': float},
                ...
            }
        }
        
    Retorna:
    --------
    pd.DataFrame
        Desglose de costes unitarios (Variable, Fijo y Total).
    """
    
    tin, tfin = t_window
    
    # PASO 1: Identificación de Parámetros
    # -------------------------------------
    
    # T_total: Tiempo total de simulación
    T_total = tfin - tin
    if T_total <= 0:
        raise ValueError("El tiempo de simulación debe ser mayor a 0")
        
    # Extracción de costes estáticos
    C_mat = costes_input.get('Materia_Prima', 0.0)
    dict_estaciones_costes = costes_input.get('Costes_Estacion', {})
    
    # N_prod: Cantidad total producida EXITOSA (Throughput vendible)
    # Filtramos la última estación (Control_QC) y solo las piezas NO Rechazadas
    qc_station = '06_Control_QC'
    df_prod = df[
        (df['Station'] == qc_station) & 
        (df['Estado_QC'] != 'Rechazada') &
        (df['Tout'] >= tin) & 
        (df['Tout'] <= tfin)
    ]
    N_prod = len(df_prod)
    
    # PASO 2: Cálculo del Coste Variable Unitario (CV_u)
    # --------------------------------------------------
    # Este coste es teórico por pieza: Materia Prima + Suma de "peajes" de cada estación.
    
    sum_proc = 0.0
    for est, costes in dict_estaciones_costes.items():
        # Sumamos el coste fijo de procesado (Coste_Procesado) de todas las estaciones definidas
        sum_proc += costes.get('Coste_Procesado', 0.0)
        
    CV_u = C_mat + sum_proc
    
    # PASO 3: Cálculo del Coste Fijo Asignado (CF_u)
    # ----------------------------------------------
    # Coste de mantener la fábrica abierta (Capacidad) dividido por producción.
    
    coste_total_operativo_capacidad = 0.0
    
    # Iteramos sobre las estaciones presentes en num_servidores (infraestructura real)
    for estacion, k_i in num_servidores.items():
        # Buscamos el coste hora en el diccionario de costes
        info_coste = dict_estaciones_costes.get(estacion, {})
        C_serv_i = info_coste.get('Coste_Hora', 0.0)
        
        # Coste Estación = Coste_Hora * Num_Servidores * Tiempo_Total
        # (Se paga por tener el servidor disponible, trabaje o no)
        coste_estacion_s = C_serv_i * k_i * T_total
        
        coste_total_operativo_capacidad += coste_estacion_s
        
    # Prorrateo Unitario
    if N_prod > 0:
        CF_u = coste_total_operativo_capacidad / N_prod
    else:
        # Si no se produce nada, el coste fijo unitario tiende a infinito 
        # (o es todo el coste operativo)
        CF_u = np.inf 
        
    # PASO 4: Consolidación (Coste Total Unitario)
    # --------------------------------------------
    
    CT_u = CV_u + CF_u
    
    # Generación de la tabla de resultados
    resultados = [
        {
            'Concepto': 'Producción Exitosa (pieza)',
            'Valor': N_prod,
            'Tipo': 'Volumen'
        },
        {
            'Concepto': 'Coste Materia Prima Base',
            'Valor': round(C_mat, 2),
            'Tipo': 'Euros'
        },
        {
            'Concepto': 'Suma Costes Procesado (Agregado)',
            'Valor': round(sum_proc, 2),
            'Tipo': 'Euros'
        },
        {
            'Concepto': 'COSTE VARIABLE UNITARIO',
            'Valor': round(CV_u, 2),
            'Tipo': 'Euros'
        },
        {
            'Concepto': 'Coste Operativo Capacidad Total',
            'Valor': round(coste_total_operativo_capacidad, 2),
            'Tipo': 'Euros'
        },
        {
            'Concepto': 'COSTE FIJO ASIGNADO',
            'Valor': round(CF_u, 2),
            'Tipo': 'Euros'
        },
        {
            'Concepto': 'COSTE TOTAL UNITARIO',
            'Valor': round(CT_u, 2),
            'Tipo': 'Euros'
        }
    ]
    
    return pd.DataFrame(resultados)

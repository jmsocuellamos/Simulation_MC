# -*- coding: utf-8 -*-
"""Gráficos (matplotlib/seaborn). Todas las funciones devuelven una Figure, que la aplicación muestra
con st.pyplot y el informe inserta como imagen."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from motor import TIPOS
from kpis import ETIQUETAS_KPI, tabla_media_mc

sns.set_theme(style="whitegrid")
COLORES_PERFIL = {"Verde": "#2ca02c", "Amarillo": "#e6c200", "Naranja": "#ff7f0e", "Rojo": "#d62728"}


def fig_a_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _paleta(etiquetas):
    pal = sns.color_palette("Set2", max(len(etiquetas), 3))
    return {e: pal[i % len(pal)] for i, e in enumerate(etiquetas)}


def fig_utilizacion(rec, titulo):
    r = rec.sort_values("Utilizacion")
    col = np.select([r["Utilizacion"] >= 1, r["Utilizacion"] >= 0.85], ["#d62728", "#ff7f0e"], "steelblue")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(r) + 1.2))
    ax.barh(r.index, r["Utilizacion"], color=col)
    ax.axvline(0.85, ls="--", c="orange", lw=1); ax.axvline(1, ls="--", c="red", lw=1)
    ax.set_xlabel("Utilización ρ"); ax.set_title(titulo, fontsize=11)
    fig.tight_layout()
    return fig


def fig_utilizacion_comparada(protocolos):
    """Utilización media por recurso y protocolo (barras agrupadas)."""
    tabla = pd.DataFrame({et: tabla_media_mc(m, "recursos")["Utilizacion"] for et, m in protocolos.items()})
    tabla = tabla.loc[tabla.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(tabla) + 1.5))
    n = tabla.shape[1]
    alto = 0.8 / n
    pal = _paleta(list(tabla.columns))
    y = np.arange(len(tabla))
    for j, c in enumerate(tabla.columns):
        ax.barh(y + j * alto - 0.4 + alto / 2, tabla[c].fillna(0), height=alto, label=c, color=pal[c])
    ax.set_yticks(y); ax.set_yticklabels(tabla.index)
    ax.axvline(0.85, ls="--", c="orange", lw=1); ax.axvline(1, ls="--", c="red", lw=1)
    ax.set_xlabel("Utilización media ρ"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Utilización media de recursos por escenario", fontsize=11)
    fig.tight_layout()
    return fig


def fig_comparativo(protocolos, kpis, ncols=3, titulo=None):
    """Boxplots de la distribución Monte Carlo de cada KPI para varios escenarios."""
    kpis = [k for k in kpis if all(k in m["resumen"].columns for m in protocolos.values())]
    largo = pd.concat([m["resumen"][kpis].astype(float).assign(Escenario=et) for et, m in protocolos.items()], ignore_index=True)
    nrows = max(1, math.ceil(len(kpis) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    pal = _paleta(list(protocolos))
    for ax, k in zip(axes.flat, kpis):
        if largo[k].notna().sum() == 0:
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_title(ETIQUETAS_KPI.get(k, k), fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            continue
        sns.boxplot(data=largo, x="Escenario", y=k, ax=ax, hue="Escenario", legend=False, palette=pal)
        ax.set_title(ETIQUETAS_KPI.get(k, k), fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=30 if len(protocolos) > 3 else 0)
    for ax in list(axes.flat)[len(kpis):]:
        ax.axis("off")
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    return fig


def fig_sintesis(ranking, protocolos_orden, ref):
    rk = ranking.reindex(protocolos_orden)
    fig, ax = plt.subplots(1, 3, figsize=(16, 0.45 * len(rk) + 2))
    for a, (k, tit) in zip(ax, [("Resultado neto anual (€)", "Resultado neto anual vs. referencia (€)"),
                                ("SLAs cumplidos", "Nº de SLAs cumplidos (de 6)"),
                                ("Mortalidad Sin_UCI N/R", "Mortalidad Sin_UCI en Naranja/Rojo")]):
        col = ["grey" if i == ref else "steelblue" for i in rk.index]
        a.barh(rk.index, rk[k], color=col); a.set_title(tit, fontsize=10); a.invert_yaxis()
        a.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def fig_tsys_perfil(protocolos):
    """Tiempo en el sistema por perfil (réplica 0 de cada escenario)."""
    datos = []
    for et, m in protocolos.items():
        d = m.get("detalle_replica0")
        if d is not None:
            datos.append(d["pacientes_fin"][["Pat_type", "T_sys"]].assign(Escenario=et))
    if not datos:
        return None
    largo = pd.concat(datos, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.boxplot(data=largo, x="Pat_type", y="T_sys", hue="Escenario", order=TIPOS, showfliers=False,
                ax=ax, palette=_paleta(list(protocolos)))
    ax.set_yscale("log"); ax.set_ylabel("T_sys (h, escala log)"); ax.set_xlabel("")
    ax.set_title("Tiempo en el sistema por perfil (réplica 1 de cada escenario)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_ocupacion_uci(protocolos, umbral=0.9):
    """Ocupación diaria de las camas de UCI (réplica 0 de cada escenario)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    pal = _paleta(list(protocolos))
    for et, m in protocolos.items():
        d = m.get("detalle_replica0")
        if d is not None:
            ax.plot(d["ocupacion_uci"].index, d["ocupacion_uci"].values, label=et, lw=1.3, color=pal[et])
    ax.axhline(umbral, ls="--", c="red", lw=1, label=f"SLA {umbral:.0%}")
    ax.set_xlabel("día"); ax.set_ylabel("ocupación UCI")
    ax.set_title("Ocupación diaria de la UCI (réplica 1 de cada escenario)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_histogramas(mc, kpis, ncols=4):
    kpis = [k for k in kpis if k in mc["resumen"].columns]
    nrows = max(1, math.ceil(len(kpis) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, k in zip(axes.flat, kpis):
        if mc["resumen"][k].notna().sum() == 0:
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_title(ETIQUETAS_KPI.get(k, k), fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            continue
        sns.histplot(mc["resumen"][k].astype(float), ax=ax, color="steelblue", bins=15)
        ax.axvline(mc["resumen"][k].mean(), color="red", lw=1.5)
        ax.set_title(ETIQUETAS_KPI.get(k, k), fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    for ax in list(axes.flat)[len(kpis):]:
        ax.axis("off")
    fig.tight_layout()
    return fig

"""Interfaz web (Streamlit) para configurar y ejecutar el simulador de bases de datos.

En vez de introducir las variables a mano en la interfaz, el usuario descarga
una plantilla Excel, la rellena fuera de la app (una fila por variable en la
hoja 'variables', y opcionalmente la matriz de correlaciones en la hoja
'correlaciones' y las relaciones categorica->continua en la hoja 'efectos'),
y la sube aqui para generar el dataset.

Ejecucion: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from db_simulator.excel_io import build_template_bytes, read_workbook
from db_simulator.simulator import Simulator
from db_simulator.spec import SpecError, spec_from_dict

st.set_page_config(page_title="Simulador de bases de datos", layout="wide")

HIST_COLOR = "#2a78d6"
GRID_COLOR = "#e7e7e5"


def render_preview(raw: dict) -> None:
    st.subheader("Vista previa de lo leido del Excel")

    var_rows = []
    for var in raw["variables"]:
        if var["type"] == "continuous":
            var_rows.append(
                {
                    "nombre": var["name"],
                    "tipo": "continuous",
                    "distribucion": var.get("distribution"),
                    "media": var.get("mean"),
                    "mediana": var.get("median"),
                    "min": var.get("min"),
                    "max": var.get("max"),
                    "sd": var.get("sd", "-"),
                    "categorias": "",
                }
            )
        else:
            categorias = "; ".join(f"{k}:{v}" for k, v in var["categories"].items())
            var_rows.append(
                {
                    "nombre": var["name"],
                    "tipo": "categorical",
                    "distribucion": "",
                    "media": "",
                    "mediana": "",
                    "min": "",
                    "max": "",
                    "sd": "",
                    "categorias": categorias,
                }
            )
    st.markdown(f"**Variables** ({len(var_rows)})")
    st.dataframe(pd.DataFrame(var_rows), hide_index=True, use_container_width=True)

    if "correlation_matrix" in raw:
        names = raw["correlation_matrix"]["variables"]
        matrix = raw["correlation_matrix"]["matrix"]
        st.markdown("**Matriz de correlaciones**")
        st.dataframe(pd.DataFrame(matrix, index=names, columns=names), use_container_width=True)

    effect_rows = []
    for var in raw["variables"]:
        for effect in var.get("effects", []):
            for level in set(effect["mean_delta"]) | set(effect["median_delta"]):
                effect_rows.append(
                    {
                        "variable": var["name"],
                        "segun": effect["by"],
                        "nivel": level,
                        "delta_media": effect["mean_delta"].get(level, 0.0),
                        "delta_mediana": effect["median_delta"].get(level, 0.0),
                    }
                )
    if effect_rows:
        st.markdown("**Relaciones categorica -> continua**")
        st.dataframe(pd.DataFrame(effect_rows), hide_index=True, use_container_width=True)


def render_histograms(df: pd.DataFrame, continuous_names: list[str]) -> None:
    if not continuous_names:
        st.caption("No hay variables continuas que representar.")
        return
    for var_name in continuous_names:
        fig = px.histogram(df, x=var_name, nbins=40, color_discrete_sequence=[HIST_COLOR])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            title=var_name,
            xaxis_title=var_name,
            yaxis_title="Frecuencia",
            bargap=0.05,
            plot_bgcolor="#fcfcfb",
            paper_bgcolor="#fcfcfb",
            showlegend=False,
            margin=dict(t=48, b=40, l=40, r=20),
        )
        fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)


def render_results(df: pd.DataFrame, report) -> None:
    tab_results, tab_hist = st.tabs(["📋 Resultados", "📊 Histogramas"])

    with tab_results:
        st.success(f"Generadas {len(df)} filas x {len(df.columns)} columnas.")
        st.dataframe(df.head(200))
        st.download_button(
            "⬇️ Descargar CSV completo",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="simulacion.csv",
            mime="text/csv",
        )

        st.subheader("Informe: objetivo vs. obtenido")
        if report.continuous:
            rows = []
            for var_name, stats in report.continuous.items():
                rows.append(
                    {
                        "variable": var_name,
                        "media objetivo": stats["target_mean"],
                        "media obtenida": round(stats["achieved_mean"], 4),
                        "mediana objetivo": stats["target_median"],
                        "mediana obtenida": round(stats["achieved_median"], 4),
                        "sd objetivo": stats["target_sd"] if stats["target_sd"] is not None else "-",
                        "sd obtenida": round(stats["achieved_sd"], 4),
                        "min objetivo": stats["target_min"],
                        "min obtenido": round(stats["achieved_min"], 4),
                        "max objetivo": stats["target_max"],
                        "max obtenido": round(stats["achieved_max"], 4),
                    }
                )
            st.markdown("**Variables continuas**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        if report.categorical:
            rows = []
            for var_name, cats in report.categorical.items():
                for cat, (target, achieved) in cats.items():
                    rows.append({"variable": var_name, "categoria": cat, "proporcion objetivo": round(target, 4), "proporcion obtenida": round(achieved, 4)})
            st.markdown("**Variables categoricas**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        if report.effects:
            rows = []
            for var_name, effect_rows in report.effects.items():
                for r in effect_rows:
                    rows.append(
                        {
                            "variable": var_name,
                            "segun": f"{r['by']}={r['level']}",
                            "media ajustada (objetivo)": round(r["media_base_ajustada"], 4),
                            "media obtenida": round(r["media_obtenida"], 4),
                            "n filas": r["n"],
                        }
                    )
            st.markdown("**Relaciones categorica → continua**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        if report.correlation is not None:
            st.markdown("**Correlaciones**")
            col_a, col_b = st.columns(2)
            col_a.caption("Objetivo")
            col_a.dataframe(report.correlation["target"])
            col_b.caption("Obtenida")
            col_b.dataframe(report.correlation["achieved"])

    with tab_hist:
        st.caption("Distribucion de las variables continuas generadas.")
        render_histograms(df, list(report.continuous.keys()))


def main() -> None:
    st.title("Simulador de bases de datos")
    st.write(
        "Rellena la plantilla Excel con tus variables (y opcionalmente la matriz de correlaciones y las "
        "relaciones categorica → continua), subela aqui, y genera un dataset simulado."
    )

    with st.sidebar:
        st.header("Configuracion general")
        n_rows = st.number_input("Numero de filas", min_value=1, value=1000, step=100)
        use_seed = st.checkbox("Fijar semilla aleatoria", value=True)
        seed = st.number_input("Semilla", min_value=0, value=42, step=1) if use_seed else None

    st.subheader("1. Descarga la plantilla")
    st.caption(
        "Incluye una hoja 'variables' (obligatoria), 'correlaciones' y 'efectos' (opcionales), con ejemplos "
        "y una hoja de instrucciones. Borra los ejemplos y rellena tus propias variables antes de subirla."
    )
    st.download_button(
        "⬇️ Descargar plantilla Excel",
        data=build_template_bytes(),
        file_name="plantilla_simulador.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("2. Sube tu Excel ya rellenado")
    uploaded = st.file_uploader("Fichero .xlsx", type=["xlsx"])

    if uploaded is None:
        st.info("Sube un fichero para ver la vista previa y poder generar la simulacion.")
        return

    try:
        raw = read_workbook(uploaded)
    except SpecError as exc:
        st.error(f"Error leyendo el Excel: {exc}")
        return

    render_preview(raw)

    st.subheader("3. Genera la simulacion")
    if st.button("🚀 Generar simulacion", type="primary"):
        raw = dict(raw)
        raw["n_rows"] = int(n_rows)
        raw["seed"] = seed
        try:
            spec = spec_from_dict(raw)
            df, report = Simulator(spec).generate_with_report()
        except SpecError as exc:
            st.error(f"Error en la especificacion: {exc}")
            st.session_state.pop("result_df", None)
            st.session_state.pop("result_report", None)
            return

        st.session_state["result_df"] = df
        st.session_state["result_report"] = report

    if "result_df" in st.session_state and "result_report" in st.session_state:
        render_results(st.session_state["result_df"], st.session_state["result_report"])


if __name__ == "__main__":
    main()

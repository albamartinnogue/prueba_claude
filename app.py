"""Interfaz web (Streamlit) para configurar y ejecutar el simulador de bases de datos.

Permite anadir/quitar variables dinamicamente, editar sus caracteristicas
(nombre, tipo, distribucion, categorias) en un desglose por variable, definir
una matriz de correlaciones entre variables continuas, y generar/descargar
el dataset resultante junto con histogramas de las variables continuas.

Los campos de variables viven dentro de un st.form: los valores tecleados
solo se envian al backend cuando se pulsa alguno de los botones de esa
seccion (Anadir, Eliminar o Aplicar cambios), evitando que se pierdan datos
ya introducidos al interactuar con otra variable.

Ejecucion: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from db_simulator.simulator import Simulator
from db_simulator.spec import CONTINUOUS_DISTRIBUTIONS, SpecError, spec_from_dict

st.set_page_config(page_title="Simulador de bases de datos", layout="wide")

HIST_COLOR = "#2a78d6"
GRID_COLOR = "#e7e7e5"


def init_state() -> None:
    if "variables" not in st.session_state:
        st.session_state.variables = []
    if "next_id" not in st.session_state:
        st.session_state.next_id = 0
    if not st.session_state.variables:
        add_variable()


def add_variable() -> None:
    vid = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.variables.append(vid)
    st.session_state[f"name_{vid}"] = f"variable_{vid + 1}"
    st.session_state[f"type_{vid}"] = "continuous"
    st.session_state[f"distribution_{vid}"] = "normal"
    st.session_state[f"mean_{vid}"] = 0.0
    st.session_state[f"median_{vid}"] = 0.0
    st.session_state[f"min_{vid}"] = 0.0
    st.session_state[f"max_{vid}"] = 1.0
    st.session_state[f"use_sd_{vid}"] = False
    st.session_state[f"sd_{vid}"] = 1.0


def ensure_categories(vid: int) -> None:
    if f"cat_ids_{vid}" not in st.session_state:
        st.session_state[f"cat_ids_{vid}"] = [0, 1]
        st.session_state[f"next_cat_id_{vid}"] = 2
        st.session_state[f"cat_name_{vid}_0"] = "A"
        st.session_state[f"cat_prop_{vid}_0"] = 50.0
        st.session_state[f"cat_name_{vid}_1"] = "B"
        st.session_state[f"cat_prop_{vid}_1"] = 50.0


def add_category(vid: int) -> None:
    cid = st.session_state[f"next_cat_id_{vid}"]
    st.session_state[f"next_cat_id_{vid}"] += 1
    st.session_state[f"cat_ids_{vid}"].append(cid)
    st.session_state[f"cat_name_{vid}_{cid}"] = f"Categoria {cid + 1}"
    st.session_state[f"cat_prop_{vid}_{cid}"] = 10.0


def render_variable(vid: int) -> bool:
    """Renderiza los campos de una variable dentro del formulario. Devuelve True si se pidio eliminarla."""
    name = st.session_state.get(f"name_{vid}", f"variable_{vid + 1}")
    delete_requested = False
    with st.expander(f"🔧 {name}", expanded=True):
        col_name, col_type, col_del = st.columns([3, 2, 1])
        col_name.text_input("Nombre", key=f"name_{vid}")
        col_type.selectbox(
            "Tipo",
            ["continuous", "categorical"],
            key=f"type_{vid}",
            format_func=lambda t: "Continua" if t == "continuous" else "Categorica",
        )
        col_del.write("")
        if col_del.form_submit_button("🗑️ Eliminar", key=f"del_{vid}"):
            delete_requested = True

        if st.session_state[f"type_{vid}"] == "continuous":
            c1, c2 = st.columns([1, 3])
            c1.selectbox("Distribucion", sorted(CONTINUOUS_DISTRIBUTIONS), key=f"distribution_{vid}")

            c3, c4, c5, c6 = st.columns(4)
            c3.number_input("Media", key=f"mean_{vid}", format="%.4f")
            c4.number_input("Mediana", key=f"median_{vid}", format="%.4f")
            c5.number_input("Minimo", key=f"min_{vid}", format="%.4f")
            c6.number_input("Maximo", key=f"max_{vid}", format="%.4f")

            csd1, csd2 = st.columns([1, 3])
            csd1.checkbox("Fijar SD", key=f"use_sd_{vid}", help="Desviacion tipica objetivo (opcional).")
            if st.session_state[f"use_sd_{vid}"]:
                csd2.number_input("Desviacion estandar", key=f"sd_{vid}", min_value=0.0001, format="%.4f")
            if st.session_state[f"distribution_{vid}"] in ("triangular", "uniform") and st.session_state[f"use_sd_{vid}"]:
                st.caption("⚠️ La distribucion elegida no tiene grados de libertad para ajustar tambien la SD; se ignorara.")
        else:
            ensure_categories(vid)
            st.caption("Categorias y proporciones (no hace falta que sumen 100, se normalizan automaticamente).")
            for cid in list(st.session_state[f"cat_ids_{vid}"]):
                cc1, cc2, cc3 = st.columns([3, 2, 1])
                cc1.text_input("Categoria", key=f"cat_name_{vid}_{cid}", label_visibility="collapsed")
                cc2.number_input("Proporcion", key=f"cat_prop_{vid}_{cid}", min_value=0.0, format="%.2f", label_visibility="collapsed")
                if cc3.form_submit_button("🗑️", key=f"cat_del_{vid}_{cid}"):
                    st.session_state[f"cat_ids_{vid}"].remove(cid)
                    st.rerun()
            if st.form_submit_button("+ Anadir categoria", key=f"cat_add_{vid}"):
                add_category(vid)
                st.rerun()
    return delete_requested


def build_variable_dict(vid: int) -> dict:
    var_type = st.session_state[f"type_{vid}"]
    raw = {"name": st.session_state[f"name_{vid}"], "type": var_type}
    if var_type == "continuous":
        raw.update(
            distribution=st.session_state[f"distribution_{vid}"],
            mean=st.session_state[f"mean_{vid}"],
            median=st.session_state[f"median_{vid}"],
            min=st.session_state[f"min_{vid}"],
            max=st.session_state[f"max_{vid}"],
            sd=st.session_state[f"sd_{vid}"] if st.session_state.get(f"use_sd_{vid}") else None,
        )
    else:
        ensure_categories(vid)
        categories = {
            st.session_state[f"cat_name_{vid}_{cid}"]: st.session_state[f"cat_prop_{vid}_{cid}"]
            for cid in st.session_state[f"cat_ids_{vid}"]
        }
        raw["categories"] = categories
    return raw


def render_correlation_section() -> tuple[list[str] | None, list[list[float]] | None]:
    continuous_ids = [vid for vid in st.session_state.variables if st.session_state[f"type_{vid}"] == "continuous"]
    if len(continuous_ids) < 2:
        st.caption("Anade al menos dos variables continuas para poder definir correlaciones entre ellas.")
        return None, None

    selected_ids = st.multiselect(
        "Variables continuas a correlacionar",
        options=continuous_ids,
        format_func=lambda vid: st.session_state.get(f"name_{vid}", f"variable_{vid + 1}"),
        key="corr_selection",
    )
    if len(selected_ids) < 2:
        st.caption("Selecciona al menos dos variables para editar su matriz de correlaciones.")
        return None, None

    names = [st.session_state[f"name_{vid}"] for vid in selected_ids]
    default_matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    editor_key = "corr_matrix_" + "_".join(str(i) for i in selected_ids)
    st.caption("Edita los valores fuera de la diagonal (entre -1 y 1). La matriz se simetriza automaticamente.")
    edited = st.data_editor(default_matrix, key=editor_key)

    matrix = edited.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return names, matrix.tolist()


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
    init_state()
    st.title("Simulador de bases de datos")
    st.write("Define tus variables, sus caracteristicas y (opcionalmente) sus correlaciones, y genera un dataset simulado.")

    with st.sidebar:
        st.header("Configuracion general")
        n_rows = st.number_input("Numero de filas", min_value=1, value=1000, step=100)
        use_seed = st.checkbox("Fijar semilla aleatoria", value=True)
        seed = st.number_input("Semilla", min_value=0, value=42, step=1) if use_seed else None

    st.subheader("Variables")
    st.caption(
        "Todos los cambios de esta seccion (variables, categorias y matriz de correlaciones) se "
        "guardan al pulsar cualquier boton de aqui, incluido 🚀 Generar simulacion."
    )

    variable_to_delete = None
    with st.form("main_form", enter_to_submit=False):
        add_clicked = st.form_submit_button("➕ Anadir variable")

        for vid in list(st.session_state.variables):
            if render_variable(vid):
                variable_to_delete = vid

        apply_clicked = st.form_submit_button("💾 Aplicar cambios", type="primary")

        st.subheader("Matriz de correlaciones (opcional)")
        corr_names, corr_matrix = render_correlation_section()

        st.divider()
        generate_clicked = st.form_submit_button("🚀 Generar simulacion", type="primary")

    if add_clicked:
        add_variable()
        st.rerun()
    if variable_to_delete is not None:
        st.session_state.variables.remove(variable_to_delete)
        st.rerun()
    if apply_clicked:
        st.toast("Cambios aplicados.")

    if generate_clicked:
        raw = {
            "n_rows": int(n_rows),
            "seed": seed,
            "variables": [build_variable_dict(vid) for vid in st.session_state.variables],
        }
        if corr_names is not None:
            raw["correlation_matrix"] = {"variables": corr_names, "matrix": corr_matrix}

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

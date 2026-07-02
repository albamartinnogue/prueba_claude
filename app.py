"""Interfaz web (Streamlit) para configurar y ejecutar el simulador de bases de datos.

Permite anadir/quitar variables dinamicamente, editar sus caracteristicas
(nombre, tipo, distribucion, categorias) en un desglose por variable, definir
una matriz de correlaciones entre variables continuas, y generar/descargar
el dataset resultante.

Ejecucion: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from db_simulator.simulator import Simulator
from db_simulator.spec import CONTINUOUS_DISTRIBUTIONS, SpecError, spec_from_dict

st.set_page_config(page_title="Simulador de bases de datos", layout="wide")


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


def render_variable(vid: int) -> None:
    name = st.session_state.get(f"name_{vid}", f"variable_{vid + 1}")
    with st.expander(f"🔧 {name}", expanded=True):
        col_name, col_type, col_del = st.columns([3, 2, 1])
        col_name.text_input("Nombre", key=f"name_{vid}")
        col_type.selectbox("Tipo", ["continuous", "categorical"], key=f"type_{vid}", format_func=lambda t: "Continua" if t == "continuous" else "Categorica")
        col_del.write("")
        if col_del.button("🗑️ Eliminar", key=f"del_{vid}"):
            st.session_state.variables.remove(vid)
            st.rerun()

        if st.session_state[f"type_{vid}"] == "continuous":
            c1, c2 = st.columns(2)
            c1.selectbox("Distribucion", sorted(CONTINUOUS_DISTRIBUTIONS), key=f"distribution_{vid}")
            c2.empty()
            c3, c4, c5, c6 = st.columns(4)
            c3.number_input("Media", key=f"mean_{vid}", format="%.4f")
            c4.number_input("Mediana", key=f"median_{vid}", format="%.4f")
            c5.number_input("Minimo", key=f"min_{vid}", format="%.4f")
            c6.number_input("Maximo", key=f"max_{vid}", format="%.4f")
        else:
            ensure_categories(vid)
            st.caption("Categorias y proporciones (no hace falta que sumen 100, se normalizan automaticamente).")
            for cid in list(st.session_state[f"cat_ids_{vid}"]):
                cc1, cc2, cc3 = st.columns([3, 2, 1])
                cc1.text_input("Categoria", key=f"cat_name_{vid}_{cid}", label_visibility="collapsed")
                cc2.number_input("Proporcion", key=f"cat_prop_{vid}_{cid}", min_value=0.0, format="%.2f", label_visibility="collapsed")
                if cc3.button("🗑️", key=f"cat_del_{vid}_{cid}"):
                    st.session_state[f"cat_ids_{vid}"].remove(cid)
                    st.rerun()
            if st.button("+ Anadir categoria", key=f"cat_add_{vid}"):
                add_category(vid)
                st.rerun()


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
    if st.button("➕ Anadir variable"):
        add_variable()
        st.rerun()

    for vid in list(st.session_state.variables):
        render_variable(vid)

    st.subheader("Matriz de correlaciones (opcional)")
    corr_names, corr_matrix = render_correlation_section()

    st.divider()
    if st.button("🚀 Generar simulacion", type="primary"):
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
            return

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


if __name__ == "__main__":
    main()

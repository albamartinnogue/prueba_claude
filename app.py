"""Interfaz web (Streamlit) para configurar y ejecutar el simulador de bases de datos.

Pensada para poder definir muchas variables rapidamente:
- Las variables se definen en una tabla editable (anadir/quitar filas, pegar
  varias de golpe) en vez de un formulario por variable.
- Las relaciones se definen aparte, en dos bloques: la matriz de
  correlaciones entre variables continuas, y los "efectos" de una variable
  categorica sobre la media/mediana de una continua (p.ej. que las mujeres
  tengan de media mas ingresos que el resto).

Todo (tabla de variables, relaciones y el boton de generar) vive dentro de un
unico st.form: los valores solo se envian al backend cuando se pulsa alguno
de los botones de esa seccion, para que nunca se pierda lo que ya se habia
introducido en otra fila o relacion.

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

VARIABLE_COLUMNS = ["nombre", "tipo", "distribucion", "media", "mediana", "min", "max", "sd", "categorias"]

DEFAULT_VARIABLES_DF = pd.DataFrame(
    [
        {
            "nombre": "edad",
            "tipo": "continuous",
            "distribucion": "normal",
            "media": 40.0,
            "mediana": 38.0,
            "min": 18.0,
            "max": 85.0,
            "sd": np.nan,
            "categorias": "",
        },
        {
            "nombre": "sexo",
            "tipo": "categorical",
            "distribucion": None,
            "media": np.nan,
            "mediana": np.nan,
            "min": np.nan,
            "max": np.nan,
            "sd": np.nan,
            "categorias": "Hombre:50;Mujer:50",
        },
    ]
)

COLUMN_CONFIG = {
    "nombre": st.column_config.TextColumn("Nombre", required=True, width="medium"),
    "tipo": st.column_config.SelectboxColumn(
        "Tipo", options=["continuous", "categorical"], required=True, width="small"
    ),
    "distribucion": st.column_config.SelectboxColumn(
        "Distribucion", options=sorted(CONTINUOUS_DISTRIBUTIONS), help="Solo si tipo = continuous", width="small"
    ),
    "media": st.column_config.NumberColumn("Media", help="Solo continuous", format="%.2f"),
    "mediana": st.column_config.NumberColumn("Mediana", help="Solo continuous", format="%.2f"),
    "min": st.column_config.NumberColumn("Min", help="Solo continuous", format="%.2f"),
    "max": st.column_config.NumberColumn("Max", help="Solo continuous", format="%.2f"),
    "sd": st.column_config.NumberColumn("SD (opcional)", help="Solo continuous; vacio = no fijar", format="%.2f"),
    "categorias": st.column_config.TextColumn(
        "Categorias (solo categorical)",
        help="Formato 'Nivel:proporcion;Nivel:proporcion', ej: Hombre:48;Mujer:50;Otro:2. No hace falta que sumen 100.",
        width="large",
    ),
}


def _parse_categories_text(name: str, text: str) -> dict[str, float]:
    categories: dict[str, float] = {}
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SpecError(
                f"Variable '{name}': formato de categorias invalido en '{part}'. Usa 'Nivel:proporcion;Nivel:proporcion'."
            )
        level, prop = part.rsplit(":", 1)
        level = level.strip()
        try:
            categories[level] = float(prop.strip())
        except ValueError as exc:
            raise SpecError(f"Variable '{name}': proporcion invalida para '{level}' en '{part}'.") from exc
    return categories


def _row_value(row: pd.Series, key: str):
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def build_variables_from_table(df: pd.DataFrame) -> list[dict]:
    variables = []
    for _, row in df.iterrows():
        name = str(_row_value(row, "nombre") or "").strip()
        if not name:
            continue
        var_type = _row_value(row, "tipo") or "continuous"
        if var_type == "continuous":
            missing = [f for f in ("media", "mediana", "min", "max") if _row_value(row, f) is None]
            if missing:
                raise SpecError(f"Variable '{name}': faltan valores en {missing}.")
            var = {
                "name": name,
                "type": "continuous",
                "distribution": _row_value(row, "distribucion") or "normal",
                "mean": _row_value(row, "media"),
                "median": _row_value(row, "mediana"),
                "min": _row_value(row, "min"),
                "max": _row_value(row, "max"),
            }
            sd = _row_value(row, "sd")
            if sd is not None:
                var["sd"] = sd
            variables.append(var)
        else:
            var = {
                "name": name,
                "type": "categorical",
                "categories": _parse_categories_text(name, _row_value(row, "categorias")),
            }
            variables.append(var)
    return variables


def get_categorical_levels(df: pd.DataFrame, cat_name: str) -> list[str]:
    matches = df[df["nombre"] == cat_name]
    if matches.empty:
        return []
    try:
        return list(_parse_categories_text(cat_name, _row_value(matches.iloc[0], "categorias")).keys())
    except SpecError:
        return []


def init_state() -> None:
    if "relations" not in st.session_state:
        st.session_state.relations = []
    if "next_relation_id" not in st.session_state:
        st.session_state.next_relation_id = 0


def add_relation() -> None:
    rid = st.session_state.next_relation_id
    st.session_state.next_relation_id += 1
    st.session_state.relations.append(rid)
    st.session_state[f"rel_continuous_{rid}"] = None
    st.session_state[f"rel_categorical_{rid}"] = None


def render_relation(rid: int, continuous_names: list[str], categorical_names: list[str], variables_df: pd.DataFrame) -> bool:
    delete_requested = False
    with st.expander(f"🔗 Relacion #{rid + 1}", expanded=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        cont_options = [None] + continuous_names
        cat_options = [None] + categorical_names
        cont_name = c1.selectbox(
            "Variable continua afectada", cont_options, key=f"rel_continuous_{rid}", format_func=lambda x: x or "—"
        )
        cat_name = c2.selectbox(
            "Segun la categorica", cat_options, key=f"rel_categorical_{rid}", format_func=lambda x: x or "—"
        )
        c3.write("")
        if c3.form_submit_button("🗑️", key=f"rel_del_{rid}"):
            delete_requested = True

        if cont_name and cat_name:
            levels = get_categorical_levels(variables_df, cat_name)
            if not levels:
                st.caption("⚠️ No se encontraron niveles validos para esta categorica (revisa la columna 'categorias').")
            else:
                st.caption(
                    f"Cuanto se suma a la media/mediana base de '{cont_name}' para cada nivel de '{cat_name}' "
                    "(deja en 0 los niveles sin efecto)."
                )
                for level in levels:
                    lc1, lc2, lc3 = st.columns([1, 1, 1])
                    lc1.markdown(f"**{level}**")
                    lc2.number_input("Δ media", key=f"rel_mean_{rid}_{level}", format="%.2f")
                    lc3.number_input("Δ mediana", key=f"rel_median_{rid}_{level}", format="%.2f")
    return delete_requested


def build_effects_for_variable(var_name: str, variables_df: pd.DataFrame) -> list[dict]:
    effects = []
    for rid in st.session_state.relations:
        if st.session_state.get(f"rel_continuous_{rid}") != var_name:
            continue
        cat_name = st.session_state.get(f"rel_categorical_{rid}")
        if not cat_name:
            continue
        levels = get_categorical_levels(variables_df, cat_name)
        mean_delta = {lvl: st.session_state.get(f"rel_mean_{rid}_{lvl}", 0.0) for lvl in levels}
        median_delta = {lvl: st.session_state.get(f"rel_median_{rid}_{lvl}", 0.0) for lvl in levels}
        mean_delta = {k: v for k, v in mean_delta.items() if v}
        median_delta = {k: v for k, v in median_delta.items() if v}
        if mean_delta or median_delta:
            effects.append({"by": cat_name, "mean_delta": mean_delta, "median_delta": median_delta})
    return effects


def render_correlation_section(continuous_names: list[str]) -> tuple[list[str] | None, list[list[float]] | None]:
    if len(continuous_names) < 2:
        st.caption("Anade al menos dos variables continuas en la tabla para poder definir correlaciones entre ellas.")
        return None, None

    selected = st.multiselect("Variables continuas a correlacionar", options=continuous_names, key="corr_selection")
    if len(selected) < 2:
        st.caption("Selecciona al menos dos variables para editar su matriz de correlaciones.")
        return None, None

    default_matrix = pd.DataFrame(np.eye(len(selected)), index=selected, columns=selected)
    editor_key = "corr_matrix_" + "_".join(selected)
    st.caption("Edita los valores fuera de la diagonal (entre -1 y 1). La matriz se simetriza automaticamente.")
    edited = st.data_editor(default_matrix, key=editor_key)

    matrix = edited.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return selected, matrix.tolist()


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
    init_state()
    st.title("Simulador de bases de datos")
    st.write(
        "Define tus variables en la tabla (puedes anadir muchas filas de golpe), configura relaciones "
        "opcionales y genera un dataset simulado."
    )

    with st.sidebar:
        st.header("Configuracion general")
        n_rows = st.number_input("Numero de filas", min_value=1, value=1000, step=100)
        use_seed = st.checkbox("Fijar semilla aleatoria", value=True)
        seed = st.number_input("Semilla", min_value=0, value=42, step=1) if use_seed else None

    relation_to_delete = None
    with st.form("main_form", enter_to_submit=False):
        st.subheader("Variables")
        st.caption(
            "Una fila por variable. Anade o borra filas con los controles de la tabla (icono + al final, o "
            "seleccionar filas y pulsar la papelera). Los cambios se guardan al pulsar cualquier boton de "
            "esta seccion, incluido 🚀 Generar simulacion."
        )
        variables_df = st.data_editor(
            DEFAULT_VARIABLES_DF,
            key="variables_editor",
            num_rows="dynamic",
            use_container_width=True,
            column_config=COLUMN_CONFIG,
            column_order=VARIABLE_COLUMNS,
        )

        apply_clicked = st.form_submit_button("💾 Aplicar cambios", type="primary")

        continuous_names = [
            str(n).strip()
            for n, t in zip(variables_df.get("nombre", []), variables_df.get("tipo", []))
            if t == "continuous" and str(n).strip()
        ]
        categorical_names = [
            str(n).strip()
            for n, t in zip(variables_df.get("nombre", []), variables_df.get("tipo", []))
            if t == "categorical" and str(n).strip()
        ]

        st.subheader("Relaciones (opcional)")

        st.markdown("**Matriz de correlaciones entre continuas**")
        corr_names, corr_matrix = render_correlation_section(continuous_names)

        st.markdown("**Efecto de una categorica sobre una continua**")
        st.caption(
            "Por ejemplo: que la media de ingresos suba para las filas donde sexo = Mujer. "
            "Se puede anadir mas de una relacion."
        )
        add_relation_clicked = st.form_submit_button("➕ Anadir relacion")
        for rid in list(st.session_state.relations):
            if render_relation(rid, continuous_names, categorical_names, variables_df):
                relation_to_delete = rid

        st.divider()
        generate_clicked = st.form_submit_button("🚀 Generar simulacion", type="primary")

    if add_relation_clicked:
        add_relation()
        st.rerun()
    if relation_to_delete is not None:
        st.session_state.relations.remove(relation_to_delete)
        st.rerun()
    if apply_clicked:
        st.toast("Cambios aplicados.")

    if generate_clicked:
        try:
            raw_variables = build_variables_from_table(variables_df)
            for var in raw_variables:
                if var["type"] == "continuous":
                    effects = build_effects_for_variable(var["name"], variables_df)
                    if effects:
                        var["effects"] = effects

            raw = {"n_rows": int(n_rows), "seed": seed, "variables": raw_variables}
            if corr_names is not None:
                raw["correlation_matrix"] = {"variables": corr_names, "matrix": corr_matrix}

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

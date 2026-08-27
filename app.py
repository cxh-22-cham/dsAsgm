from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    actual_vs_predicted_figure, forecast_figure, metric_by_horizon_figure, rmse_figure,
)
from src.constants import MODEL_NAMES, PREDICTORS, VISIBLE_FIELDS
from src.contracts import validate_contracts
from src.data_access import load_all_bundles, load_all_data
from src.feature_builder import (
    EXTERNAL_FIELDS, build_feature_row, latest_manual_defaults, validate_manual_inputs,
)
from src.prediction import historical_results, predict_manual, selected_models


st.set_page_config(page_title="Daily Gold Price Forecasting", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Validating saved coursework evidence…")
def cached_data() -> dict:
    return load_all_data()


@st.cache_resource(show_spinner="Loading four saved deployment models…")
def cached_bundles() -> dict:
    return load_all_bundles()


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp, [data-testid="stAppViewContainer"] { background: #f8fafc !important; color: #111827 !important; }
        [data-testid="stHeader"] { background: rgba(248,250,252,.97) !important; }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span { color: #111827 !important; opacity: 1 !important; }

        /* Keep every input readable even when the viewer uses a dark browser theme. */
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] input {
            background: #ffffff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            border-color: #64748b !important;
            opacity: 1 !important;
        }
        [data-testid="stNumberInput"] input:disabled {
            background: #e2e8f0 !important;
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
        }
        [data-testid="stNumberInput"] button {
            background: #e2e8f0 !important;
            color: #111827 !important;
        }
        [data-testid="stNumberInput"] button svg { fill: #111827 !important; }
        [data-testid="stWidgetLabel"] p { color: #0f172a !important; font-weight: 700 !important; }

        /* Selectors and tabs remain high contrast under either system theme. */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #64748b !important;
        }
        [data-baseweb="tab-list"] { gap: .35rem; }
        button[data-baseweb="tab"] p { color: #334155 !important; font-weight: 700 !important; }
        button[data-baseweb="tab"][aria-selected="true"] p { color: #1d4ed8 !important; }

        .hero { padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#0f172a,#1d4ed8); color: white; margin-bottom: 1rem; }
        .hero h1, .hero p { color: white !important; margin: 0; }
        .hero p { margin-top: .55rem; opacity: .92; }
        .winner { border: 2px solid #d97706; background: #fffbeb; border-radius: 16px; padding: 1.1rem 1.25rem; color: #111827; }
        .winner, .winner div { color: #111827 !important; }
        .winner h3 { margin: 0 0 .5rem 0; color: #92400e !important; }
        .status { border-left: 6px solid #047857; background: #ecfdf5; color: #064e3b; padding: .8rem 1rem; border-radius: 8px; }
        .status, .status * { color: #064e3b !important; font-weight: 650; opacity: 1 !important; }
        .note { background:#eef2ff; border-left:5px solid #4338ca; color:#1e1b4b; padding:.8rem 1rem; border-radius:8px; }
        .note, .note * { color:#1e1b4b !important; opacity:1 !important; }

        div[data-testid="stMetric"] { background:#ffffff !important; border:1px solid #94a3b8; padding:.9rem; border-radius:12px; }
        [data-testid="stMetricLabel"] p { color:#334155 !important; font-weight:750 !important; opacity:1 !important; }
        [data-testid="stMetricValue"] { color:#0f172a !important; opacity:1 !important; }
        [data-testid="stMetricValue"] > div { color:#0f172a !important; opacity:1 !important; }

        button[kind="primary"], [data-testid="stFormSubmitButton"] button {
            background:#1d4ed8 !important;
            border-color:#1d4ed8 !important;
        }
        button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p,
        button[kind="primary"] span, [data-testid="stFormSubmitButton"] button span {
            color:#ffffff !important;
            font-weight:750 !important;
        }
        .table-wrap { overflow-x:auto; margin:.5rem 0 1.25rem 0; }
        .data-table { border-collapse:collapse; width:100%; background:white; color:#111827; font-size:.92rem; }
        .data-table th { background:#0f172a; color:white; padding:.65rem; text-align:left; white-space:nowrap; }
        .data-table td { border-bottom:1px solid #d1d5db; padding:.58rem .65rem; white-space:nowrap; }
        .data-table tr:nth-child(even) { background:#f1f5f9; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_results(frame: pd.DataFrame, historical: bool = False) -> pd.DataFrame:
    columns = [
        "Model", "Horizon", "Current Price", "Predicted Return",
        "Predicted Return Percentage", "Predicted Price Change", "Predicted Price", "Direction",
    ]
    if historical:
        columns += [
            "Target Date", "Actual Return (revealed after prediction)",
            "Actual Price (revealed after prediction)", "Persistence Price",
        ]
    return frame[columns].copy()


def show_result_table(frame: pd.DataFrame, historical: bool = False) -> None:
    display = format_results(frame, historical)
    formats = {
        "Current Price": "{:,.2f}",
        "Predicted Return": "{:.6f}",
        "Predicted Return Percentage": "{:.3f}%",
        "Predicted Price Change": "{:+,.2f}",
        "Predicted Price": "{:,.2f}",
    }
    if historical:
        formats.update({
            "Target Date": lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            "Actual Return (revealed after prediction)": "{:.6f}",
            "Actual Price (revealed after prediction)": "{:,.2f}",
            "Persistence Price": "{:,.2f}",
        })
    styled = display.style.format(formats).map(
        lambda value: "color:#047857;font-weight:700" if value == "Up" else (
            "color:#b91c1c;font-weight:700" if value == "Down" else "color:#4b5563;font-weight:700"
        ), subset=["Direction"]
    ).hide(axis="index").set_table_attributes('class="data-table"')
    st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)


def show_plain_table(frame: pd.DataFrame, formats: dict | None = None) -> None:
    styled = frame.style.hide(axis="index").set_table_attributes('class="data-table"')
    if formats:
        styled = styled.format(formats)
    st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)


def show_featured(featured: pd.DataFrame, best_model: str) -> None:
    h1 = featured.loc[(featured["Model"].eq(best_model)) & (featured["Horizon"].eq("H1"))].iloc[0]
    direction = h1["Direction"]
    st.markdown(f'<div class="winner"><h3>★ Official comparison winner: {best_model}</h3><div>This is the winner from the saved full Evaluation ranking—not from the current input.</div></div>', unsafe_allow_html=True)
    values = [
        ("Current Price", f'{h1["Current Price"]:,.2f}'),
        ("H1 Return", f'{h1["Predicted Return"]:.6f}'),
        ("H1 Return %", f'{h1["Predicted Return Percentage"]:.3f}%'),
        ("Price Change", f'{h1["Predicted Price Change"]:+,.2f}'),
        ("H1 Next Price", f'{h1["Predicted Price"]:,.2f}'),
        ("Direction", direction),
    ]
    for start in (0, 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, values[start:start + 3]):
            column.metric(label, value)


def historical_tab(data: dict, contracts: dict, selection: str) -> None:
    st.subheader("Historical leakage-safe Evaluation replay")
    st.caption("Uses saved expanding walk-forward predictions. The all-data deployment models are not called in this tab.")
    date_labels = [date.strftime("%Y-%m-%d") for date in contracts["common_dates"]]
    chosen_label = st.selectbox("Evaluation Origin Date (609 common dates only)", date_labels)
    chosen_date = pd.Timestamp(chosen_label)
    canonical_row = data["canonical"].loc[data["canonical"]["Origin_Date"].eq(chosen_date)].iloc[0]
    cols = st.columns(4)
    for index, field in enumerate(VISIBLE_FIELDS):
        cols[index % 4].number_input(field, value=float(canonical_row[field]), disabled=True, key=f"history_{field}")

    if st.button("Predict from saved Evaluation evidence", type="primary", key="history_predict"):
        detailed_models = selected_models(selection, MODEL_NAMES)
        featured = historical_results(data["predictions"], chosen_date, [contracts["best_model"]])
        detailed = historical_results(data["predictions"], chosen_date, detailed_models)
        st.session_state["historical_output"] = (chosen_label, selection, featured, detailed)

    output = st.session_state.get("historical_output")
    if output and output[0] == chosen_label and output[1] == selection:
        _, _, featured, detailed = output
        show_featured(featured, contracts["best_model"])
        st.markdown("#### Selected model results")
        show_result_table(detailed, historical=True)
        st.plotly_chart(forecast_figure(detailed, float(canonical_row["Current_Price"]), contracts["best_model"], historical=True), width="stretch")
        st.caption("Actual values are shown only because this is a historical replay; they were revealed after each original forecast.")


def manual_tab(data: dict, bundles: dict, contracts: dict, selection: str) -> None:
    st.subheader("Manual Input — deployment forecast")
    st.caption("Enter one current market record. Each H1–H7 value is produced directly by its own saved fitted Pipeline.")
    defaults, prior, external = latest_manual_defaults(data["canonical"])
    with st.form("manual_form"):
        columns = st.columns(4)
        entered = {}
        for index, field in enumerate(VISIBLE_FIELDS):
            step = 1.0 if field != "Current_Volume" else 100.0
            entered[field] = columns[index % 4].number_input(field, value=float(defaults[field]), step=step, format="%.6f")
        st.markdown(
            f'<div class="note"><b>Read-only stored external values</b><br>'
            f'USD Index return lag 1: {external[EXTERNAL_FIELDS[0]]:.6f}<br>'
            f'US 10Y real-yield change lag 1: {external[EXTERNAL_FIELDS[1]]:.6f}<br>'
            'These are the latest stored coursework values—not live data.</div>',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Generate direct H1–H7 forecast", type="primary")

    if submitted:
        errors = validate_manual_inputs(entered)
        if errors:
            for error in errors:
                st.error(error)
        else:
            feature_row = build_feature_row(entered, prior, external, PREDICTORS)
            detailed_models = selected_models(selection, MODEL_NAMES)
            featured = predict_manual(feature_row, bundles, [contracts["best_model"]])
            detailed = predict_manual(feature_row, bundles, detailed_models)
            st.session_state["manual_output"] = (selection, tuple(entered.values()), featured, detailed, feature_row)

    output = st.session_state.get("manual_output")
    current_signature = tuple(entered.values())
    if output and output[0] == selection and output[1] == current_signature:
        _, _, featured, detailed, feature_row = output
        show_featured(featured, contracts["best_model"])
        st.markdown("#### Selected model results")
        show_result_table(detailed)
        st.plotly_chart(forecast_figure(detailed, float(feature_row.iloc[0]["Current_Price"]), contracts["best_model"], historical=False), width="stretch")
        with st.expander("View the 22 ordered predictors sent to every Pipeline"):
            show_plain_table(feature_row, {column: "{:.8g}" for column in feature_row.columns})


def comparison_section(data: dict, best_model: str) -> None:
    st.divider()
    st.header("Historical four-model comparison")
    st.caption("Always based on the saved 609-origin leakage-safe Evaluation evidence, never on current manual inputs.")
    ranking = data["ranking"].sort_values("Rank")
    show_plain_table(ranking, {
        "Overall_Price_RMSE": "{:,.2f}", "Overall_Price_MAE": "{:,.2f}",
        "Overall_Price_MAPE_Percent": "{:.3f}%", "Overall_Return_R2": "{:.4f}",
        "Overall_RMSE_Skill_vs_Persistence": "{:.4f}",
    })
    winner = ranking.iloc[0]
    st.success(
        f'{best_model} is the official winner because it has the lowest saved pooled Overall Price RMSE '
        f'({winner["Overall_Price_RMSE"]:,.2f}) without recalculating from rounded display values. '
        f'Its pooled RMSE skill versus persistence is {winner["Overall_RMSE_Skill_vs_Persistence"] * 100:.2f}%.'
    )

    metrics = data["comparison_metrics"]
    left, right = st.columns(2)
    left.plotly_chart(rmse_figure(metrics), width="stretch")
    metric_options = {
        "RMSE skill vs persistence": "RMSE_Skill_vs_Persistence",
        "Price MAE": "Price_MAE",
        "Price MAPE (%)": "Price_MAPE_Percent",
        "Return R²": "Return_R2",
        "Directional accuracy (%)": "Directional_Accuracy_Percent",
    }
    label = right.selectbox("Additional metric", list(metric_options), key="comparison_metric")
    right.plotly_chart(metric_by_horizon_figure(metrics, metric_options[label], label), width="stretch")

    st.markdown("#### Directional accuracy versus Always-Up")
    direction = metrics.loc[metrics["Horizon"].ne("Overall"), ["Model", "Horizon", "Directional_Accuracy_Percent", "Always_Up_Accuracy_Percent"]]
    show_plain_table(direction, {
        "Directional_Accuracy_Percent": "{:.2f}%", "Always_Up_Accuracy_Percent": "{:.2f}%",
    })

    chart_col1, chart_col2 = st.columns([1, 1])
    historical_model = chart_col1.selectbox("Historical chart model", MODEL_NAMES, index=MODEL_NAMES.index(best_model))
    historical_horizon = chart_col2.selectbox("Historical chart horizon", [f"H{h}" for h in range(1, 8)])
    st.plotly_chart(actual_vs_predicted_figure(data["predictions"][historical_model], historical_model, historical_horizon), width="stretch")

    st.markdown(
        '<div class="note"><b>How to read these metrics:</b> Positive RMSE skill means the model beat persistence; negative skill means persistence was better. '
        'A high Price R² can mainly reflect adjacent-price persistence, while Return R² tests the harder return signal. '
        'Directional accuracy should be judged against the saved Always-Up accuracy. Overall rows pool all H1–H7 forecasts.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_style()
    st.markdown('<div class="hero"><h1>📈 Daily Gold Price Forecasting</h1><p>Direct cumulative return forecasts for H1–H7 using Ridge, KNN, SVR and XGBoost.</p></div>', unsafe_allow_html=True)
    try:
        data = cached_data()
        bundles = cached_bundles()
        contracts = validate_contracts(data, bundles)
    except Exception as error:
        st.error(f"Startup validation failed: {error}")
        st.stop()

    st.markdown(
        f'<div class="status"><b>Startup validation: PASS</b> — 4 models, 7 Pipelines each, 22 predictors, '
        f'609 Evaluation origins, and exact feature parity (max error {contracts["feature_parity"]["max_absolute_error"]:.2e}).</div>',
        unsafe_allow_html=True,
    )
    selection = st.selectbox("Forecast display", ["All Models", *MODEL_NAMES])
    tab1, tab2 = st.tabs(["Existing Evaluation Date", "Manual Input"])
    with tab1:
        historical_tab(data, contracts, selection)
    with tab2:
        manual_tab(data, bundles, contracts, selection)

    st.info("The two tabs use different valid model states: historical replay uses step-specific saved walk-forward evidence; manual forecasting uses final all-data deployment Pipelines. Their outputs are not expected to match.")
    comparison_section(data, contracts["best_model"])

    with st.expander("Model information, methodology and limitations", expanded=True):
        st.markdown(
            """
            - **Objective:** educational comparison of Ridge, KNN, SVR and XGBoost for direct cumulative gold-price return forecasting.
            - **Direct horizons:** H1–H7 mean recorded observations ahead, not guaranteed calendar days. No H1 forecast is fed recursively into H2–H7.
            - **Leakage safety:** Existing Date mode only replays saved walk-forward Evaluation predictions. Manual mode only calls saved fitted deployment Pipelines. The app never fits, tunes, resplits or shuffles.
            - **Feature construction:** the 15 backend predictors use verified percentage-change, rolling-mean, sample-volatility (`ddof=1`) and momentum formulas. External lag values are the latest stored coursework values.
            - **Limitations:** this is an educational forecasting prototype, not financial advice. No live data is fetched. Regime shift can weaken deployment performance; all predictions are uncertain and are not guaranteed prices.
            """
        )


if __name__ == "__main__":
    main()

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
        .block-container { max-width: 1480px !important; padding-top: 2.2rem !important; padding-bottom: 4rem !important; }
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
        [data-baseweb="tab-list"] { gap:.35rem; background:#e8eef7; padding:.28rem; border-radius:12px; width:fit-content; }
        button[data-baseweb="tab"] { border-radius:9px; padding:.45rem .9rem; }
        button[data-baseweb="tab"] p { color:#334155 !important; font-weight:750 !important; }
        button[data-baseweb="tab"][aria-selected="true"] { background:#ffffff !important; box-shadow:0 2px 8px rgba(15,23,42,.10); }
        button[data-baseweb="tab"][aria-selected="true"] p { color:#1d4ed8 !important; }

        .hero {
            position:relative;
            overflow:hidden;
            padding:2.25rem 2.5rem;
            border:1px solid rgba(255,255,255,.16);
            border-radius:24px;
            background:linear-gradient(118deg,#071426 0%,#102a5f 54%,#1d4ed8 100%);
            box-shadow:0 18px 45px rgba(15,23,42,.18);
            color:#ffffff !important;
            margin-bottom:1.15rem;
        }
        .hero::before {
            content:"";
            position:absolute;
            width:320px;
            height:320px;
            right:-75px;
            top:-150px;
            border-radius:50%;
            background:radial-gradient(circle,rgba(251,191,36,.35),rgba(251,191,36,0) 68%);
        }
        .hero::after {
            content:"";
            position:absolute;
            width:220px;
            height:220px;
            right:15%;
            bottom:-170px;
            border-radius:50%;
            background:rgba(96,165,250,.22);
            filter:blur(4px);
        }
        .hero-content { position:relative; z-index:2; }
        .hero-badge {
            display:inline-flex;
            align-items:center;
            padding:.35rem .72rem;
            border:1px solid rgba(255,255,255,.28);
            border-radius:999px;
            background:rgba(255,255,255,.10);
            color:#dbeafe !important;
            font-size:.78rem;
            font-weight:800;
            letter-spacing:.09em;
            text-transform:uppercase;
        }
        .hero-title {
            max-width:950px;
            margin-top:1rem;
            color:#ffffff !important;
            font-size:clamp(2.25rem,4.2vw,3.85rem);
            font-weight:850;
            letter-spacing:-.045em;
            line-height:1.02;
            text-shadow:0 3px 18px rgba(0,0,0,.18);
        }
        .hero-title .gold { color:#fbbf24 !important; }
        .hero-subtitle {
            max-width:850px;
            margin-top:1rem;
            color:#dbeafe !important;
            font-size:1.08rem;
            line-height:1.6;
        }
        .hero-tags { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.15rem; }
        .hero .hero-tag {
            padding:.4rem .72rem;
            border-radius:9px;
            background:rgba(15,23,42,.42);
            color:#ffffff !important;
            font-size:.84rem;
            font-weight:700;
            border:1px solid rgba(255,255,255,.16);
        }
        .winner { border: 1px solid #f59e0b; background: #fffbeb; border-radius: 12px; padding: .72rem 1rem; color: #111827; }
        .winner, .winner div { color: #111827 !important; }
        .winner h3 { margin: 0; color: #92400e !important; font-size:1.02rem; }
        .status { border-left: 6px solid #047857; background: #ecfdf5; color: #064e3b; padding: .8rem 1rem; border-radius: 8px; }
        .status, .status * { color: #064e3b !important; font-weight: 650; opacity: 1 !important; }
        .note { background:#eef2ff; border-left:5px solid #4338ca; color:#1e1b4b; padding:.8rem 1rem; border-radius:8px; }
        .note, .note * { color:#1e1b4b !important; opacity:1 !important; }

        .section-heading {
            display:flex;
            align-items:flex-start;
            gap:.9rem;
            margin:1.8rem 0 .9rem 0;
            padding:1rem 1.15rem;
            background:#ffffff;
            border:1px solid #dbe2ea;
            border-left:5px solid #1d4ed8;
            border-radius:14px;
            box-shadow:0 5px 16px rgba(15,23,42,.06);
        }
        .section-heading .section-number {
            display:flex;
            align-items:center;
            justify-content:center;
            flex:0 0 2.45rem;
            height:2.45rem;
            border-radius:10px;
            background:#dbeafe;
            color:#1d4ed8 !important;
            font-size:.82rem;
            font-weight:850;
            letter-spacing:.03em;
        }
        .section-heading .section-heading-title {
            color:#0f172a !important;
            font-size:1.08rem;
            font-weight:850;
            line-height:1.25;
            margin:.02rem 0 .22rem 0;
        }
        .section-heading .section-heading-copy {
            color:#475569 !important;
            font-size:.9rem;
            line-height:1.45;
        }
        .comparison-banner {
            margin:2.2rem 0 1.1rem 0;
            padding:1.35rem 1.5rem;
            border:1px solid #bfdbfe;
            border-radius:18px;
            background:linear-gradient(115deg,#eff6ff 0%,#ffffff 72%);
            box-shadow:0 8px 22px rgba(30,64,175,.07);
        }
        .comparison-banner .comparison-eyebrow {
            color:#1d4ed8 !important;
            font-size:.77rem;
            font-weight:850;
            letter-spacing:.1em;
            text-transform:uppercase;
        }
        .comparison-banner .comparison-title {
            color:#0f172a !important;
            font-size:1.65rem;
            font-weight:850;
            letter-spacing:-.025em;
            margin:.25rem 0 .3rem 0;
        }
        .comparison-banner .comparison-copy {
            color:#475569 !important;
            font-size:.95rem;
            line-height:1.5;
        }

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

        /* Streamlit Cloud may otherwise render the expander as black-on-black. */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details {
            background:#ffffff !important;
            border:1px solid #cbd5e1 !important;
            border-radius:12px !important;
            overflow:hidden;
        }
        [data-testid="stExpander"] summary {
            background:#e8eef7 !important;
            color:#0f172a !important;
            padding:.75rem 1rem !important;
        }
        [data-testid="stExpander"] summary:hover { background:#dbe7f7 !important; }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color:#0f172a !important;
            font-weight:750 !important;
            opacity:1 !important;
        }
        [data-testid="stExpander"] summary svg {
            fill:#1d4ed8 !important;
            color:#1d4ed8 !important;
        }
        [data-testid="stExpanderDetails"] {
            background:#ffffff !important;
            color:#111827 !important;
            padding:.75rem 1rem 1rem !important;
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


def section_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-number">{number}</div>
          <div>
            <div class="section-heading-title">{title}</div>
            <div class="section-heading-copy">{description}</div>
          </div>
        </div>
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


def show_result_browser(frame: pd.DataFrame, historical: bool, key_prefix: str) -> None:
    view = st.radio(
        "Results view",
        ["Show one horizon", "Show all horizons"],
        horizontal=True,
        key=f"{key_prefix}_results_view",
    )

    if view == "Show all horizons":
        show_result_table(frame, historical=historical)
        return

    horizon_options = frame["Horizon"].drop_duplicates().tolist()
    chosen_horizon = st.selectbox(
        "Horizon",
        horizon_options,
        key=f"{key_prefix}_result_horizon",
    )
    selected = frame.loc[frame["Horizon"].eq(chosen_horizon)]
    show_result_table(selected, historical=historical)


def show_plain_table(frame: pd.DataFrame, formats: dict | None = None) -> None:
    styled = frame.style.hide(axis="index").set_table_attributes('class="data-table"')
    if formats:
        styled = styled.format(formats)
    st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)


def show_featured(featured: pd.DataFrame, best_model: str) -> None:
    h1 = featured.loc[(featured["Model"].eq(best_model)) & (featured["Horizon"].eq("H1"))].iloc[0]
    direction = h1["Direction"]
    st.markdown(f'<div class="winner"><h3>★ Best overall Evaluation model: {best_model}</h3></div>', unsafe_allow_html=True)
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
    st.subheader("Historical Evaluation results")
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
        section_header("A", "Detailed forecasts", "The results follow the model selected above. Show one horizon or all seven horizons.")
        show_result_browser(
            detailed,
            historical=True,
            key_prefix=f'historical_{selection.replace(" ", "_").lower()}',
        )
        section_header("B", "Forecast path", "Compare the seven direct forecast horizons with the current price and the revealed historical path.")
        st.plotly_chart(
            forecast_figure(detailed, float(canonical_row["Current_Price"]), contracts["best_model"], historical=True),
            width="stretch", theme=None,
        )
        st.caption("Actual values are available here because this date is part of the completed Evaluation period.")


def manual_tab(data: dict, bundles: dict, contracts: dict, selection: str) -> None:
    st.subheader("Manual Input Forecast")
    st.caption("Enter the current market values to generate H1–H7 forecasts.")
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
            'Stored values used by the models; they are not live data.</div>',
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
        section_header("A", "Detailed forecasts", "The results follow the model selected above. Show one horizon or all seven horizons.")
        show_result_browser(
            detailed,
            historical=False,
            key_prefix=f'manual_{selection.replace(" ", "_").lower()}',
        )
        section_header("B", "Forecast path", "See the projected price path across all seven forecast horizons in one full-width chart.")
        st.plotly_chart(
            forecast_figure(detailed, float(feature_row.iloc[0]["Current_Price"]), contracts["best_model"], historical=False),
            width="stretch", theme=None,
        )
        with st.expander("View the 22 ordered predictors sent to every Pipeline"):
            show_plain_table(feature_row, {column: "{:.8g}" for column in feature_row.columns})


def comparison_section(data: dict, best_model: str) -> None:
    st.markdown(
        """
        <div class="comparison-banner">
          <div class="comparison-eyebrow">Model Performance</div>
          <div class="comparison-title">Historical Four-Model Comparison</div>
          <div class="comparison-copy">Compare Ridge, KNN, SVR and XGBoost across the same Evaluation period.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("01", "Overall model ranking", "Models are ranked using the saved pooled Overall Price RMSE across H1–H7.")
    ranking = data["ranking"].sort_values("Rank")
    show_plain_table(ranking, {
        "Overall_Price_RMSE": "{:,.2f}", "Overall_Price_MAE": "{:,.2f}",
        "Overall_Price_MAPE_Percent": "{:.3f}%", "Overall_Return_R2": "{:.4f}",
        "Overall_RMSE_Skill_vs_Persistence": "{:.4f}",
    })
    winner = ranking.iloc[0]
    st.markdown(
        f'**Best overall model: {best_model}** — Overall Price RMSE: '
        f'{winner["Overall_Price_RMSE"]:,.2f} · RMSE skill: '
        f'{winner["Overall_RMSE_Skill_vs_Persistence"] * 100:.2f}%'
    )

    metrics = data["comparison_metrics"]
    section_header("02", "Price RMSE by horizon", "Compare prediction error for every model from H1 to H7. Lower RMSE indicates better price accuracy.")
    st.plotly_chart(rmse_figure(metrics), width="stretch", theme=None)

    section_header("03", "Explore another metric", "Choose one additional measure below. Its chart is displayed on a separate full-width row for easier reading.")
    metric_options = {
        "RMSE skill vs persistence": "RMSE_Skill_vs_Persistence",
        "Price MAE": "Price_MAE",
        "Price MAPE (%)": "Price_MAPE_Percent",
        "Return R²": "Return_R2",
        "Directional accuracy (%)": "Directional_Accuracy_Percent",
    }
    label = st.selectbox("Additional metric", list(metric_options), key="comparison_metric")
    st.plotly_chart(
        metric_by_horizon_figure(metrics, metric_options[label], label),
        width="stretch", theme=None,
    )

    section_header("04", "Directional accuracy versus Always-Up", "Check whether each model predicts up/down movement better than the simple Always-Up benchmark.")
    direction = metrics.loc[metrics["Horizon"].ne("Overall"), ["Model", "Horizon", "Directional_Accuracy_Percent", "Always_Up_Accuracy_Percent"]]
    show_plain_table(direction, {
        "Directional_Accuracy_Percent": "{:.2f}%", "Always_Up_Accuracy_Percent": "{:.2f}%",
    })

    section_header("05", "Historical actual versus predicted prices", "Select one model and one horizon, then inspect its saved prediction path against actual prices and persistence.")
    chart_col1, chart_col2 = st.columns([1, 1])
    historical_model = chart_col1.selectbox("Historical chart model", MODEL_NAMES, index=MODEL_NAMES.index(best_model))
    historical_horizon = chart_col2.selectbox("Historical chart horizon", [f"H{h}" for h in range(1, 8)])
    st.plotly_chart(
        actual_vs_predicted_figure(data["predictions"][historical_model], historical_model, historical_horizon),
        width="stretch", theme=None,
    )

def main() -> None:
    apply_style()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-content">
            <div class="hero-badge">BMDS2003 · Data Science Project</div>
            <div class="hero-title">Daily Gold Price<br><span class="gold">Forecasting Dashboard</span></div>
            <div class="hero-subtitle">
              Compare four machine-learning models and explore direct H1–H7 gold-price forecasts.
            </div>
            <div class="hero-tags">
              <span class="hero-tag">H1–H7 Direct Forecasts</span>
              <span class="hero-tag">Ridge · KNN · SVR · XGBoost</span>
              <span class="hero-tag">No Retraining</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        data = cached_data()
        bundles = cached_bundles()
        contracts = validate_contracts(data, bundles)
    except Exception as error:
        st.error(f"Startup validation failed: {error}")
        st.stop()

    selection = st.selectbox("Forecast display", ["All Models", *MODEL_NAMES])
    tab1, tab2 = st.tabs(["Existing Evaluation Date", "Manual Input"])
    with tab1:
        historical_tab(data, contracts, selection)
    with tab2:
        manual_tab(data, bundles, contracts, selection)

    comparison_section(data, contracts["best_model"])

    with st.expander("Methodology and limitations", expanded=False):
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

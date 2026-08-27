from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {"KNN": "#F59E0B", "Ridge": "#2563EB", "SVR": "#059669", "XGBoost": "#DC2626"}
FUTURE_COLOR = "#7C3AED"


def forecast_figure(
    results: pd.DataFrame,
    current_price: float,
    best_model: str,
    historical: bool,
    forecast_origin_date=None,
) -> go.Figure:
    figure = go.Figure()
    for model, group in results.groupby("Model", sort=False):
        ordered = group.assign(Horizon_Number=group["Horizon"].str.removeprefix("H").astype(int)).sort_values("Horizon_Number")
        if historical:
            x_values = ordered["Target Date"].dt.strftime("%Y-%m-%d").tolist()
        elif forecast_origin_date is not None:
            origin = pd.Timestamp(forecast_origin_date)
            future_dates = pd.bdate_range(origin + pd.offsets.BDay(1), periods=len(ordered))
            x_values = future_dates.tolist()
        else:
            x_values = ordered["Horizon"].tolist()
        y_values = ordered["Predicted Price"].tolist()
        figure.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", name=model,
            customdata=[[horizon] for horizon in ordered["Horizon"]],
            hovertemplate="%{customdata[0]}<br>Price: %{y:,.2f}<extra>%{fullData.name}</extra>",
            line={"width": 3 if model == best_model else 2, "color": COLORS.get(model)},
            marker={"size": 6},
        ))
    if historical:
        actual = results.sort_values("Horizon").drop_duplicates("Horizon")
        figure.add_trace(go.Scatter(
            x=actual["Target Date"].dt.strftime("%Y-%m-%d").tolist(),
            y=actual["Actual Price (revealed after prediction)"].tolist(),
            mode="lines+markers", name="Actual (revealed later)", line={"color": "#111827", "dash": "dash", "width": 3},
        ))
        figure.add_trace(go.Scatter(
            x=actual["Target Date"].dt.strftime("%Y-%m-%d").tolist(),
            y=actual["Persistence Price"].tolist(),
            mode="lines", name="Persistence", line={"color": "#6B7280", "dash": "dot"},
        ))
        x_title = "Actual target dates"
    elif forecast_origin_date is not None:
        x_title = "Forecast date (next recorded business-day positions)"
    else:
        x_title = "Recorded observations ahead"
    figure.update_layout(
        template="plotly_white", title="Direct H1–H7 reconstructed price forecasts",
        xaxis_title=x_title, yaxis_title="Gold price (dataset units)", hovermode="x unified",
        legend_title="Series", height=470, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"color": "#111827"},
    )
    figure.update_xaxes(gridcolor="#e5e7eb")
    figure.update_yaxes(gridcolor="#e5e7eb")
    return figure


def metric_by_horizon_figure(metrics: pd.DataFrame, metric: str, title: str) -> go.Figure:
    horizon = metrics.loc[metrics["Horizon"].ne("Overall")].copy()
    horizon["Horizon_Number"] = horizon["Horizon"].str.removeprefix("H").astype(int)
    figure = px.line(
        horizon, x="Horizon_Number", y=metric, color="Model", markers=True,
        title=title, color_discrete_map=COLORS,
        labels={"Horizon_Number": "Horizon", metric: title}, template="plotly_white",
    )
    figure.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#111827"})
    return figure


def rmse_figure(metrics: pd.DataFrame) -> go.Figure:
    horizon = metrics.loc[metrics["Horizon"].ne("Overall")].copy()
    horizon["Horizon_Number"] = horizon["Horizon"].str.removeprefix("H").astype(int)
    figure = px.line(horizon, x="Horizon_Number", y="Price_RMSE", color="Model", markers=True, color_discrete_map=COLORS, title="Price RMSE by horizon")
    persistence = horizon.drop_duplicates("Horizon_Number").sort_values("Horizon_Number")
    figure.add_trace(go.Scatter(x=persistence["Horizon_Number"], y=persistence["Persistence_Price_RMSE"], name="Persistence", mode="lines+markers", line={"color": "#6B7280", "dash": "dash", "width": 3}))
    figure.update_layout(
        template="plotly_white", xaxis_title="Horizon", yaxis_title="RMSE",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#111827"},
    )
    return figure


def actual_vs_predicted_figure(
    predictions: pd.DataFrame,
    model: str,
    horizon: str,
    future_results: pd.DataFrame | None = None,
    forecast_origin_date=None,
) -> go.Figure:
    available_models = predictions["Model"].drop_duplicates().tolist()
    displayed_models = available_models if model == "All Models" else [model]
    reference_model = displayed_models[0]
    reference_predictions = predictions.loc[predictions["Model"].eq(reference_model)].copy()
    data = reference_predictions.loc[reference_predictions["Horizon"].eq(horizon)].sort_values("Target_Date")

    # Actual prices are shared across horizons. Combine all available target
    # dates so the real dataset line always reaches its true final record,
    # regardless of which prediction horizon is selected for comparison.
    actual_history = (
        reference_predictions[["Target_Date", "Actual_Price"]]
        .sort_values("Target_Date")
        .drop_duplicates("Target_Date", keep="last")
    )

    # Keep the selected-horizon persistence history, then complete its missing
    # tail through the final dataset date using the latest available origin for
    # each subsequent target date.
    selected_persistence = data[["Target_Date", "Persistence_Price"]].copy()
    persistence_tail = (
        reference_predictions.loc[
            reference_predictions["Target_Date"].gt(selected_persistence["Target_Date"].max()),
            ["Origin_Date", "Target_Date", "Persistence_Price"],
        ]
        .sort_values(["Target_Date", "Origin_Date"])
        .drop_duplicates("Target_Date", keep="last")
        [["Target_Date", "Persistence_Price"]]
    )
    persistence_history = (
        pd.concat([selected_persistence, persistence_tail], ignore_index=True)
        .sort_values("Target_Date")
        .drop_duplicates("Target_Date", keep="first")
    )

    figure = go.Figure()
    focus_start = None
    focus_end = None

    # Actual is deliberately drawn first as the bottom layer.
    figure.add_trace(go.Scatter(
        x=actual_history["Target_Date"], y=actual_history["Actual_Price"],
        name="Actual",
        line={"color": "#111827", "width": 2.2},
    ))
    for displayed_model in displayed_models:
        model_history = predictions.loc[
            predictions["Model"].eq(displayed_model)
            & predictions["Horizon"].eq(horizon)
        ].sort_values("Target_Date")
        figure.add_trace(go.Scatter(
            x=model_history["Target_Date"], y=model_history["Predicted_Price"],
            name=f"{displayed_model} historical prediction",
            line={"color": COLORS.get(displayed_model), "width": 1.8},
        ))
    figure.add_trace(go.Scatter(
        x=persistence_history["Target_Date"], y=persistence_history["Persistence_Price"],
        name="Persistence",
        line={"color": "#6B7280", "width": 1.8, "dash": "dot"},
    ))

    title = f"Actual vs predicted — {model} {horizon}"
    if future_results is not None and not future_results.empty and forecast_origin_date is not None:
        origin = pd.Timestamp(forecast_origin_date)
        final_actual = actual_history.iloc[-1]
        future_dates = None
        for future_model, future_group in future_results.groupby("Model", sort=False):
            future = future_group.assign(
                Horizon_Number=future_group["Horizon"].str.removeprefix("H").astype(int)
            ).sort_values("Horizon_Number")
            future_dates = pd.bdate_range(origin + pd.offsets.BDay(1), periods=len(future))
            future_color = FUTURE_COLOR if len(displayed_models) == 1 else COLORS.get(future_model)
            figure.add_trace(go.Scatter(
                x=[final_actual["Target_Date"]] + future_dates.tolist(),
                y=[float(final_actual["Actual_Price"])] + future["Predicted Price"].tolist(),
                mode="lines",
                name=f"{future_model} future H1–H7",
                hovertemplate="(%{x|%b %d, %Y}, %{y:,.2f})<extra></extra>",
                hoverlabel={
                    "bgcolor": future_color,
                    "bordercolor": future_color,
                    "font": {"color": "#ffffff"},
                },
                connectgaps=True,
                line={"color": future_color, "width": 2.8, "dash": "solid", "shape": "linear"},
            ))
        focus_start = pd.Timestamp(actual_history["Target_Date"].max()) - pd.Timedelta(days=45)
        focus_end = pd.Timestamp(future_dates[-1]) + pd.Timedelta(days=1)
        title = (
            "Historical performance + future H1–H7 forecast — All Models"
            if model == "All Models"
            else f"Historical performance + future H1–H7 forecast — {model}"
        )

    figure.update_layout(
        template="plotly_white", title=title,
        xaxis_title="Target date", yaxis_title="Gold price (dataset units)", height=430,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#111827"},
    )
    xaxis_settings = {
        "gridcolor": "#e5e7eb",
        "rangeselector": {
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
            "bgcolor": "#ffffff",
            "bordercolor": "#cbd5e1",
            "borderwidth": 1,
        },
    }
    if focus_start is not None and focus_end is not None:
        xaxis_settings["range"] = [focus_start, focus_end]
    figure.update_xaxes(**xaxis_settings)
    figure.update_yaxes(gridcolor="#e5e7eb")
    return figure

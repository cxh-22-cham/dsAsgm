from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {"KNN": "#F59E0B", "Ridge": "#2563EB", "SVR": "#059669", "XGBoost": "#DC2626"}


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
            x_values = ["H0"] + ordered["Target Date"].dt.strftime("%Y-%m-%d").tolist()
        elif forecast_origin_date is not None:
            origin = pd.Timestamp(forecast_origin_date)
            future_dates = pd.bdate_range(origin + pd.offsets.BDay(1), periods=len(ordered))
            x_values = [origin] + future_dates.tolist()
        else:
            x_values = ["H0"] + ordered["Horizon"].tolist()
        y_values = [current_price] + ordered["Predicted Price"].tolist()
        figure.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", name=model,
            customdata=[["H0"]] + [[horizon] for horizon in ordered["Horizon"]],
            hovertemplate="%{customdata[0]}<br>Price: %{y:,.2f}<extra>%{fullData.name}</extra>",
            line={"width": 5 if model == best_model else 2.5, "color": COLORS.get(model)},
        ))
    if historical:
        actual = results.sort_values("Horizon").drop_duplicates("Horizon")
        figure.add_trace(go.Scatter(
            x=["H0"] + actual["Target Date"].dt.strftime("%Y-%m-%d").tolist(),
            y=[current_price] + actual["Actual Price (revealed after prediction)"].tolist(),
            mode="lines+markers", name="Actual (revealed later)", line={"color": "#111827", "dash": "dash", "width": 3},
        ))
        figure.add_trace(go.Scatter(
            x=["H0"] + actual["Target Date"].dt.strftime("%Y-%m-%d").tolist(),
            y=[current_price] + actual["Persistence Price"].tolist(),
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
    data = predictions.loc[(predictions["Model"].eq(model)) & (predictions["Horizon"].eq(horizon))].sort_values("Target_Date")
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data["Target_Date"], y=data["Actual_Price"], name="Actual", line={"color": "#111827"}))
    figure.add_trace(go.Scatter(x=data["Target_Date"], y=data["Predicted_Price"], name=f"{model} historical prediction", line={"color": COLORS.get(model)}))
    figure.add_trace(go.Scatter(x=data["Target_Date"], y=data["Persistence_Price"], name="Persistence", line={"color": "#6B7280", "dash": "dot"}))

    title = f"Actual vs predicted — {model} {horizon}"
    if future_results is not None and not future_results.empty and forecast_origin_date is not None:
        future = future_results.assign(
            Horizon_Number=future_results["Horizon"].str.removeprefix("H").astype(int)
        ).sort_values("Horizon_Number")
        origin = pd.Timestamp(forecast_origin_date)
        future_dates = pd.bdate_range(origin + pd.offsets.BDay(1), periods=len(future))
        current_price = float(future.iloc[0]["Current Price"])
        figure.add_trace(go.Scatter(
            x=[origin] + future_dates.tolist(),
            y=[current_price] + future["Predicted Price"].tolist(),
            mode="lines+markers",
            name=f"{model} future H1–H7",
            customdata=[["H0"]] + [[horizon_name] for horizon_name in future["Horizon"]],
            hovertemplate="%{customdata[0]}<br>Price: %{y:,.2f}<extra>%{fullData.name}</extra>",
            line={"color": COLORS.get(model), "width": 4, "dash": "dash"},
            marker={"size": 8},
        ))
        figure.add_vline(
            x=origin.to_pydatetime(),
            line={"color": "#7C3AED", "width": 2, "dash": "dot"},
        )
        title = f"Historical performance + future H1–H7 forecast — {model}"

    figure.update_layout(
        template="plotly_white", title=title,
        xaxis_title="Target date", yaxis_title="Gold price (dataset units)", height=430,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#111827"},
    )
    figure.update_xaxes(gridcolor="#e5e7eb")
    figure.update_yaxes(gridcolor="#e5e7eb")
    return figure

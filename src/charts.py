from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {"KNN": "#F59E0B", "Ridge": "#2563EB", "SVR": "#059669", "XGBoost": "#DC2626"}


def forecast_figure(results: pd.DataFrame, current_price: float, best_model: str, historical: bool) -> go.Figure:
    figure = go.Figure()
    for model, group in results.groupby("Model", sort=False):
        ordered = group.assign(Horizon_Number=group["Horizon"].str.removeprefix("H").astype(int)).sort_values("Horizon_Number")
        x_values = ["H0"] + (ordered["Target Date"].dt.strftime("%Y-%m-%d").tolist() if historical else ordered["Horizon"].tolist())
        y_values = [current_price] + ordered["Predicted Price"].tolist()
        figure.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", name=model,
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
    else:
        x_title = "Recorded observations ahead"
    figure.update_layout(title="Direct H1–H7 reconstructed price forecasts", xaxis_title=x_title, yaxis_title="Gold price (dataset units)", hovermode="x unified", legend_title="Series", height=470)
    return figure


def metric_by_horizon_figure(metrics: pd.DataFrame, metric: str, title: str) -> go.Figure:
    horizon = metrics.loc[metrics["Horizon"].ne("Overall")].copy()
    horizon["Horizon_Number"] = horizon["Horizon"].str.removeprefix("H").astype(int)
    return px.line(horizon, x="Horizon_Number", y=metric, color="Model", markers=True, title=title, color_discrete_map=COLORS, labels={"Horizon_Number": "Horizon", metric: title})


def rmse_figure(metrics: pd.DataFrame) -> go.Figure:
    horizon = metrics.loc[metrics["Horizon"].ne("Overall")].copy()
    horizon["Horizon_Number"] = horizon["Horizon"].str.removeprefix("H").astype(int)
    figure = px.line(horizon, x="Horizon_Number", y="Price_RMSE", color="Model", markers=True, color_discrete_map=COLORS, title="Price RMSE by horizon")
    persistence = horizon.drop_duplicates("Horizon_Number").sort_values("Horizon_Number")
    figure.add_trace(go.Scatter(x=persistence["Horizon_Number"], y=persistence["Persistence_Price_RMSE"], name="Persistence", mode="lines+markers", line={"color": "#6B7280", "dash": "dash", "width": 3}))
    figure.update_layout(xaxis_title="Horizon", yaxis_title="RMSE")
    return figure


def actual_vs_predicted_figure(predictions: pd.DataFrame, model: str, horizon: str) -> go.Figure:
    data = predictions.loc[(predictions["Model"].eq(model)) & (predictions["Horizon"].eq(horizon))].sort_values("Target_Date")
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data["Target_Date"], y=data["Actual_Price"], name="Actual", line={"color": "#111827"}))
    figure.add_trace(go.Scatter(x=data["Target_Date"], y=data["Predicted_Price"], name=f"{model} predicted", line={"color": COLORS.get(model)}))
    figure.add_trace(go.Scatter(x=data["Target_Date"], y=data["Persistence_Price"], name="Persistence", line={"color": "#6B7280", "dash": "dot"}))
    figure.update_layout(title=f"Actual vs predicted — {model} {horizon}", xaxis_title="Target date", yaxis_title="Gold price (dataset units)", height=430)
    return figure

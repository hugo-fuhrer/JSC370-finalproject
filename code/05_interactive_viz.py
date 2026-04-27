import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(DOCS, exist_ok=True)

panel = pd.read_csv(os.path.join(DATA, "panel.csv"), parse_dates=["date"])
weather = pd.read_csv(os.path.join(DATA, "weather.csv"),  parse_dates=["date"])
stations = pd.read_csv(os.path.join(DATA, "stations.csv"))
prices = pd.read_csv(os.path.join(DATA, "raw_prices.csv"), parse_dates=["Date"]).rename(columns={"Date": "date"})

TICKERS = sorted(panel["ticker"].unique().tolist())
SECTOR = panel.set_index("ticker")["sector"].to_dict()


# viz 1: map of weather stations + extreme-event intensity
weather["year"] = weather["date"].dt.year
yearly_extremes = (
    weather.groupby("year")[["extreme_cold", "extreme_heat",
                             "extreme_precip", "extreme_wind"]]
    .sum().reset_index()
)
# hover text
stations["hover"] = (
    "Station: " + stations["station_name"].astype(str)
    + "<br>Lat: "  + stations["lat"].round(2).astype(str)
    + "<br>Lon: "  + stations["lon"].round(2).astype(str)
    + "<br>Elev: " + stations["elevation"].round(0).astype(str) + " m"
)

fig1 = px.scatter_map(
    stations,
    lat="lat", lon="lon",
    hover_name="station_name",
    custom_data=["station_id", "elevation"],
    size_max=10, zoom=5, height=520,
    map_style="open-street-map",
    title="Ontario weather stations contributing to the daily climate panel",
    color_discrete_sequence=["#3b76b8"],
)
fig1.update_traces(marker=dict(size=10, opacity=0.75))
fig1.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
    title=dict(font=dict(size=15)),
)
fig1.write_html(os.path.join(DOCS, "viz1_map.html"),
                include_plotlyjs="cdn", full_html=True)
print("viz1 written")

# viz 2: dashboard of annual extreme counts with sector boxplot
extreme_long = (
    yearly_extremes.melt(id_vars="year", var_name="type", value_name="count")
)
extreme_long["type"] = (
    extreme_long["type"].str.replace("extreme_", "").str.title()
)

panel["Weather"] = panel["extreme_any"].map({0: "Normal", 1: "Extreme"})
panel["ndr_pct"] = panel["next_day_return"] * 100

fig2 = make_subplots(
    rows=1, cols=2, column_widths=[0.55, 0.45],
    subplot_titles=("Annual extreme event counts (Ontario)",
                    "Next-day returns by sector (extreme vs normal)"),
)
colors = {"Cold": "#4c8be0", "Heat": "#e74c3c",
          "Precip": "#27ae60", "Wind": "#8e44ad"}
for typ, sub in extreme_long.groupby("type"):
    fig2.add_trace(
        go.Bar(name=typ, x=sub["year"], y=sub["count"],
               marker_color=colors.get(typ, "#666666")),
        row=1, col=1,
    )

# boxplot side
for w, color in [("Normal", "#3b76b8"), ("Extreme", "#e74c3c")]:
    sub = panel[panel["Weather"] == w]
    fig2.add_trace(
        go.Box(y=sub["ndr_pct"], x=sub["sector"], name=w,
               marker_color=color, boxpoints="outliers", line_width=1),
        row=1, col=2,
    )

fig2.update_xaxes(title_text="Year", row=1, col=1)
fig2.update_yaxes(title_text="Number of days", row=1, col=1)
fig2.update_xaxes(title_text="Sector", row=1, col=2)
fig2.update_yaxes(title_text="Next-day return (%)",
                  range=[-4, 4], row=1, col=2)
fig2.update_layout(
    boxmode="group", barmode="group", height=520,
    title=dict(text="Extreme weather events and TSX next-day returns",
               font=dict(size=15)),
    legend=dict(orientation="h", yanchor="top", y=-0.12),
    margin=dict(l=40, r=20, t=70, b=40),
)
fig2.write_html(os.path.join(DOCS, "viz2_dashboard.html"),
                include_plotlyjs="cdn", full_html=True)
print("viz2 written")

# viz 3: 3D scatter of temp vs precip vs next-day return
sample = (
    panel.sample(n=min(8000, len(panel)), random_state=65)
    .copy()
)
sample["mean_temp"] = sample["mean_temp"].round(1)
sample["total_precip"] = sample["total_precip"].fillna(0).round(1)
sample["ndr_pct"] = sample["ndr_pct"].round(2)

fig3 = px.scatter_3d(
    sample,
    x="mean_temp", y="total_precip", z="ndr_pct",
    color="sector", symbol="Weather",
    hover_data=["ticker", "date"],
    title="3D view: temperature, precipitation, and next-day TSX return",
    template="plotly_white",
    opacity=0.55,
)
fig3.update_traces(marker=dict(size=2.5))
fig3.update_layout(
    height=600,
    scene=dict(
        xaxis_title="Mean temperature (C)",
        yaxis_title="Total precipitation (mm)",
        zaxis_title="Next-day log return (%)",
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    title=dict(font=dict(size=15)),
)
fig3.write_html(os.path.join(DOCS, "viz3_3d.html"),
                include_plotlyjs="cdn", full_html=True)
print("viz3 written")

print("viz files created")

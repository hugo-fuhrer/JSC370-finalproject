import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (
    ggplot, aes, geom_histogram, geom_vline, geom_col, geom_line,
    geom_boxplot, geom_hline, geom_smooth, geom_point, geom_tile,
    geom_text, scale_fill_manual, scale_fill_brewer, scale_fill_gradient2,
    scale_color_manual, scale_color_brewer, facet_wrap, coord_flip,
    coord_cartesian, labs, theme_bw, theme, element_text, annotate,
    position_dodge, scale_y_continuous, ggsave,
)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

weather = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"), parse_dates=["date"])
panel   = pd.read_csv(os.path.join(DATA_DIR, "panel.csv"),   parse_dates=["date"])

# fig 1: temp dist with extreme thresholds
cold_thresh = weather["mean_temp"].quantile(0.05)
heat_thresh = weather["mean_temp"].quantile(0.95)

p1 = (
    ggplot(weather, aes(x="mean_temp"))
    + geom_histogram(bins=60, fill="steelblue", color="white", size=0.2)
    + geom_vline(xintercept=cold_thresh, linetype="dashed", size=1)
    + geom_vline(xintercept=heat_thresh, linetype="dashed", size=1)
    + annotate("text", x=cold_thresh - 1, y=80,
               label=f"Cold ≤ {cold_thresh:.1f}°C", ha="right", size=9)
    + annotate("text", x=heat_thresh + 1, y=80,
               label=f"Heat ≥ {heat_thresh:.1f}°C", ha="left", size=9)
    + labs(title="Distribution of Ontario daily mean temperature, 2008–2024",
           x="Mean daily temperature (C)", y="Count (days)")
    + theme_bw()
    + theme(figure_size=(8, 4))
)
ggsave(p1, os.path.join(FIG_DIR, "fig01_temp_distribution.png"), dpi=160)

# fig 2: annual extreme events
extreme_long = (
    weather.assign(year=weather["date"].dt.year)
    .groupby("year")[["extreme_cold", "extreme_heat", "extreme_precip", "extreme_wind"]]
    .sum()
    .reset_index()
    .melt(id_vars="year", var_name="type", value_name="count")
)
extreme_long["type"] = extreme_long["type"].str.replace("extreme_", "").str.title()

p2 = (
    ggplot(extreme_long, aes(x="year", y="count", fill="type"))
    + geom_col(position="dodge", width=0.7)
    + scale_fill_manual(values=["#4c8be0", "#e74c3c", "#27ae60", "#8e44ad"])
    + labs(title="Annual counts of extreme weather days (Ontario, 2008–2024)",
           x="Year", y="Number of days", fill="Event type")
    + theme_bw()
    + theme(axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(9, 4))
)
ggsave(p2, os.path.join(FIG_DIR, "fig02_extreme_counts.png"), dpi=160)

# fig 3: cum log returns by sector 
TICKERS = panel["ticker"].unique().tolist()
SECTOR = panel.set_index("ticker")["sector"].to_dict()

cum_ret = (
    panel.pivot(index="date", columns="ticker", values="log_return")
    .fillna(0).cumsum()
    .reset_index()
    .melt(id_vars="date", var_name="ticker", value_name="cum_return")
)
cum_ret["sector"] = cum_ret["ticker"].map(SECTOR)

p3 = (
    ggplot(cum_ret, aes(x="date", y="cum_return", color="ticker", group="ticker"))
    + geom_line(size=0.7, alpha=0.85)
    + facet_wrap("~sector", ncol=4, scales="free_y")
    + labs(title="Cumulative log returns by sector, 2008–2024",
           x="Date", y="Cumulative log return")
    + theme_bw()
    + theme(axis_text_x=element_text(angle=45, hjust=1),
            legend_position="none",
            figure_size=(11, 6))
)
ggsave(p3, os.path.join(FIG_DIR, "fig03_cumulative_returns.png"), dpi=160)

# fig 4: next-day return dist by extreme/normal
p4 = (
    ggplot(panel.dropna(subset=["ndr_pct", "Weather"]),
           aes(x="ndr_pct", fill="Weather"))
    + geom_histogram(bins=60, alpha=0.6, position="identity",
                     color="white", size=0.1)
    + facet_wrap("~ticker", ncol=5, scales="free_y")
    + scale_fill_manual(values={"Normal": "steelblue", "Extreme": "#e74c3c"})
    + coord_cartesian(xlim=(-4, 4))
    + labs(title="Next-day log returns: extreme vs normal weather days",
           x="Next-day log return (%)", y="Count", fill="")
    + theme_bw()
    + theme(legend_position="top",
            figure_size=(11, 6))
)
ggsave(p4, os.path.join(FIG_DIR, "fig04_ndr_distribution.png"), dpi=160)

# fig 5: boxplot by sector
p5 = (
    ggplot(panel.dropna(subset=["ndr_pct", "Weather"]),
           aes(x="sector", y="ndr_pct", fill="Weather"))
    + geom_boxplot(outlier_alpha=0.15, outlier_size=0.4, width=0.65)
    + scale_fill_manual(values={"Normal": "steelblue", "Extreme": "#e74c3c"})
    + coord_cartesian(ylim=(-4, 4))
    + geom_hline(yintercept=0, linetype="dashed", color="grey", size=0.6)
    + labs(title="Next-day returns by sector: extreme vs normal weather",
           x="Sector", y="Next-day log return (%)", fill="")
    + theme_bw()
    + theme(legend_position="top",
            figure_size=(9, 4))
)
ggsave(p5, os.path.join(FIG_DIR, "fig05_box_by_sector.png"), dpi=160)

# fig 6: monthly mean temp vs avg next-day return
monthly = (
    panel.groupby(["year", "month"])
    .agg(mean_temp=("mean_temp", "mean"),
         ndr_pct=("ndr_pct", "mean"))
    .reset_index()
)
p6 = (
    ggplot(monthly, aes(x="mean_temp", y="ndr_pct"))
    + geom_point(alpha=0.5, size=1.5, color="steelblue")
    + geom_smooth(method="lm", color="#e74c3c", se=True)
    + labs(title="Monthly mean temperature vs average next-day return",
           x="Mean temperature (C)", y="Avg next-day return (%)")
    + theme_bw()
    + theme(figure_size=(7, 4))
)
ggsave(p6, os.path.join(FIG_DIR, "fig06_monthly_temp_vs_ndr.png"), dpi=160)

# fig 7: corr heatmap of weather features 
wcorr = (
    panel[["mean_temp", "total_precip", "max_gust", "temp_5d_anom",
           "lagged_return", "mkt_return", "next_day_return"]]
    .corr()
    .round(2)
    .reset_index()
    .melt(id_vars="index", var_name="var2", value_name="corr")
    .rename(columns={"index": "var1"})
)
p7 = (
    ggplot(wcorr, aes(x="var1", y="var2", fill="corr"))
    + geom_tile(color="white")
    + geom_text(aes(label="corr"), size=8)
    + scale_fill_gradient2(low="#3b76b8", mid="white", high="#c0392b",
                           midpoint=0, limits=[-0.6, 0.6])
    + labs(title="Correlation between weather and return features",
           x="", y="", fill="r")
    + theme_bw()
    + theme(axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(7, 5.5))
)
ggsave(p7, os.path.join(FIG_DIR, "fig07_correlation_heatmap.png"), dpi=160)

# table 1: weather summary
weather_summary = (
    weather[["mean_temp", "total_precip", "total_snow", "max_gust"]]
    .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    .T.round(2)
)
weather_summary.index = ["Mean Temp (C)", "Total Precip (mm)",
                         "Total Snow (cm)", "Max Gust (km/h)"]
weather_summary.to_csv(os.path.join(DATA_DIR, "table1_weather_summary.csv"))

# table 2: returns by ticker
ret_summary = (
    panel.groupby("ticker")["log_return"]
    .agg(["mean", "std", "count"])
    .round(5)
)
ret_summary["Mean Daily Ret (%)"] = (ret_summary["mean"] * 100).round(3)
ret_summary["Ann. Volatility (%)"] = (ret_summary["std"] * np.sqrt(252) * 100).round(2)
ret_summary["Sector"] = ret_summary.index.map(SECTOR)
ret_summary = ret_summary[["Sector", "Mean Daily Ret (%)",
                           "Ann. Volatility (%)", "count"]].rename(columns={"count": "N"})
ret_summary.to_csv(os.path.join(DATA_DIR, "table2_ret_summary.csv"))

# table 3: extreme vs normal
table3 = (
    panel.groupby(["ticker", "extreme_any"])["next_day_return"]
    .mean().mul(100).round(4)
    .unstack("extreme_any")
    .rename(columns={0: "Normal (%)", 1: "Extreme (%)"})
)
table3["Diff (bp)"] = ((table3["Extreme (%)"] - table3["Normal (%)"]) * 100).round(2)
table3["Sector"]    = table3.index.map(SECTOR)
table3 = table3[["Sector", "Normal (%)", "Extreme (%)", "Diff (bp)"]]
table3.to_csv(os.path.join(DATA_DIR, "table3_extreme_vs_normal.csv"))

print("EDA done. Figures saved to figures/")

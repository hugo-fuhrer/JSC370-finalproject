import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TICKERS = [
    "TD.TO", "RY.TO", "CM.TO", "BNS.TO", "BMO.TO",
    "IFC.TO", "MFC.TO", "POW.TO",
    "FTS.TO", "EMA.TO",
    "ENB.TO", "SU.TO",
    "BCE.TO", "REI-UN.TO",
    "L.TO",
]
SECTOR = {
    "TD.TO": "Bank", "RY.TO": "Bank", "CM.TO": "Bank",
    "BNS.TO": "Bank", "BMO.TO": "Bank",
    "IFC.TO": "Insurance", "MFC.TO": "Insurance", "POW.TO": "Insurance",
    "FTS.TO": "Utilities", "EMA.TO": "Utilities",
    "ENB.TO": "Energy", "SU.TO": "Energy",
    "BCE.TO": "Telecom",
    "REI-UN.TO": "REIT",
    "L.TO": "Retail",
}
BENCHMARK = "XIU.TO"

# get raw data
raw_weather = pd.read_csv(os.path.join(DATA_DIR, "raw_weather.csv"))
raw_weather["date"] = pd.to_datetime(raw_weather["date"])

raw_prices = pd.read_csv(os.path.join(DATA_DIR, "raw_prices.csv"),
                         index_col=0, parse_dates=True)
raw_prices.index.name = "date"

# missing values
def summarize_missing(df, name):
    rep = (
        df.isna().mean().mul(100).round(2).rename("pct_missing").to_frame()
        .assign(dataset=name, n_rows=len(df))
    )
    rep["variable"] = rep.index
    return rep[["dataset", "n_rows", "variable", "pct_missing"]].reset_index(drop=True)

q1 = summarize_missing(
    raw_weather[["mean_temp", "max_temp", "min_temp",
                 "total_precip", "total_snow", "max_gust"]],
    "raw_weather (per-station daily)",
)
q2 = summarize_missing(raw_prices, "raw_prices (daily close)")
quality = pd.concat([q1, q2], ignore_index=True)
quality.to_csv(os.path.join(DATA_DIR, "data_quality.csv"), index=False)

# create weather series
wcols = ["mean_temp", "max_temp", "min_temp", "total_precip", "total_snow", "max_gust"]
raw_weather[wcols] = raw_weather[wcols].apply(pd.to_numeric, errors="coerce")

weather = (
    raw_weather.groupby("date")
    .agg(
        mean_temp    = ("mean_temp",    "mean"),
        max_temp     = ("max_temp",     "mean"),
        min_temp     = ("min_temp",     "mean"),
        total_precip = ("total_precip", "max"),
        total_snow   = ("total_snow",   "max"),
        max_gust     = ("max_gust",     "max"),
        n_stations   = ("station_id",   "nunique"),
    )
    .reset_index()
)

# assume <=3 days are short outages/delays, drop remaining mean_temp NaN
weather[wcols] = weather[wcols].ffill(limit=3)
weather = weather.dropna(subset=["mean_temp"])

# extreme weather flags
def q(s, p): return s.quantile(p / 100)

weather["extreme_cold"] = (weather["mean_temp"] <= q(weather["mean_temp"], 5)).astype(int)
weather["extreme_heat"] = (weather["mean_temp"] >= q(weather["mean_temp"], 95)).astype(int)
weather["extreme_precip"] = (weather["total_precip"].fillna(0) >= q(weather["total_precip"].fillna(0), 95)).astype(int)
weather["extreme_wind"] = (weather["max_gust"].fillna(0) >= q(weather["max_gust"].fillna(0), 95)).astype(int)
weather["extreme_any"] = weather[["extreme_cold", "extreme_heat", "extreme_precip", "extreme_wind"]].max(axis=1)

# continuous features
weather["temp_z"] = (weather["mean_temp"] - weather["mean_temp"].mean()) / weather["mean_temp"].std()
weather["precip_z"] = (weather["total_precip"].fillna(0) - weather["total_precip"].mean()) / weather["total_precip"].std()
weather["wind_z"] = (weather["max_gust"].fillna(0) - weather["max_gust"].mean()) / weather["max_gust"].std()
# 5-day rolling temperature anomaly captures short cold/hot snaps
weather["temp_5d_anom"] = (weather["mean_temp"] - weather["mean_temp"].rolling(5, min_periods=1).mean())

weather.to_csv(os.path.join(DATA_DIR, "weather.csv"), index=False)

# stock returns
log_ret = np.log(raw_prices / raw_prices.shift(1)).dropna(how="all")
mkt_ret = log_ret[[BENCHMARK]].rename(columns={BENCHMARK: "mkt_return"})

panel = (
    log_ret[TICKERS]
    .reset_index()
    .melt(id_vars="date", var_name="ticker", value_name="log_return")
)
panel["sector"] = panel["ticker"].map(SECTOR)

# next-day returns
ndr = (
    log_ret[TICKERS].shift(-1)
    .reset_index()
    .melt(id_vars="date", var_name="ticker", value_name="next_day_return")
)

panel = (
    panel
    .merge(ndr, on=["date", "ticker"])
    .merge(mkt_ret.reset_index(), on="date")
    .merge(weather, on="date", how="inner")
    .dropna(subset=["log_return", "next_day_return", "mean_temp"])
)

panel = panel.sort_values(["ticker", "date"])

panel["lagged_return"] = panel.groupby("ticker")["log_return"].shift(1)
panel["lag2_return"] = panel.groupby("ticker")["log_return"].shift(2)
panel["roll5_return"] = panel.groupby("ticker")["log_return"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
panel["roll5_vol"] = panel.groupby("ticker")["log_return"].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)

panel["month"] = panel["date"].dt.month
panel["year"] = panel["date"].dt.year
panel["dow"] = panel["date"].dt.dayofweek
panel["Weather"] = panel["extreme_any"].map({0: "Normal", 1: "Extreme"})
panel["ndr_pct"] = panel["next_day_return"] * 100

# to classify increase/decrease
panel["next_day_up"] = (panel["next_day_return"] > 0).astype(int)

panel = panel.dropna(subset=["lagged_return", "lag2_return", "roll5_return", "roll5_vol"])
panel.to_csv(os.path.join(DATA_DIR, "panel.csv"), index=False)

print(f"Panel rows: {len(panel):,}")
print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
print(f"Tickers:    {panel['ticker'].nunique()}")
print(f"Extreme-any days in panel: {panel['extreme_any'].mean():.1%}")

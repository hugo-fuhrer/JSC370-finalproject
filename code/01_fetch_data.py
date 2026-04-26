import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# Config
BASE_URL     = "https://api.weather.gc.ca"
ONTARIO_BBOX = "-83.0,41.5,-74.5,46.5"   # expanded from midterm
START_DATE   = "2008-01-01"
END_DATE     = "2024-12-31"
BENCHMARK    = "XIU.TO"

# Expanded tickers
TICKERS = [
    # Banks
    "TD.TO", "RY.TO", "CM.TO", "BNS.TO", "BMO.TO",
    # Insurance / financial
    "IFC.TO", "MFC.TO", "POW.TO",
    # Utilities
    "FTS.TO", "EMA.TO",
    # Energy
    "ENB.TO", "SU.TO",
    # Telecom / Real estate
    "BCE.TO", "REI-UN.TO",
    # Retail
    "L.TO",
]

TICKER_SECTOR = {
    "TD.TO": "Bank", "RY.TO": "Bank", "CM.TO": "Bank",
    "BNS.TO": "Bank", "BMO.TO": "Bank",
    "IFC.TO": "Insurance", "MFC.TO": "Insurance", "POW.TO": "Insurance",
    "FTS.TO": "Utilities", "EMA.TO": "Utilities",
    "ENB.TO": "Energy", "SU.TO": "Energy",
    "BCE.TO": "Telecom",
    "REI-UN.TO": "REIT",
    "L.TO": "Retail",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# Need to first get the weather stations
def fetch_stations(bbox=ONTARIO_BBOX):
    resp = requests.get(
        f"{BASE_URL}/collections/climate-stations/items",
        params={"bbox": bbox, "f": "json", "limit": 1500},
        timeout=120,
    )
    resp.raise_for_status()
    feats = resp.json().get("features", [])
    rows = []
    for feat in feats:
        p = feat["properties"]
        last = p.get("DLY_LAST_DATE") or ""
        if last[:4] < "2015": # active after 2015
            continue
        rows.append({
            "station_id":   p.get("CLIMATE_IDENTIFIER"),
            "station_name": p.get("STATION_NAME"),
            "lat":          feat["geometry"]["coordinates"][1],
            "lon":          feat["geometry"]["coordinates"][0],
            "elevation":    p.get("ELEVATION"),
            "dly_last":     last,
        })
    return pd.DataFrame(rows)


# Once we have the station, we pull it's daily climate obs
def fetch_daily_climate(station_id):
    rows, offset = [], 0
    while True:
        params = {
            "CLIMATE_IDENTIFIER": station_id,
            "datetime": f"{START_DATE}/{END_DATE}",
            "f": "json",
            "limit": 10000,
            "offset": offset,
            "sortby": "LOCAL_DATE",
        }
        r = requests.get(
            f"{BASE_URL}/collections/climate-daily/items",
            params=params, timeout=120,
        )
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            break
        for feat in feats:
            p = feat["properties"]
            rows.append({
                "date":         p.get("LOCAL_DATE", "")[:10],
                "station_id":   station_id,
                "mean_temp":    p.get("MEAN_TEMPERATURE"),
                "max_temp":     p.get("MAX_TEMPERATURE"),
                "min_temp":     p.get("MIN_TEMPERATURE"),
                "total_precip": p.get("TOTAL_PRECIPITATION"),
                "total_snow":   p.get("TOTAL_SNOW"),
                "max_gust":     p.get("SPEED_MAX_GUST"),
            })
        if len(feats) < 10000:
            break
        offset += 10000
    return pd.DataFrame(rows)


def main():
    # weather
    print("Fetching stations")
    stations = fetch_stations()
    stations.to_csv(os.path.join(DATA_DIR, "stations.csv"), index=False)
    print(f"  -> {len(stations)} stations active since 2015")

    frames = []
    for i, sid in enumerate(stations["station_id"], 1):
        try:
            df = fetch_daily_climate(sid)
            frames.append(df)
        except Exception as e:
            print(f"  ! station {sid} failed: {e}")
        if i % 25 == 0:
            print(f"  fetched {i}/{len(stations)} stations")
        time.sleep(0.4)  # API rate

    raw_weather = pd.concat(frames, ignore_index=True)
    raw_weather["date"] = pd.to_datetime(raw_weather["date"], errors="coerce")
    raw_weather = raw_weather.dropna(subset=["date"])
    raw_weather.to_csv(os.path.join(DATA_DIR, "raw_weather.csv"), index=False)
    print(f"Weather rows: {len(raw_weather):,}")

    # stocks
    print("Downloading stock prices")
    px = yf.download(
        tickers=TICKERS + [BENCHMARK],
        start=START_DATE, end=END_DATE,
        auto_adjust=True, progress=False,
    )["Close"]
    px.index = pd.to_datetime(px.index)
    px.to_csv(os.path.join(DATA_DIR, "raw_prices.csv"))
    print(f"Prices: {px.shape}")


if __name__ == "__main__":
    main()
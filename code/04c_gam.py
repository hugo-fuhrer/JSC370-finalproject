import os, time, joblib, warnings
import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "_cache")

panel = pd.read_csv(os.path.join(DATA, "panel.csv"), parse_dates=["date"])
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

gam_features = ["mean_temp","total_precip","max_gust","temp_5d_anom",
                "lagged_return","roll5_vol","mkt_return","extreme_any"]
Xg = panel[gam_features]
y = panel["next_day_return"].astype(float)
mask = panel["date"] < "2022-01-01"
Xg_train, Xg_test = Xg[mask].values, Xg[~mask].values
y_train,  y_test  = y[mask].values,   y[~mask].values

t0 = time.time()
gam = LinearGAM(
    s(0, n_splines=10) + s(1, n_splines=8) + s(2, n_splines=8)
    + s(3, n_splines=8) + s(4, n_splines=10) + s(5, n_splines=8)
    + s(6, n_splines=10) + f(7)
).fit(Xg_train, y_train)
yhat = gam.predict(Xg_test)
gam_metrics = {
    "model": "GAM",
    "test_rmse": float(np.sqrt(mean_squared_error(y_test, yhat))),
    "test_mae":  float(mean_absolute_error(y_test, yhat)),
    "test_r2":   float(r2_score(y_test, yhat)),
    "oob_r2":    float("nan"),
}
print(f"GAM fit in {time.time()-t0:.0f}s: {gam_metrics}", flush=True)

joblib.dump({"gam": gam, "yhat_gam": yhat, "gam_metrics": gam_metrics,
             "gam_features": gam_features},
            os.path.join(CACHE, "stage3_gam.joblib"))

import os, time, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, roc_auc_score)
warnings.filterwarnings("ignore")
np.random.seed(370)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "_cache")
os.makedirs(CACHE, exist_ok=True)

panel = pd.read_csv(os.path.join(DATA, "panel.csv"), parse_dates=["date"])
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
panel = pd.concat([panel, pd.get_dummies(panel["sector"], prefix="sec", drop_first=True)], axis=1)

base_features = ["mean_temp","total_precip","max_gust","temp_5d_anom",
                 "extreme_cold","extreme_heat","extreme_precip","extreme_wind",
                 "lagged_return","lag2_return","roll5_return","roll5_vol",
                 "mkt_return","month","dow"]
sector_cols = [c for c in panel.columns if c.startswith("sec_")]
features = base_features + sector_cols

X = panel[features].copy()
y_reg = panel["next_day_return"].astype(float)
y_clf = panel["next_day_up"].astype(int)
# split by time instead of random to prevent data leakage given time series
mask = panel["date"] < "2022-01-01"
X_train, X_test = X[mask], X[~mask]
y_train_reg, y_test_reg = y_reg[mask], y_reg[~mask]
y_train_clf, y_test_clf = y_clf[mask], y_clf[~mask]
print(f"setup: train={len(X_train):,} test={len(X_test):,}", flush=True)

t0 = time.time()

# Regression
rf_reg = RandomForestRegressor(
    n_estimators=200, max_features="sqrt", min_samples_leaf=10,
    oob_score=True, random_state=65, n_jobs=-1,
)
rf_reg.fit(X_train, y_train_reg)
yhat_rf = rf_reg.predict(X_test)
print(f"RF reg fit: {time.time()-t0:.0f}s  oob_r2={rf_reg.oob_score_:.4f}", flush=True)

rf_metrics = {
    "model": "Random Forest",
    "test_rmse": float(np.sqrt(mean_squared_error(y_test_reg, yhat_rf))),
    "test_mae":  float(mean_absolute_error(y_test_reg, yhat_rf)),
    "test_r2":   float(r2_score(y_test_reg, yhat_rf)),
    "oob_r2":    float(rf_reg.oob_score_),
}

# Classification
t1 = time.time()
rf_clf = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", min_samples_leaf=10,
    oob_score=True, random_state=65, n_jobs=-1,
)
rf_clf.fit(X_train, y_train_clf)
prob_rf = rf_clf.predict_proba(X_test)[:, 1]
pred_rf = (prob_rf >= 0.5).astype(int)
print(f"RF clf fit: {time.time()-t1:.0f}s  oob={rf_clf.oob_score_:.4f}", flush=True)

rf_clf_metrics = {
    "model": "Random Forest",
    "test_accuracy": float(accuracy_score(y_test_clf, pred_rf)),
    "test_auc":      float(roc_auc_score(y_test_clf, prob_rf)),
}

# Save everything
joblib.dump({
    "rf_reg": rf_reg,
    "rf_clf": rf_clf,
    "yhat_rf": yhat_rf,
    "prob_rf": prob_rf,
    "pred_rf": pred_rf,
    "rf_metrics": rf_metrics,
    "rf_clf_metrics": rf_clf_metrics,
    "features": features,
}, os.path.join(CACHE, "stage1_rf.joblib"))

rf_imp = pd.DataFrame({"feature": features, "importance": rf_reg.feature_importances_}).sort_values("importance")
rf_imp.to_csv(os.path.join(DATA, "feature_importance_rf.csv"), index=False)

print(f"\nDone in {time.time()-t0:.0f}s")
print(rf_metrics)
print(rf_clf_metrics)

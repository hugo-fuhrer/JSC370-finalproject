import os, time, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, roc_auc_score)
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore")
np.random.seed(370)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "_cache")

panel = pd.read_csv(os.path.join(DATA, "panel.csv"), parse_dates=["date"])
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
panel = pd.concat([panel, pd.get_dummies(panel["sector"], prefix="sec", drop_first=True)], axis=1)

cached = joblib.load(os.path.join(CACHE, "stage1_rf.joblib"))
features = cached["features"]

X = panel[features].copy()
y_reg = panel["next_day_return"].astype(float)
y_clf = panel["next_day_up"].astype(int)
mask = panel["date"] < "2022-01-01"
X_train, X_test = X[mask], X[~mask]
y_train_reg, y_test_reg = y_reg[mask], y_reg[~mask]
y_train_clf, y_test_clf = y_clf[mask], y_clf[~mask]

t0 = time.time()

# Regressor with random search
xgb_base = XGBRegressor(n_estimators=120, random_state=370, n_jobs=-1)
param_dist = {
    "max_depth":     [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "reg_lambda":    [1, 10],
    "subsample":     [0.85, 1.0],
    "colsample_bytree": [0.85, 1.0],
}
rs = RandomizedSearchCV(
    estimator=xgb_base, param_distributions=param_dist,
    n_iter=5, cv=KFold(n_splits=3, shuffle=False),
    scoring="neg_root_mean_squared_error",
    random_state=65, n_jobs=-1, verbose=0,
)
rs.fit(X_train, y_train_reg)
best_params = rs.best_params_
print(f"XGB best params: {best_params}  [{time.time()-t0:.0f}s]", flush=True)

xgb_reg = XGBRegressor(n_estimators=300, **best_params, random_state=65, n_jobs=-1)
xgb_reg.fit(X_train, y_train_reg)
yhat_xgb = xgb_reg.predict(X_test)
xgb_metrics = {
    "model": "XGBoost",
    "test_rmse": float(np.sqrt(mean_squared_error(y_test_reg, yhat_xgb))),
    "test_mae":  float(mean_absolute_error(y_test_reg, yhat_xgb)),
    "test_r2":   float(r2_score(y_test_reg, yhat_xgb)),
    "oob_r2":    float("nan"),
}
print(f"XGB regression done [{time.time()-t0:.0f}s]: {xgb_metrics}", flush=True)

# Classifier
t1 = time.time()
xgb_clf = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    random_state=65, n_jobs=-1, eval_metric="logloss",
)
xgb_clf.fit(X_train, y_train_clf)
prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
pred_xgb = (prob_xgb >= 0.5).astype(int)
xgb_clf_metrics = {
    "model": "XGBoost",
    "test_accuracy": float(accuracy_score(y_test_clf, pred_xgb)),
    "test_auc":      float(roc_auc_score(y_test_clf, prob_xgb)),
}
print(f"XGB classification done [{time.time()-t1:.0f}s]: {xgb_clf_metrics}", flush=True)

# Feature importance
xgb_imp = pd.DataFrame({"feature": features, "importance": xgb_reg.feature_importances_}).sort_values("importance")
xgb_imp.to_csv(os.path.join(DATA, "feature_importance_xgb.csv"), index=False)

joblib.dump({
    "xgb_reg": xgb_reg, "xgb_clf": xgb_clf,
    "yhat_xgb": yhat_xgb, "prob_xgb": prob_xgb, "pred_xgb": pred_xgb,
    "xgb_metrics": xgb_metrics, "xgb_clf_metrics": xgb_clf_metrics,
    "best_params": best_params,
}, os.path.join(CACHE, "stage2_xgb.joblib"))
print(f"Total: {time.time()-t0:.0f}s")

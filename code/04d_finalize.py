import os, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, roc_curve)
from xgboost import XGBRegressor
from plotnine import (
    ggplot, aes, geom_col, geom_line, geom_point, geom_abline,
    coord_flip, labs, theme_bw, theme, scale_color_manual, ggsave,
)
warnings.filterwarnings("ignore")
np.random.seed(370)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "_cache")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

st1 = joblib.load(os.path.join(CACHE, "stage1_rf.joblib"))
st2 = joblib.load(os.path.join(CACHE, "stage2_xgb.joblib"))
st3 = joblib.load(os.path.join(CACHE, "stage3_gam.joblib"))

panel = pd.read_csv(os.path.join(DATA, "panel.csv"), parse_dates=["date"])
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
panel = pd.concat([panel, pd.get_dummies(panel["sector"], prefix="sec", drop_first=True)], axis=1)

features = st1["features"]
X = panel[features].copy()
y_reg = panel["next_day_return"].astype(float)
y_clf = panel["next_day_up"].astype(int)
mask = panel["date"] < "2022-01-01"
X_train, X_test = X[mask], X[~mask]
y_train_reg, y_test_reg = y_reg[mask], y_reg[~mask]
y_train_clf, y_test_clf = y_clf[mask], y_clf[~mask]

# naive baseline
naive_pred = np.full_like(y_test_reg, fill_value=y_train_reg.mean())
naive_metrics = {
    "model": "Naive (train mean)",
    "test_rmse": float(np.sqrt(mean_squared_error(y_test_reg, naive_pred))),
    "test_mae":  float(mean_absolute_error(y_test_reg, naive_pred)),
    "test_r2":   float(r2_score(y_test_reg, naive_pred)),
    "oob_r2":    float("nan"),
}
# regression results
reg_results = pd.DataFrame([
    naive_metrics, st1["rf_metrics"], st2["xgb_metrics"], st3["gam_metrics"],
])[["model","test_rmse","test_mae","test_r2","oob_r2"]]
reg_results[["test_rmse","test_mae"]] = reg_results[["test_rmse","test_mae"]].round(5)
reg_results[["test_r2","oob_r2"]] = reg_results[["test_r2","oob_r2"]].round(4)
reg_results.to_csv(os.path.join(DATA, "results_regression.csv"), index=False)
print("Regression results:\n", reg_results, flush=True)

# classification results
clf_results = pd.DataFrame([st1["rf_clf_metrics"], st2["xgb_clf_metrics"]]).round(4)
clf_results.to_csv(os.path.join(DATA, "results_classification.csv"), index=False)
print("\nClassification results:\n", clf_results, flush=True)

# fig 8: XGB feature importance
xgb_imp = pd.read_csv(os.path.join(DATA, "feature_importance_xgb.csv"))
top_xgb = xgb_imp.tail(15)
top_xgb["feature"] = pd.Categorical(top_xgb["feature"], categories=top_xgb["feature"])
ggsave(
    ggplot(top_xgb, aes(x="feature", y="importance"))
    + geom_col(fill="#3b76b8") + coord_flip()
    + labs(title="XGBoost feature importance (gain) — top 15",
           x="", y="Importance (mean gain)")
    + theme_bw() + theme(figure_size=(7, 5)),
    os.path.join(FIG, "fig08_xgb_importance.png"), dpi=160
)
print("fig08 saved", flush=True)

# fig 9: RF feature importance
rf_imp = pd.read_csv(os.path.join(DATA, "feature_importance_rf.csv"))
top_rf = rf_imp.tail(15)
top_rf["feature"] = pd.Categorical(top_rf["feature"], categories=top_rf["feature"])
ggsave(
    ggplot(top_rf, aes(x="feature", y="importance"))
    + geom_col(fill="#27ae60") + coord_flip()
    + labs(title="Random Forest feature importance — top 15",
           x="", y="Importance")
    + theme_bw() + theme(figure_size=(7, 5)),
    os.path.join(FIG, "fig09_rf_importance.png"), dpi=160
)
print("fig09 saved", flush=True)

# figure 10: Partial dependence XGB
pdp_features = ["mkt_return","lagged_return","roll5_vol",
                "mean_temp","extreme_cold","temp_5d_anom"]
sample_idx = np.random.RandomState(65).choice(len(X_train), size=1500, replace=False)
sample_X = X_train.iloc[sample_idx]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
PartialDependenceDisplay.from_estimator(
    st2["xgb_reg"], sample_X, pdp_features, ax=axes.ravel(), grid_resolution=20,
)
plt.suptitle("Partial dependence plots — tuned XGBoost", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig10_pdp.png"), dpi=160, bbox_inches="tight")
plt.close()
print("fig10 saved", flush=True)

# fig 11: GAM partial-effect
gam = st3["gam"]
gam_titles = ["Mean Temp","Total Precip","Max Gust","5-day Temp Anomaly",
              "Lagged Return","5d Volatility","Mkt Return","Extreme Day"]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
for i, (ax, name) in enumerate(zip(axes, gam_titles)):
    XX = gam.generate_X_grid(term=i)
    pdp = gam.partial_dependence(term=i, X=XX)
    _, ci = gam.partial_dependence(term=i, X=XX, width=0.95)
    if i < 7:
        ax.plot(XX[:, i], pdp, color="#c0392b", lw=2)
        ax.plot(XX[:, i], ci, color="#c0392b", ls="--", lw=1)
        ax.set_xlabel(name)
    else:
        ax.bar([0, 1], pdp[[0, -1]], color="#c0392b", alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal","Extreme"])
    ax.set_ylabel("Partial effect")
    ax.set_title(name)
plt.suptitle("GAM partial-effect curves (95% CIs)", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig11_gam_partial.png"), dpi=160, bbox_inches="tight")
plt.close()
print("fig11 saved", flush=True)

# fig 12: predicted vs actual (XGB)
yhat_xgb = st2["yhat_xgb"]
diag = pd.DataFrame({"actual": y_test_reg.values * 100,
                     "predicted": yhat_xgb * 100})
ggsave(
    ggplot(diag, aes(x="actual", y="predicted"))
    + geom_point(alpha=0.15, size=0.6, color="#3b76b8")
    + geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
    + labs(title="XGBoost predictions vs actual next-day returns (test set)",
           x="Actual next-day log return (%)",
           y="Predicted next-day log return (%)")
    + theme_bw() + theme(figure_size=(6, 6)),
    os.path.join(FIG, "fig12_pred_vs_actual.png"), dpi=160
)
print("fig12 saved", flush=True)

# fig 13: ROC
fpr_rf,  tpr_rf,  _ = roc_curve(y_test_clf, st1["prob_rf"])
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_clf, st2["prob_xgb"])
roc_df = pd.concat([
    pd.DataFrame({"fpr": fpr_rf,  "tpr": tpr_rf,  "model": "Random Forest"}),
    pd.DataFrame({"fpr": fpr_xgb, "tpr": tpr_xgb, "model": "XGBoost"}),
], ignore_index=True)
ggsave(
    ggplot(roc_df, aes(x="fpr", y="tpr", color="model"))
    + geom_line(size=0.9)
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="grey")
    + scale_color_manual(values={"Random Forest": "#27ae60",
                                 "XGBoost":      "#3b76b8"})
    + labs(title="ROC curves — predicting next-day up/down",
           x="False Positive Rate", y="True Positive Rate", color="")
    + theme_bw() + theme(figure_size=(6, 5)),
    os.path.join(FIG, "fig13_roc.png"), dpi=160
)
print("fig13 saved", flush=True)

# fig 14: Bias-variance XGB
print("computing bias-variance curve", flush=True)
bp = st2["best_params"]
xgb_bv = XGBRegressor(n_estimators=200, max_depth=bp["max_depth"],
                      learning_rate=bp["learning_rate"],
                      random_state=65, n_jobs=-1)
xgb_bv.fit(X_train, y_train_reg)
n_rounds = list(range(20, 201, 20))
train_err, test_err = [], []
for n in n_rounds:
    train_err.append(np.sqrt(mean_squared_error(
        y_train_reg, xgb_bv.predict(X_train, iteration_range=(0, n)))))
    test_err.append(np.sqrt(mean_squared_error(
        y_test_reg,  xgb_bv.predict(X_test,  iteration_range=(0, n)))))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n_rounds, train_err, "o-", label="Training RMSE", markersize=4, color="#27ae60")
ax.plot(n_rounds, test_err,  "s-", label="Test RMSE",     markersize=4, color="#c0392b")
ax.set_xlabel("Boosting rounds (n_estimators)")
ax.set_ylabel("RMSE (log return units)")
ax.set_title("Bias-variance trade-off — XGBoost")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig14_bias_variance.png"), dpi=160, bbox_inches="tight")
plt.close()
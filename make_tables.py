"""Turn the metric CSVs from pipeline_v2 into LaTeX table fragments that
final_report/report.tex \\input{}s."""
import os, pandas as pd
T = "/Users/abdulazizal-haidary/development/Machine Learning/Project/final_report/tables"

reg = pd.read_csv(os.path.join(T, "metrics_regression.csv"))
cls = pd.read_csv(os.path.join(T, "metrics_classification.csv"))
per = pd.read_csv(os.path.join(T, "dataset_per_flight.csv"))

def _best(df, metric, higher_better):
    idx = df.groupby("features")[metric].idxmax() if higher_better else df.groupby("features")[metric].idxmin()
    return set(idx.values)

# ------------------------------------------------------------------ regression
best_rmse = _best(reg, "RMSE", higher_better=False)
with open(os.path.join(T, "metrics_regression.tex"), "w") as fh:
    fh.write(r"""\begin{table}[t]
\centering
\caption{Regression --- 5-fold GroupKFold CV. Best RMSE per feature set in \textbf{bold}.}
\label{tab:reg_final}
\begin{tabular}{llrrr}
\toprule
Features & Model & MAE (m) & RMSE (m) & $R^2$ \\
\midrule
""")
    for fs in ["WEATHER", "ONBOARD", "ALL"]:
        for _, r in reg[reg.features == fs].iterrows():
            bold = r.name in best_rmse
            fmt = r"\textbf{{{:.2f}}}" if bold else r"{:.2f}"
            fh.write(f"{fs} & {r.model} & "
                     + fmt.format(r.MAE) + " & "
                     + fmt.format(r.RMSE) + " & "
                     + (r"\textbf{{{:+.3f}}}" if bold else r"{:+.3f}").format(r.R2)
                     + r" \\" + "\n")
        fh.write(r"\midrule" + "\n")
    fh.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# --------------------------------------------------------------- classification
best_f1 = _best(cls, "F1", higher_better=True)
with open(os.path.join(T, "metrics_classification.tex"), "w") as fh:
    fh.write(r"""\begin{table}[t]
\centering
\caption{Classification (deviation $\ge$ 75th pct) --- 5-fold GroupKFold CV. Best F1 per feature set in \textbf{bold}.}
\label{tab:cls_final}
\begin{tabular}{llrrrr}
\toprule
Features & Model & Acc & P & R & F1 \\
\midrule
""")
    for fs in ["WEATHER", "ONBOARD", "ALL"]:
        for _, r in cls[cls.features == fs].iterrows():
            bold = r.name in best_f1
            fmt = r"\textbf{{{:.3f}}}" if bold else r"{:.3f}"
            fh.write(f"{fs} & {r.model} & "
                     + fmt.format(r.Accuracy) + " & "
                     + fmt.format(r.Precision) + " & "
                     + fmt.format(r.Recall) + " & "
                     + fmt.format(r.F1) + r" \\" + "\n")
        fh.write(r"\midrule" + "\n")
    fh.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ----------------------------------------------------------------- dataset
summary = {
    "Flights": len(per),
    "Samples (after 1\,Hz)": int(per.N.sum()),
    "Mean deviation (m)": f"{(per.N*per.mean_dev).sum()/per.N.sum():.2f}",
    "Median flight duration (min)": f"{per.duration_min.median():.1f}",
    "75th-pct threshold (m)": "populated from pipeline",
}
with open(os.path.join(T, "dataset_summary.tex"), "w") as fh:
    fh.write(r"""\begin{table}[t]
\centering
\caption{Dataset summary (15 ArduPilot BIN-logged flights, Nov.~7--26, 2025).}
\label{tab:data}
\begin{tabular}{lr}
\toprule
Quantity & Value \\
\midrule
""")
    for k, v in summary.items():
        fh.write(f"{k} & {v} \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

print("wrote", os.listdir(T))

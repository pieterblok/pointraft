import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
resultsfolder = os.path.join(project_root, "results")

df_lr = pd.read_csv(os.path.join(resultsfolder, "linear_regression.csv"))
df_pr = pd.read_csv(os.path.join(resultsfolder, "pointraft.csv"))

gt_lr, pred_lr = df_lr["gt"].values, df_lr["pred"].values
gt_pr, pred_pr = df_pr["gt"].values, df_pr["pred"].values

err_lr = np.abs(pred_lr - gt_lr)
err_pr = np.abs(pred_pr - gt_pr)

def empirical_cdf(errors):
    errors_sorted = np.sort(errors)
    cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    return errors_sorted, cdf

x_lr, y_lr = empirical_cdf(err_lr)
x_pr, y_pr = empirical_cdf(err_pr)


plt.figure(figsize=(12, 8))

line_lr, = plt.plot(
    x_lr, y_lr,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Linear regression"
)

line_pr, = plt.plot(
    x_pr, y_pr,
    color="blue",
    linestyle="-",
    linewidth=2,
    label="PointRAFT"
)

plt.xlabel("Absolute weight error [g]", fontsize=16)
plt.ylabel("Fraction of samples", fontsize=16)

plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xticks(np.arange(0, 101, 10), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)


vertical_lines = [10, 20, 30, 40, 50]
for x_val in vertical_lines:    
    y_lr_val = y_lr[np.searchsorted(x_lr, x_val)]
    y_pr_val = y_pr[np.searchsorted(x_pr, x_val)]

    plt.axvline(x=x_val, ymin=0, ymax=y_pr_val, color="black", linestyle=":")
    
    plt.text(x_val + 0.5, y_lr_val - 0.025, f"{y_lr_val:.2f}", color="red", fontsize=14, va="bottom")
    plt.text(x_val + 0.5, y_pr_val - 0.025, f"{y_pr_val:.2f}", color="blue", fontsize=14, va="bottom")

plt.tight_layout()
plt.show()
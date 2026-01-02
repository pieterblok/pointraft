import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
resultsfolder = os.path.join(project_root, "results")

df_lr = pd.read_csv(os.path.join(resultsfolder, "linear_regression.csv"))
df_pr = pd.read_csv(os.path.join(resultsfolder, "pointraft.csv"))

gt_lr, pred_lr = df_lr["gt"], df_lr["pred"]
gt_pr, pred_pr = df_pr["gt"], df_pr["pred"]

r2_lr = r2_score(gt_lr, pred_lr)
r2_pr = r2_score(gt_pr, pred_pr)

bias_lr = (pred_lr - gt_lr).mean()
bias_pr = (pred_pr - gt_pr).mean()
print(f"Linear regression bias: {bias_lr:.2f} g")
print(f"PointRAFT bias: {bias_pr:.2f} g")

plt.figure(figsize=(12, 8))

plt.scatter(
    gt_lr, pred_lr,
    color="red",
    marker="x",
    label=f"Linear regression (R² = {r2_lr:.2f}, bias = +{bias_lr:.2f} g)"
)

plt.scatter(
    gt_pr, pred_pr,
    color="blue",
    marker="o",
    label=f"PointRAFT (R² = {r2_pr:.2f}, bias = +{bias_pr:.2f} g)"
)

plt.plot(
    [0, 600],
    [0, 600],
    linestyle=":",
    color="black",
    label="x = y"
)

plt.xlabel("Ground truth weight [g]", fontsize=16)
plt.ylabel("Predicted weight [g]", fontsize=16)
plt.legend(fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(0, 600)
plt.ylim(0, 600)

plt.tight_layout()
plt.show()
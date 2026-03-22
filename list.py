import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
df = pd.read_csv("runs/improved/results.csv")

# 画Loss曲线
plt.figure(figsize=(8,6))

plt.plot(df['epoch'], df['train/box_loss'], label='train box loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='train cls loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='train dfl loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline_training Loss Curve")
plt.legend()
plt.grid()

plt.savefig("Baseline_loss_curve.png", dpi=300)
plt.show()
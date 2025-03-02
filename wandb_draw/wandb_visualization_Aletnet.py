import wandb
import pandas as pd

# 初始化 W&B 运行
wandb.init(project="Alexnet", name="csv_line_chart")

# 读取 CSV
df = pd.read_csv("/gpfs/workdir/islamm/wandb_draw/training_metrics1.csv")

# 记录每一行的数据
for i in range(len(df)):
    wandb.log({"epoch": df["epoch"][i], "nmi": df["nmi"][i], "average_loss ": df["average_loss"][i]})

# 结束 W&B 运行
wandb.finish()

import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the dataset from the CSV file
file_path = 'training_metrics.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Ensure the column names are consistent
data.rename(columns={"epoch": "Epoch", "nmi": "NMI", "average_loss": "Average_Loss"}, inplace=True)

# Plotting the results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Average Loss
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Average Loss", color="tab:blue")
ax1.plot(data["Epoch"], data["Average_Loss"], color="tab:blue", marker="o", label="Average Loss")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Creating a twin y-axis for NMI
ax2 = ax1.twinx()
ax2.set_ylabel("NMI", color="tab:green")
ax2.plot(data["Epoch"], data["NMI"], color="tab:green", marker="s", label="NMI")
ax2.tick_params(axis="y", labelcolor="tab:green")

# Adding a title and legend
fig.suptitle("Training Metrics Over Epochs", fontsize=16)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Define the output path
output_dir = '/gpfs/workdir/islamm/output/'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
output_file = os.path.join(output_dir, 'training_metrics_plot.png')

# Save the plot
plt.tight_layout()
plt.savefig(output_file, format='png')

# Inform the user
print(f"Plot saved to {output_file}")

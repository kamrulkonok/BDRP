
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 加载特征数据
features = np.load("/gpfs/workdir/islamm/alexnet_features.npy")
# 准备标签数据
#labels = pd.read_csv("/gpfs/workdir/islamm/cluster_assignments.csv").values.squeeze()
encoded_labels = pd.read_csv("/gpfs/workdir/islamm/output_with_labelID.csv").values.squeeze()

# ---------------------------
# Load cluster assignments from the CSV file
# ---------------------------

cluster_assignments = pd.read_csv("/gpfs/workdir/islamm/cluster_assignments.csv").values.squeeze()

# ---------------------------
# Compute Normalized Mutual Information (NMI) score
# ---------------------------
nmi_score = normalized_mutual_info_score(encoded_labels, cluster_assignments, average_method='geometric')
print("NMI score (cluster quality): {:.4f}".format(nmi_score))

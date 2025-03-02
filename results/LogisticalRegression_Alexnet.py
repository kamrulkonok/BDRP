import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载特征数据
features = np.load("/gpfs/workdir/islamm/alexnet_features.npy")
# 准备标签数据
#labels = pd.read_csv("/gpfs/workdir/islamm/cluster_assignments.csv").values.squeeze()
labels = pd.read_csv("/gpfs/workdir/islamm/output_with_labelID.csv").values.squeeze()

# 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 归一化：使用StandardScaler对数据进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建逻辑回归模型，并显式使用L2正则化z
logreg = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)  # C是正则化强度的反向，越小正则化越强
# 训练模型
logreg.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = logreg.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"logistic regression accuracy: {accuracy:.4f}")

#(deepcluster) [islamm@ruche02 islamm]$ python LogisticalRegression.py 
# #logistic regression accuracy: 0.5640
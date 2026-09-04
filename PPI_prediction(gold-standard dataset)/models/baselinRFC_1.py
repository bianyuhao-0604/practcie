import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import time

# ==================== 文件路径（请根据实际情况修改） ====================
train_path = "D:/PPI_prediction_study-master/data/mean_pooled_features/ppi_train_mean_pooled.npz"
val_path   = "D:/PPI_prediction_study-master/data/mean_pooled_features/ppi_val_mean_pooled.npz"
test_path  = "D:/PPI_prediction_study-master/data/mean_pooled_features/ppi_test_mean_pooled.npz"

# ==================== 加载数据 ====================
print("加载数据...")
train_data = np.load(train_path)
val_data = np.load(val_path)
test_data = np.load(test_path)

X_train, y_train = train_data['X'], train_data['y']
X_val,   y_val   = val_data['X'], val_data['y']
X_test,  y_test  = test_data['X'], test_data['y']

print(f"训练集 X 形状: {X_train.shape}, y 分布: {np.bincount(y_train)}")
print(f"验证集 X 形状: {X_val.shape}, y 分布: {np.bincount(y_val)}")
print(f"测试集 X 形状: {X_test.shape}, y 分布: {np.bincount(y_test)}")

# ==================== 训练随机森林 ====================
print("训练随机森林...")
start_time = time.time()
rfc = RandomForestClassifier(
    n_estimators=100,           # 树的数量
    random_state=42,            # 随机种子，保证可复现   # 处理类别不平衡
    n_jobs=-1                   # 使用所有 CPU 核心
)
rfc.fit(X_train, y_train)

# ==================== 验证集评估 ====================
y_val_pred = rfc.predict(X_val)
y_val_prob = rfc.predict_proba(X_val)[:, 1]
print("\n--- 验证集性能 ---")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"AUROC: {roc_auc_score(y_val, y_val_prob):.4f}")

# ==================== 测试集评估 ====================
y_test_pred = rfc.predict(X_test)
y_test_prob = rfc.predict_proba(X_test)[:, 1]
print("\n--- 测试集性能 ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"AUROC: {roc_auc_score(y_test, y_test_prob):.4f}")

# ==================== 混淆矩阵 ====================
cm = confusion_matrix(y_test, y_test_pred)
print("\n混淆矩阵:")
print(cm)

print(f"\n训练总耗时: {time.time() - start_time:.2f} 秒")
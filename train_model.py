import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# 讀取資料
csv_path = "casme_features.csv"
df = pd.read_csv(csv_path)

# 特徵與標籤
X = df[["mlen", "ewid", "abs_mlen", "abs_ewid", "AU4", "AU20", "AU23"]]
y = df["label"]

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練 Logistic Regression 模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("==== 模型評估結果 ====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\n完整報告:\n")
print(classification_report(y_test, y_pred))

# 儲存模型
joblib.dump(model, "lie_detector_model.pkl")
print("模型已儲存為 lie_detector_model.pkl")
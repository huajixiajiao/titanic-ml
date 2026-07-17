"""
泰坦尼克号生存预测 - 逻辑回归 (Logistic Regression)

这是整个项目的第一步：用最简单的线性模型建立基线 (baseline)。
逻辑回归虽然名字里有"回归"，但它实际上是一个分类模型。

核心原理：
  1. 线性组合：z = w1*x1 + w2*x2 + ... + wn*xn + b
  2. Sigmoid 压缩：p = 1 / (1 + e^(-z))，把分数压缩成 0~1 的概率
  3. 阈值判定：p >= 0.5 预测为生还(1)，否则预测为遇难(0)

运行方式：
  python src/01_logistic_regression.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 路径配置（让脚本从任何目录运行都能找到数据文件）
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submissions')

# ============================================================
# 第 1 步：加载数据
# ============================================================
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("=" * 50)
print("逻辑回归 - 泰坦尼克号生存预测")
print("=" * 50)
print(f"训练集大小: {train.shape}")  # (891, 12)
print(f"测试集大小: {test.shape}")   # (418, 11)

# ============================================================
# 第 2 步：数据预处理
# ============================================================
# --- 缺失值填充 ---
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Fare'] = train['Fare'].fillna(train['Fare'].median())

# --- 类别编码 ---
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = pd.get_dummies(train, columns=['Embarked'], prefix='Emb')

# --- 特征工程 ---
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)

# --- 丢弃无用列 ---
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# ============================================================
# 第 3 步：选取特征 & 拆分数据
# ============================================================
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'Emb_C', 'Emb_Q', 'Emb_S']

X = train[features]
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 第 4 步：创建模型 & 训练
# ============================================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# 第 5 步：评估
# ============================================================
val_pred = model.predict(X_val)

print(f"\n准确率: {accuracy_score(y_val, val_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_val, val_pred))
print("混淆矩阵:")
print(confusion_matrix(y_val, val_pred))

# ============================================================
# 第 6 步：查看模型学到的权重（逻辑回归的独有优势）
# ============================================================
weights = pd.DataFrame({
    '特征': features,
    '权重 w': model.coef_[0]
}).sort_values('权重 w', ascending=False)
print("\n模型学到的权重:")
print(weights.to_string(index=False))

# ============================================================
# 第 7 步：对 test.csv 预测 & 生成提交文件
# ============================================================
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test = pd.get_dummies(test, columns=['Embarked'], prefix='Emb')
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

X_test = test[features]
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'Passenger': range(1, len(test_pred) + 1),
    'Survived': test_pred
})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_logistic_regression.csv'), index=False)
print(f"\n提交文件已生成: submissions/submission_logistic_regression.csv")

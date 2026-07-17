"""
泰坦尼克号生存预测 - 随机森林 (Random Forest)

在逻辑回归基线之上，换用随机森林模型。
随机森林的核心思想：训练 100 棵决策树，每棵树独立预测，最后少数服从多数。

与逻辑回归的区别：
  - 能自动捕捉非线性特征交互（如"女性 AND 一等舱"的组合效应）
  - 能自动处理相关特征（Fare 和 Pclass 不会互相"抢"权重）
  - 输出特征重要性而非权重方向

运行方式：
  python src/02_random_forest.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 路径配置
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
print("随机森林 - 泰坦尼克号生存预测")
print("=" * 50)
print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

# ============================================================
# 第 2 步：数据预处理（与逻辑回归完全一致）
# ============================================================
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = pd.get_dummies(train, columns=['Embarked'], prefix='Emb')
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
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
# 第 4 步：创建随机森林模型 & 训练
# ============================================================
# max_depth=5 是初始默认值，后续在 03 脚本中会通过调参优化为 8
model = RandomForestClassifier(
    n_estimators=100,   # 100 棵决策树
    max_depth=5,        # 每棵树最多 5 层（防止过拟合）
    random_state=42,    # 固定随机种子，保证可复现
    n_jobs=-1           # 使用所有 CPU 核心并行训练
)
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
# 第 6 步：特征重要性（随机森林独有，替代逻辑回归的权重）
# ============================================================
importance = pd.DataFrame({
    '特征': features,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)
print("\n特征重要性:")
print(importance.to_string(index=False))

# ============================================================
# 第 7 步：交叉验证（比单次拆分更可靠）
# ============================================================
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n5折交叉验证: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"  各折分数: {[f'{s:.4f}' for s in scores]}")

# ============================================================
# 第 8 步：对 test.csv 预测 & 生成提交文件
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
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_random_forest.csv'), index=False)
print(f"\n提交文件已生成: submissions/submission_random_forest.csv")

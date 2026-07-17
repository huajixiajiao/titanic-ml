"""
泰坦尼克号生存预测 - XGBoost (梯度提升树)

XGBoost 是 Kaggle 表格数据的标配模型。与随机森林的不同之处：
  - 随机森林：100 棵树各练各的，最后投票
  - XGBoost：树 1 先练，树 2 专门纠正树 1 的错误，树 3 纠正树 2 的错误...接力纠错

关键参数：
  - learning_rate：每棵树的"步幅"，越小越稳但需要更多树
  - n_estimators：树的数量，与 learning_rate 搭配使用
  - max_depth：每棵树的深度，比随机森林浅（因为靠接力取胜）

运行方式：
  python src/04_xgboost.py

注意：需要先安装 xgboost：pip install xgboost
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submissions')

# ============================================================
# 第 1 步：加载数据（与之前完全一样）
# ============================================================
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("=" * 50)
print("XGBoost - 泰坦尼克号生存预测")
print("=" * 50)
print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

# ============================================================
# 第 2 步：数据预处理（与之前完全一样）
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
# 第 4 步：创建 XGBoost 模型 & 训练
# ============================================================
# learning_rate=0.1 + n_estimators=300 是经典的"小步多走"搭配
model = XGBClassifier(
    n_estimators=300,        # 300 棵树（配合 0.1 学习率）
    learning_rate=0.1,       # 每棵树贡献 10%，慢慢逼近最优解
    max_depth=4,             # 比随机森林浅（XGBoost 靠接力取胜）
    subsample=0.8,           # 每棵树用 80% 的训练数据（防过拟合）
    colsample_bytree=0.8,    # 每棵树用 80% 的特征（增加树之间差异）
    random_state=42,
    eval_metric='logloss',   # 评估指标：对数损失
    use_label_encoder=False  # 禁用旧版编码器警告
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

# 特征重要性
importance = pd.DataFrame({
    '特征': features,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)
print("\n特征重要性:")
print(importance.to_string(index=False))

# 交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n5折交叉验证: {scores.mean():.4f} +/- {scores.std():.4f}")

# ============================================================
# 第 6 步：对 test.csv 预测 & 生成提交文件
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
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_xgboost.csv'), index=False)
print(f"\n提交文件已生成: submissions/submission_xgboost.csv")

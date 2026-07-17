"""
泰坦尼克号生存预测 - 随机森林超参数调优

在 02 脚本的基础上，通过交叉验证系统搜索最优 max_depth。
这是整个项目中最重要的一课：单次验证会被"运气"骗，交叉验证才可靠。

关键发现：
  - 单次验证选出 depth=2（0.8156），但 CV 只有 0.7947（最不稳定）
  - 交叉验证选出 depth=8（CV=0.8272），比默认 depth=5 提升 1.8%
  - 训练集准确率随 depth 增大一路升到 98%，但验证集先涨后跌 = 过拟合

运行方式：
  python src/03_rf_hyperparameter_tuning.py
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
# 第 1~3 步：数据加载 + 预处理 + 特征选择
# ============================================================
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = pd.get_dummies(train, columns=['Embarked'], prefix='Emb')
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'Emb_C', 'Emb_Q', 'Emb_S']
X = train[features]
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 第 4 步：调参循环 - 用交叉验证选最佳 max_depth
# ============================================================
print("=" * 70)
print("随机森林超参数调优：搜索最优 max_depth")
print("=" * 70)
print(f"{'depth':<8} {'单次验证':<14} {'CV均值':<14} {'CV标准差':<12} {'训练集':<10}")
print("-" * 70)

best_cv_score = 0
best_depth = None

for depth in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, None]:
    model = RandomForestClassifier(
        n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1
    )
    # 单次验证
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    train_acc = accuracy_score(y_train, model.predict(X_train))
    # 5 折交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    depth_label = str(depth) if depth is not None else "None"
    print(f"{depth_label:<8} {val_acc:<14.4f} {cv_scores.mean():<14.4f} "
          f"{cv_scores.std():<12.4f} {train_acc:<10.4f}")

    # 以交叉验证分数为准选最优
    if cv_scores.mean() > best_cv_score:
        best_cv_score = cv_scores.mean()
        best_depth = depth

print("-" * 70)
print(f"\n交叉验证最优 depth = {best_depth}，CV = {best_cv_score:.4f}")

# ============================================================
# 第 5 步：用最佳 depth 训练最终模型 & 评估
# ============================================================
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=best_depth, random_state=42, n_jobs=-1
)
final_model.fit(X_train, y_train)
val_pred = final_model.predict(X_val)

print(f"\n最终模型 (max_depth={best_depth}):")
print(f"准确率: {accuracy_score(y_val, val_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_val, val_pred))
print("混淆矩阵:")
print(confusion_matrix(y_val, val_pred))

# 特征重要性
importance = pd.DataFrame({
    '特征': features,
    '重要性': final_model.feature_importances_
}).sort_values('重要性', ascending=False)
print("\n特征重要性:")
print(importance.to_string(index=False))

# 最终交叉验证
final_cv = cross_val_score(final_model, X, y, cv=5, scoring='accuracy')
print(f"\n最终 5 折交叉验证: {final_cv.mean():.4f} +/- {final_cv.std():.4f}")

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
test_pred = final_model.predict(X_test)

submission = pd.DataFrame({
    'Passenger': range(1, len(test_pred) + 1),
    'Survived': test_pred
})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_rf_tuned.csv'), index=False)
print(f"\n提交文件已生成: submissions/submission_rf_tuned.csv")

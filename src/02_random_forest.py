import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submissions')

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("=" * 50)
print("随机森林 - 泰坦尼克号生存预测")
print("=" * 50)
print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

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

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)

print(f"\n准确率: {accuracy_score(y_val, val_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_val, val_pred))
print("混淆矩阵:")
print(confusion_matrix(y_val, val_pred))

importance = pd.DataFrame({
    '特征': features,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)
print("\n特征重要性:")
print(importance.to_string(index=False))

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n5折交叉验证: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"  各折分数: {[f'{s:.4f}' for s in scores]}")

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

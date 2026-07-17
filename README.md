# Titanic 生存预测

基于 Kaggle 泰坦尼克号数据集，用逻辑回归、随机森林、XGBoost 三种模型完成生存预测，包含完整的预处理、调参和模型对比。

## 怎么跑

```bash
git clone https://github.com/huajixiajiao/titanic-ml.git
cd titanic-ml
pip install -r requirements.txt

python src/01_logistic_regression.py
python src/02_random_forest.py
python src/03_rf_hyperparameter_tuning.py
python src/04_xgboost.py
```

每个脚本跑完会打印准确率、分类报告和混淆矩阵，预测结果写到 `submissions/` 下。

## 目录结构

```
titanic-ml/
├── data/
│   ├── train.csv                 # 训练集 891×12
│   ├── test.csv                  # 测试集 418×11
│   └── gender_submission.csv     # 提交样例
├── src/
│   ├── 01_logistic_regression.py
│   ├── 02_random_forest.py
│   ├── 03_rf_hyperparameter_tuning.py
│   └── 04_xgboost.py
├── submissions/
├── requirements.txt
└── README.md
```

## 数据

训练集 891 条，测试集 418 条。主要用到的字段：

| 字段 | 说明 | 备注 |
|------|------|------|
| Pclass | 舱位等级 1/2/3 | 有序 |
| Sex | 性别 | 编码为 0/1 |
| Age | 年龄 | 缺失 177 个，用中位数补 |
| SibSp / Parch | 同船家属数 | 合成 FamilySize |
| Fare | 票价 | 缺失少量，中位数补 |
| Embarked | 登船港口 C/Q/S | One-Hot 编码 |
| Cabin | 客舱号 | 缺失 77%，直接丢掉 |

Name、Ticket、PassengerId 没用上。另外从 SibSp 和 Parch 拼了一个 `FamilySize = SibSp + Parch + 1`，再派生出 `IsAlone` 标记是否独自登船。

## 模型和结果

四个脚本对应一条递进路线：线性模型 → 树模型集成 → 调参 → 梯度提升。

| 模型 | 5 折 CV | 标准差 |
|------|:-------:|:------:|
| 逻辑回归 | ~0.80 | — |
| 随机森林 (depth=5) | 0.8092 | 0.0201 |
| 随机森林 (depth=8, 调参后) | **0.8272** | 0.0270 |
| XGBoost | 0.8249 | **0.0228** |

几个值得记的点：

- **性别是压倒性的第一特征。** 三个模型都把 Sex 排在最重要的位置，和"妇女儿童优先"的历史情况一致。
- **调参时踩了单次验证的坑。** 一开始用 80/20 拆分选 max_depth，depth=2 的验证准确率最高（0.8156），但换成交叉验证后 CV 只有 0.7947，反而是 depth=8 最稳。单次拆分运气成分太大，以后默认用 CV。
- **Fare 在逻辑回归里几乎没用，在随机森林里排第二。** 原因是 Fare 和 Pclass 高度相关，逻辑回归里 Pclass 把信息抢走了，权重被压到 0.002；树模型对共线性没那么敏感，Fare 的作用就体现出来了。
- **XGBoost 没赢调参后的随机森林。** 891 条数据太小，几种模型差不多都到天花板了，换更强的模型未必有用。

## 依赖

- Python 3.12+
- pandas
- scikit-learn
- xgboost

## 数据来源

[Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

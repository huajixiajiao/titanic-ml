# 泰坦尼克号生存预测 - 机器学习完整项目

基于 Kaggle 经典泰坦尼克号数据集，从零开始使用 Python 进行完整的机器学习预测流程。

项目按照"从简单到复杂"的学习路径，依次使用三种模型进行预测：
**逻辑回归 -> 随机森林 -> XGBoost**，并包含完整的超参数调优过程。

## 项目结构

```
titanic-ml/
├── data/                           # 原始数据
│   ├── train.csv                   # 训练集 (891 rows, 12 cols)
│   ├── test.csv                    # 测试集 (418 rows, 11 cols)
│   └── gender_submission.csv       # 提交格式样例
├── src/                            # 源代码
│   ├── 01_logistic_regression.py   # 逻辑回归基线
│   ├── 02_random_forest.py         # 随机森林 (默认参数)
│   ├── 03_rf_hyperparameter_tuning.py  # 随机森林超参数调优
│   └── 04_xgboost.py              # XGBoost 梯度提升树
├── submissions/                    # 预测结果输出目录
├── requirements.txt                # Python 依赖
├── .gitignore
└── README.md
```

## 数据集说明

| 文件 | 说明 | 行数 | 列数 |
|------|------|------|------|
| `train.csv` | 训练集，包含 Survived 标签 | 891 | 12 |
| `test.csv` | 测试集，需要预测 Survived | 418 | 11 |
| `gender_submission.csv` | 提交格式样例 | 418 | 2 |

### 字段说明

| 字段 | 含义 | 类型 |
|------|------|------|
| PassengerId | 乘客 ID | int |
| Survived | 是否生还 (0=遇难, 1=生还) | int (标签) |
| Pclass | 舱位等级 (1/2/3) | int (有序类别) |
| Name | 姓名 | str |
| Sex | 性别 (male/female) | str |
| Age | 年龄 | float (含 177 个缺失) |
| SibSp | 同船兄弟姐妹/配偶数 | int |
| Parch | 同船父母/子女数 | int |
| Ticket | 票号 | str |
| Fare | 票价 | float |
| Cabin | 客舱号 | str (缺失 77%) |
| Embarked | 登船港口 (C/Q/S) | str (缺失 2 个) |

## 环境配置

```bash
# 克隆仓库
git clone https://github.com/huajixiajiao/titanic-ml.git
cd titanic-ml

# 安装依赖
pip install -r requirements.txt
```

## 运行方式

按顺序执行四个脚本，观察模型逐步演进：

```bash
# 1. 逻辑回归基线
python src/01_logistic_regression.py

# 2. 随机森林 (默认 max_depth=5)
python src/02_random_forest.py

# 3. 随机森林超参数调优 (搜索最优 max_depth)
python src/03_rf_hyperparameter_tuning.py

# 4. XGBoost 梯度提升树
python src/04_xgboost.py
```

每个脚本会自动输出准确率、分类报告、混淆矩阵、特征重要性，并在 `submissions/` 目录下生成 CSV 提交文件。

## 数据预处理流程

所有模型共用相同的预处理流程：

1. **缺失值填充**
   - Age: 用中位数填充 (约 28 岁)
   - Embarked: 用众数填充 ('S')
   - Fare: 用中位数填充
   - Cabin: 缺失 77%，直接丢弃

2. **类别编码**
   - Sex: male -> 0, female -> 1
   - Embarked: One-Hot 编码 -> Emb_C, Emb_Q, Emb_S

3. **特征工程**
   - FamilySize = SibSp + Parch + 1
   - IsAlone = (FamilySize == 1) ? 1 : 0

4. **特征选择**
   - 保留 11 个特征: Pclass, Sex, Age, SibSp, Parch, Fare, FamilySize, IsAlone, Emb_C, Emb_Q, Emb_S
   - 丢弃: PassengerId, Name, Ticket, Cabin

## 模型演进与结果

### 结果对比

| 模型 | 单次验证准确率 | 5 折 CV 均值 | CV 标准差 |
|------|:------------:|:-----------:|:--------:|
| 逻辑回归 | 0.8045 | ~0.80 | - |
| 随机森林 (depth=5) | 0.8045 | 0.8092 | 0.0201 |
| 随机森林 (depth=8, 调参后) | 0.8156 | **0.8272** | 0.0270 |
| XGBoost | 0.8156 | 0.8249 | **0.0228** |

### 各模型特点

**逻辑回归**
- 优点: 可解释性强，权重直接反映特征方向和大小
- 缺点: 只能学线性边界，无法捕捉特征交互
- 关键发现: Sex 权重 +2.49 (最强特征)，Fare 权重仅 +0.002 (因与 Pclass 共线性被压缩)

**随机森林**
- 优点: 自动处理非线性交互和相关特征，不需要标准化
- 调参关键: max_depth 是最重要参数，depth=8 时 CV 达到峰值 0.8272
- 重要教训: 单次验证选出的 depth=2 (0.8156) 被 CV 推翻 (CV 仅 0.7947)，说明交叉验证更可靠

**XGBoost**
- 优点: CV 标准差最小 (0.0228)，模型最稳定；调参空间更大
- 结果: 与调参后的随机森林几乎一致，说明已接近数据本身的信息上限
- 核心参数: learning_rate=0.1 + n_estimators=300 (小步多走)

### 特征重要性 (Top 4)

| 特征 | 逻辑回归 | 随机森林 (d=8) | XGBoost |
|------|:--------:|:-------------:|:-------:|
| Sex | 1st (+2.49) | 0.32 | 0.32 |
| Fare | Last (+0.002) | 0.23 | 0.23 |
| Age | -0.037 | 0.17 | 0.17 |
| Pclass | 2nd (-1.05) | 0.10 | 0.10 |

三个模型一致认为 **性别 (Sex) 是决定生还的最关键因素**，符合"妇女和儿童优先"的历史事实。

## 关键学习要点

1. **交叉验证优于单次拆分**: 单次 80/20 拆分受运气影响，5 折 CV 更可靠
2. **过拟合是最大陷阱**: max_depth 过大时训练集 98% 但验证集下降
3. **没有免费午餐**: 更强的模型 (XGBoost) 不一定在每个数据集上都赢
4. **特征重要性 vs 权重**: 树模型输出重要性 (只有大小)，逻辑回归输出权重 (有方向)
5. **瓶颈通常在数据而非模型**: 891 条小数据集上，三种模型性能趋同

## 技术栈

- Python 3.12+
- pandas - 数据处理
- scikit-learn - 机器学习框架
- XGBoost - 梯度提升树

## 数据来源

[Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

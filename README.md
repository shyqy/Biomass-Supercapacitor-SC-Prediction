# 超级电容器比电容预测软件

> 基于堆叠集成学习（Stacking Ensemble）的碳基电极材料比电容（Capacitance, F/g）预测工具，附带 Tkinter 图形界面。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange)

---

## 项目简介

多孔碳材料的比电容取决于孔结构、元素掺杂、石墨化程度与测试条件，靠实验逐一筛选成本高。本软件把这些表征与工况参数作为输入，用 5 个基学习器 + 岭回归元模型组成的堆叠集成模型，直接预测比电容：

- 输入 **12 个特征**：活化温度/时间、SSA、总孔容、平均孔径、微孔孔容、C/O/N 原子比、ID/IG、电流密度、电解液；
- 输出 **1 个指标**：比电容（F/g）；
- 提供 **Tkinter 界面**，填入数值点「Predict」即可出结果，无需写代码。

脚本在启动时读取 Excel/CSV 训练数据并现场训练模型，因此换一份结构相同的数据即可预测别的体系；继续补充样本也能扩大模型的适用范围、提升预测精度（见「扩充数据集」一节）。

---

## 主要特性

| 特性 | 说明 |
|---|---|
| 堆叠集成 | 基模型 RandomForest、XGBoost、CatBoost、LightGBM、KNN，元模型 RidgeCV（α ∈ {0.1, 1, 10}） |
| OOF 元特征 | 5 折交叉验证生成基模型的 out-of-fold 预测作为元模型输入，避免信息泄漏 |
| 异构预处理 | `ColumnTransformer` 数值特征直通、类别特征（电解液）独热编码；KNN 分支额外做标准化 |
| 列名自适应 | 自动归一化列名并按关键字映射（如 `Activation temperature/℃` → `activation_temperature`），支持中英文、大小写、空格与单位符号（℃、−、³）的不同写法 |
| 类别容错 | `OneHotEncoder(handle_unknown="ignore")`，遇到训练时未见过的电解液不会报错，仅置零 |
| 输入范围提示 | 界面直接标注每个特征的推荐取值范围 |
| 结果醒目展示 | 大号红字显示预测值并弹窗提示，避免漏看 |

---

## 输入特征

| 特征（内部名） | 界面显示 | 推荐范围 |
|---|---|---|
| `activation_temperature` | Activation temperature/℃ | [600, 1000] |
| `activation_time` | Activation time/min | [60, 240] |
| `ssa` | SSA/m²g⁻¹ | [72.7, 3977.3] |
| `total_pore_volume` | Total pore volume/cm³/g | [0.07, 2.372] |
| `average_pore_size` | Average pore size/nm | [0.77, 5.89] |
| `vmicro` | Vmicro/cm³/g | [0, 1.286] |
| `c` | C/at% | [41.05, 96.8] |
| `o` | O/at% | [1.93, 32.07] |
| `n` | N/at% | [0, 19.14] |
| `id_ig` | ID/IG | [0.12, 4.64] |
| `current_density` | Current Density / A/g | [0.1, 200] |
| `electrolyte` | Electrolyte（下拉选择） | `6 M KOH`、`4 M KOH`、`1 M H₂SO₄`、`2 M H₂SO₄` |

---

## 环境要求

- Python 3.13（Windows / macOS / Linux；图形界面需要系统自带 tkinter）
- 依赖见 [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

> CatBoost 在部分平台上需要较新的 C++ 运行库；若安装失败可先升级 pip 再装。

---

## 使用

### 1. 克隆并安装

```bash
git clone https://github.com/shyqy/Biomass-Supercapacitor-SC-Prediction.git
cd Biomass-Supercapacitor-SC-Prediction
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 启动

```bash
python 运行窗口.py
```

运行后：

1. 弹出文件选择框 → 选择 `Data.xlsx`（或任何列名可识别的 CSV/XLSX）；
2. 控制台打印「正在训练基模型…模型训练完成！」，一般十几秒到几十秒；
3. 弹出预测界面：左侧为材料与孔结构参数、中间为工况与电解液、右侧为预测结果；
4. 填写完毕后点击 **Predict** 得到比电容（F/g），**Clear** 清空重填。

### 3. 替换自己的数据

只要表头能通过关键字识别（含 `capacitance`、`activation temperature`、`ssa`、`total pore volume`、`average pore size`、`vmicro`、`electrolyte`、`current density`、`ID/IG`、`C/O/N` 等），就能直接换数据使用；列名大小写、空格、单位符号都会被自动归一化。

---

## 扩充数据集以提升适用范围与预测精度

模型的适用范围与精度上限，取决于训练数据覆盖了多少体系。仓库内置的 `Data.xlsx` 有 697 行、13 列（剔除缺失后 695 条有效样本），比电容覆盖 66.7 – 752.6 F/g，电解液只包含 KOH 与 H₂SO₄ 两类的四个浓度。继续向表中补充新的实验或文献数据，是提升模型最直接的方式。

**扩充为什么有效**

- **覆盖面**：新增不同前驱体与活化剂（KOH / ZnCl₂ / CO₂ / 水蒸气）、更宽的活化温度与时间、更宽的孔径与孔容、杂原子掺杂（N/S/P）、更多电解液（如 `3 M KOH`、`Na₂SO₄`、离子液体），模型才能在界面所示范围之外给出合理外推，而不只是对训练分布内插；
- **稳定性**：样本量变大后，5 折 OOF 元特征的方差下降，树模型不再依赖个别样本，KNN 分支的邻域也更具代表性；
- **趋势学习**：同一材料在不同电流密度下的样本成组保留，模型才能学到比电容随电流密度衰减的规律，而不是把它当作噪声。

**扩充时请注意**

1. 直接按行追加样本，表头保持可识别即可，列顺序不要求一致；
2. 单位统一：比表面积 m²/g、孔容 cm³/g、孔径 nm、电流密度 A/g、比电容 F/g；
3. 类别列（电解液）写字符串（如 `6 M KOH`），不要改成数值编码；
4. 含缺失值或无法转成数值的行会被整行剔除，建议先人工核对；
5. 若新增了训练数据里没有的电解液，界面下拉框的候选值写在 `运行窗口.py` 的 `ttk.Combobox(..., values=[...])` 处，按需补充即可。

**扩充后请重新评估**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# X、y 为按上文流程整理好的特征表与比电容
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pred = train_stacking_model(X_tr, y_tr)      # 用扩充后的数据训练
pred_te = pred.predict(X_te)
print("R²:", r2_score(y_te, pred_te))
print("RMSE:", mean_squared_error(y_te, pred_te) ** 0.5)
```

> 扩充不是「行数越多越好」：若新数据来自不同测试口径（两电极 / 三电极、电解液浓度标注混乱），可能反而拉低精度。补充前先统一测量与命名口径，并留出一部分数据完全不参与训练，用于外部验证。

---

## 项目结构

```
Biomass-Supercapacitor-SC-Prediction/
├── 运行窗口.py        # 入口：读数据 → 训练堆叠模型 → 启动 Tkinter 预测界面
├── Data.xlsx          # 训练数据（697 行 × 13 列）
├── requirements.txt
├── LICENSE
└── README.md
```

代码结构（`运行窗口.py` 内部）：

| 组件 | 作用 |
|---|---|
| `normalize_column_name` / `convert_ratio` | 列名归一化、比例字符串转数值 |
| `DataPreprocessor` | 数值/类别分流 + 独热编码，并锁定训练列顺序 |
| `StackingPredictor` | 保存预处理、基模型、元模型，对外提供 `predict(DataFrame)` |
| `train_stacking_model` | 训练基模型、5 折 OOF 生成元特征、拟合 RidgeCV |
| `PredictionApp` | Tkinter 三栏界面与输入校验 |

---

## 常见问题

| 问题 | 处理 |
|---|---|
| 弹出「Target variable not found」 | 数据里缺少含 `capacitance` 的列 |
| 报 `No module named 'catboost'` | `pip install -r requirements.txt` |
| 报 `No module named 'tkinter'` | Linux 安装系统包 `python3-tk` |
| 预测值明显偏离预期 | 检查输入是否超出界面标注的范围；比电容强依赖电流密度与电解液，二者须与训练数据口径一致 |
| 每次启动都要等训练 | 属正常设计：脚本每次启动都重新训练，不做模型持久化 |
| 数据里的电解液在下拉框里选不到 | 下拉候选值在脚本中固定为四种，需按需在 `values=[...]` 里补充 |

---

## 免责声明

本项目面向科研与教学用途。模型预测基于有限的历史实验数据，不能替代实测；用于材料筛选时请结合实际表征与电化学测试验证。

---

## 许可证

本项目采用 [MIT License](LICENSE)。

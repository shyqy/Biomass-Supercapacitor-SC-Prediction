import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 44


# ==================== 工具函数 ====================
def normalize_column_name(col):
    return str(col).strip().lower().replace(" ", "_").replace("℃", "c").replace("−", "-").replace("/", "_")


def convert_ratio(x):
    if isinstance(x, str) and ":" in x:
        parts = x.split(":")
        try:
            return float(parts[0]) / float(parts[1])
        except Exception:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ==================== 数据预处理与模型类 ====================
class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.fitted = False
        self.column_order_ = None
        self.numerical_cols = []
        self.categorical_cols = []

    def fit(self, X, y=None):
        self.column_order_ = X.columns.tolist()
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        force_numerical = ['sample_activator', 'activation_temperature', 'activation_time',
                           'ssa', 'total_pore_volume', 'average_pore_size', 'vmicro',
                           'c', 'o', 'n', 'id_ig', 'current_density']
        for col in force_numerical:
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)
                self.numerical_cols.append(col)
            elif col not in self.numerical_cols and col in X.columns:
                self.numerical_cols.append(col)

        self.preprocessor = ColumnTransformer([
            ("num", "passthrough", self.numerical_cols),
            ("cat", Pipeline([("onehot", make_onehot_encoder())]), self.categorical_cols),
        ])
        self.preprocessor.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("必须先fit")
        # 严格保证预测时的列顺序与训练时一致
        X = X[self.column_order_]
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class StackingPredictor:
    def __init__(self, preprocessor, scaler, base_models, meta_model, raw_feature_names):
        self.preprocessor = preprocessor
        self.scaler = scaler
        self.base_models = base_models
        self.meta_model = meta_model
        self.raw_feature_names = list(raw_feature_names)

    def predict(self, X_df):
        X_proc = self.preprocessor.transform(X_df)
        stack_feats = []
        for name, model, needs_scaling in self.base_models:
            X_use = self.scaler.transform(X_proc) if needs_scaling else X_proc
            stack_feats.append(model.predict(X_use))
        return self.meta_model.predict(np.column_stack(stack_feats))


# ==================== 核心训练流程 ====================
def train_stacking_model(X_raw, y_series):
    print("开始预处理数据...")
    preprocessor = DataPreprocessor()
    X_proc = preprocessor.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)

    base_configs = [
        ("RandomForest", RandomForestRegressor(
            n_estimators=194, max_depth=12, min_samples_split=2, min_samples_leaf=1, max_features='log2',
            random_state=RANDOM_STATE), False),
        ("XGBoost", XGBRegressor(
            n_estimators=156, max_depth=6, learning_rate=0.13, subsample=0.8, colsample_bytree=0.756,
            min_child_weight=4, random_state=RANDOM_STATE, verbosity=0), False),
        ("CatBoost", CatBoostRegressor(
            iterations=200, depth=5, learning_rate=0.15, l2_leaf_reg=2, border_count=32, verbose=False,
            random_state=RANDOM_STATE), False),
        ("LGBM", LGBMRegressor(
            n_estimators=300, max_depth=10, num_leaves=15, learning_rate=0.07, min_child_samples=5, subsample=1.0,
            colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=10.0, verbose=-1, random_state=RANDOM_STATE), False),
        ("KNN", KNeighborsRegressor(
            n_neighbors=3, weights='distance', p=2), True)
    ]

    print("正在训练基模型...")
    fitted_models = []
    for name, model, needs_scaling in base_configs:
        X_use = X_scaled if needs_scaling else X_proc
        model.fit(X_use, y_series)
        fitted_models.append((name, model, needs_scaling))

    print("正在生成 OOF 特征并训练元模型...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stack_features = np.zeros((X_proc.shape[0], len(fitted_models)))

    for train_idx, val_idx in kf.split(X_proc):
        for i, (name, model, needs_scaling) in enumerate(fitted_models):
            model_copy = deepcopy(model)
            X_tr = X_scaled[train_idx] if needs_scaling else X_proc[train_idx]
            X_val = X_scaled[val_idx] if needs_scaling else X_proc[val_idx]
            model_copy.fit(X_tr, y_series.iloc[train_idx])
            stack_features[val_idx, i] = model_copy.predict(X_val)

    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_model.fit(stack_features, y_series)

    predictor = StackingPredictor(
        preprocessor=preprocessor,
        scaler=scaler,
        base_models=fitted_models,
        meta_model=meta_model,
        raw_feature_names=X_raw.columns.tolist()
    )
    print("模型训练完成！")
    return predictor


# ==================== Tkinter 预测界面 ====================
class PredictionApp:
    def __init__(self, root, predictor, feature_names, categorical_features):
        self.root = root
        self.predictor = predictor
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.entries = {}

        self.root.title("Hydrothermal product properties prediction")
        self.root.geometry("1350x650")
        self.root.configure(bg="#f0f0f0")

        self.font_title = ("Times New Roman", 16, "bold")
        self.font_subtitle = ("Times New Roman", 14, "bold")
        self.font_normal = ("Times New Roman", 14)

        self.BG = "#f0f0f0"

        # 建立特征名映射与范围显示
        self.display_map = {
            "activation_temperature": ("Activation temperature/℃", "[600, 1000]"),
            "activation_time": ("Activation time/min", "[60, 240]"),
            "ssa": ("SSA/m²g⁻¹", "[72.7, 3977.3]"),
            "total_pore_volume": ("Total pore volume/cm³/g", "[0.07, 2.372]"),
            "average_pore_size": ("Average pore size/nm", "[0.77, 5.89]"),
            "vmicro": ("Vmicro /cm³/g", "[0, 1.286]"),
            "c": ("C/at%", "[41.05, 96.8]"),
            "o": ("O/at%", "[1.93, 32.07]"),
            "n": ("N/at%", "[0, 19.14]"),
            "id_ig": ("ID/IG", "[0.12, 4.64]"),
            "current_density": ("Current DensityA/g", "[0.1, 200]"),
            "electrolyte": ("Electrolyte", "")
        }

        self.create_widgets()

    def create_widgets(self):
        self.status_var = tk.StringVar(value="Ready.")

        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        main.grid_columnconfigure(0, weight=1, minsize=380)
        main.grid_columnconfigure(1, weight=1, minsize=380)
        main.grid_columnconfigure(2, weight=1, minsize=300)

        # ---------- 左侧面板 (Material & Pore Properties) ----------
        left_frame = tk.LabelFrame(main, text="Material & Pore Properties", font=self.font_title, bg=self.BG, padx=6,
                                   pady=6)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        elem_frame = tk.LabelFrame(left_frame, text="Textural Properties", font=self.font_subtitle, bg=self.BG, padx=5,
                                   pady=5)
        elem_frame.pack(fill="x", pady=4)
        elem_keys = ["ssa", "total_pore_volume", "average_pore_size", "vmicro"]
        for i, key in enumerate(elem_keys):
            if key in self.feature_names:
                self._add_input(elem_frame, key, i)

        comp_frame = tk.LabelFrame(left_frame, text="Elemental Composition & Raman", font=self.font_subtitle,
                                   bg=self.BG, padx=5, pady=5)
        comp_frame.pack(fill="x", pady=4)
        comp_keys = ["c", "o", "n", "id_ig"]
        for i, key in enumerate(comp_keys):
            if key in self.feature_names:
                self._add_input(comp_frame, key, i)

        # ---------- 中间面板 (Operating Conditions) ----------
        mid_frame = tk.LabelFrame(main, text="Operating Conditions", font=self.font_title, bg=self.BG, padx=6, pady=6)
        mid_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        act_frame = tk.LabelFrame(mid_frame, text="Activation Conditions", font=self.font_subtitle, bg=self.BG, padx=5,
                                  pady=5)
        act_frame.pack(fill="x", pady=4)
        act_keys = ["activation_temperature", "activation_time"]
        for i, key in enumerate(act_keys):
            if key in self.feature_names:
                self._add_input(act_frame, key, i)

        elec_frame = tk.LabelFrame(mid_frame, text="Electrochemical Setup", font=self.font_subtitle, bg=self.BG, padx=5,
                                   pady=5)
        elec_frame.pack(fill="x", pady=4)

        if "current_density" in self.feature_names:
            self._add_input(elec_frame, "current_density", 0)

        if "electrolyte" in self.feature_names:
            disp_name, rng = self.display_map.get("electrolyte", ("Electrolyte", ""))
            tk.Label(elec_frame, text=disp_name, font=self.font_normal, bg=self.BG, anchor="w").grid(row=1, column=0,
                                                                                                     sticky="w",
                                                                                                     padx=(8, 4),
                                                                                                     pady=3)
            entry = ttk.Combobox(elec_frame, values=['6 M KOH', '1 M H₂SO₄', '4 M KOH', '2 M H₂SO₄'], state="readonly",
                                 width=10, font=self.font_normal)
            entry.grid(row=1, column=1, sticky="w", padx=(0, 4), pady=3)
            self.entries["electrolyte"] = entry

        note_label = tk.Label(mid_frame, text="Note: Ensure all features are within\nthe recommended ranges.",
                              font=("Times New Roman", 12, "italic"), bg=self.BG, fg="#4a6a8b", wraplength=300)
        note_label.pack(pady=10)

        # ---------- 右侧面板 (Predicted Performance) ----------
        right_frame = tk.LabelFrame(main, text="Predicted Performance", font=self.font_title, bg=self.BG, padx=6,
                                    pady=6)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        out_frame = tk.LabelFrame(right_frame, text="Capacitance", font=self.font_subtitle, bg=self.BG, padx=5, pady=5)
        out_frame.pack(fill="x", pady=5)

        tk.Label(out_frame, text="Capacitance (F/g)", font=self.font_normal, bg=self.BG).grid(row=0, column=0,
                                                                                              sticky="w", padx=8,
                                                                                              pady=8)

        # 修改：使用大号红色 Label 直接显示结果，更加醒目且不会出现 Entry 的渲染 bug
        self.result_label = tk.Label(out_frame, text="", font=("Times New Roman", 20, "bold"), fg="#d32f2f",
                                     bg="#ffffff", width=15, relief="solid", bd=1, anchor="center")
        self.result_label.grid(row=0, column=1, sticky="w", padx=10, pady=8)

        # ---------- 按钮与状态 ----------
        btn_frame = tk.Frame(main, bg=self.BG)
        btn_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)

        btn_container = tk.Frame(btn_frame, bg=self.BG)
        btn_container.pack(anchor="center")

        tk.Button(btn_container, text="Predict", width=10, command=self.predict, font=self.font_normal).pack(
            side="left", padx=15)
        tk.Button(btn_container, text="Clear", width=10, command=self.clear, font=self.font_normal).pack(side="left",
                                                                                                         padx=15)

        status_label = tk.Label(main, textvariable=self.status_var, bg=self.BG, fg="#d32f2f",
                                font=("Times New Roman", 12))
        status_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))

    def _add_input(self, parent, key, row):
        disp_name, rng = self.display_map.get(key, (key, ""))

        tk.Label(parent, text=disp_name, font=self.font_normal, bg=self.BG, anchor="w").grid(row=row, column=0,
                                                                                             sticky="w", padx=(8, 4),
                                                                                             pady=3)

        entry = tk.Entry(parent, width=10, font=self.font_normal, bg="#ffffff", relief="solid", bd=1)
        entry.grid(row=row, column=1, sticky="w", padx=(0, 4), pady=3)

        if rng:
            tk.Label(parent, text=rng, font=self.font_normal, fg="#5f6f82", bg=self.BG).grid(row=row, column=2,
                                                                                             sticky="w", padx=(0, 5),
                                                                                             pady=3)

        self.entries[key] = entry

    def predict(self):
        print("\n--- 点击预测，开始读取输入数据 ---")
        self.status_var.set("Predicting...")
        self.root.update()

        try:
            input_data = {}

            for feat in self.feature_names:
                entry = self.entries.get(feat)
                if not entry:
                    msg = f"内部错误：未找到 {feat} 的输入框"
                    print(msg)
                    messagebox.showerror("Error", msg)
                    return

                val_str = entry.get().strip()
                disp_name = self.display_map.get(feat, (feat, ""))[0]

                if not val_str:
                    msg = f"请填写 {disp_name} 的值！"
                    print(msg)
                    messagebox.showwarning("数据缺失", msg)
                    self.status_var.set(msg)
                    return

                if feat in self.categorical_features:
                    input_data[feat] = [val_str]
                else:
                    try:
                        input_data[feat] = [float(val_str)]
                    except ValueError:
                        msg = f"{disp_name} 必须输入有效的数字！当前输入: {val_str}"
                        print(msg)
                        messagebox.showerror("输入错误", msg)
                        self.status_var.set(msg)
                        return

            X_df = pd.DataFrame(input_data, index=[0])
            X_df = X_df[self.feature_names]

            print("用户输入的特征数据：")
            print(X_df.T)

            # 执行预测
            pred = self.predictor.predict(X_df)[0]
            print(f"预测结果: {pred:.4f} F/g")

            # 更新界面并弹窗提示
            result_text = f"{pred:.4f} F/g"
            self.result_label.config(text=result_text)
            self.status_var.set("Prediction successful!")

            # 弹出提示框确保用户能看到结果
            messagebox.showinfo("Prediction Result", f"Predicted Capacitance:\n{result_text}")

        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            print("发生异常:\n", err_trace)
            messagebox.showerror("Prediction Error", f"预测过程中发生错误:\n{str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def clear(self):
        for entry in self.entries.values():
            if isinstance(entry, ttk.Combobox):
                entry.set("")
            else:
                entry.delete(0, tk.END)
        self.result_label.config(text="")
        self.status_var.set("Ready.")


# ==================== 主入口 ====================
if __name__ == "__main__":
    print("Please select the training data file...")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select Training Data",
        filetypes=[("Excel Files", "*.xlsx *.xls"), ("CSV Files", "*.csv")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        exit()

    try:
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)
        print(f"Data loaded successfully, total rows: {len(data)}.")
    except Exception as e:
        print(f"Failed to read file: {e}")
        exit()

    # ================= 关键修复：强制将原始列名映射为统一的内部命名 =================
    data.columns = [normalize_column_name(c) for c in data.columns]

    rename_dict = {}
    for col in data.columns:
        if 'capacitance' in col:
            rename_dict[col] = 'capacitance'
        elif 'activation' in col and 'temperature' in col:
            rename_dict[col] = 'activation_temperature'
        elif 'activation' in col and 'time' in col:
            rename_dict[col] = 'activation_time'
        elif 'ssa' in col:
            rename_dict[col] = 'ssa'
        elif 'total' in col and 'pore' in col and 'volume' in col:
            rename_dict[col] = 'total_pore_volume'
        elif 'average' in col and 'pore' in col and 'size' in col:
            rename_dict[col] = 'average_pore_size'
        elif 'vmicro' in col:
            rename_dict[col] = 'vmicro'
        elif 'current' in col and 'density' in col:
            rename_dict[col] = 'current_density'
        elif 'electrolyte' in col:
            rename_dict[col] = 'electrolyte'
        elif 'id' in col and 'ig' in col:
            rename_dict[col] = 'id_ig'
        elif col in ['c', 'c_content', 'c_at%', 'c_wt%', 'c_(at%)', 'c_(wt%)'] or 'carbon' in col:
            rename_dict[col] = 'c'
        elif col in ['o', 'o_content', 'o_at%', 'o_wt%', 'o_(at%)', 'o_(wt%)'] or 'oxygen' in col:
            rename_dict[col] = 'o'
        elif col in ['n', 'n_content', 'n_at%', 'n_wt%', 'n_(at%)', 'n_(wt%)'] or 'nitrogen' in col:
            rename_dict[col] = 'n'

    data = data.rename(columns=rename_dict)
    # =========================================================================

    output_col = None
    for col in data.columns:
        if "capacitance" in col:
            output_col = col
            break

    if not output_col:
        print("Error: Target variable (containing 'capacitance') not found in data!")
        exit()

    # 严格按照 12 个标准特征来提取
    standard_features = [
        "activation_temperature", "activation_time",
        "ssa", "total_pore_volume", "average_pore_size", "vmicro",
        "c", "o", "n", "id_ig", "current_density", "electrolyte"
    ]
    available_input = [c for c in standard_features if c in data.columns]

    # 数据类型清洗
    data[output_col] = pd.to_numeric(data[output_col], errors="coerce")
    if 'sample_activator' in data.columns:
        data['sample_activator'] = data['sample_activator'].apply(convert_ratio)

    categorical_feature_names = [c for c in ['agent', 'electrolyte'] if c in available_input]

    for col in available_input:
        if col in categorical_feature_names:
            data[col] = data[col].astype(str).str.strip()
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=[output_col] + available_input)

    # 3. 训练模型
    X_train = data[available_input]
    y_train = data[output_col]

    predictor = train_stacking_model(X_train, y_train)

    # 4. 启动预测界面
    root_app = tk.Tk()
    app = PredictionApp(root_app, predictor, available_input, categorical_feature_names)
    root_app.mainloop()
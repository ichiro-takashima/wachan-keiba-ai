"""LightGBM二値分類モデルの学習・評価まわりの共通処理。

train_base.py(ベースモデル)・tune_optuna.py(チューニング)・train_anauma.py(穴馬モデル)
の3つから共通で呼び出す。
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# 当該レース自身の結果に由来し、予測時点では手に入らない列(リークになるため除外)。
# 直近5走の履歴列(例: オッズ_1, 着順_1 など)は過去情報なので除外しない。
LEAK_COLUMNS = {"オッズ", "人気", "上がり", "走破時間", "通過順"}

# 特徴量として使わない識別子・生テキスト・目的変数列。
NON_FEATURE_COLUMNS = {
    "race_id", "馬", "騎手", "レース名", "開催", "場名", "日付",
    "着順", "着順_数値", "top3", "top3_anauma",
}


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c not in LEAK_COLUMNS]


def _to_numeric_matrix(df, feature_cols):
    return df[feature_cols].apply(pd.to_numeric, errors="coerce")


def _split_indices(df, split, test_size, random_state, y):
    if split == "time":
        sortable = pd.to_datetime(df["日付"], format="%Y/%m/%d", errors="coerce")
        order = sortable.sort_values(kind="stable").index
        cut = int(len(order) * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        return train_idx, test_idx

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )
    return train_idx, test_idx


@dataclass
class TrainResult:
    model: lgb.LGBMClassifier
    metrics: dict
    importance: pd.DataFrame
    test_df: pd.DataFrame  # race_id, 馬番, オッズ, y_true, y_pred_proba を含むテスト結果
    feature_cols: list


def train_and_evaluate(
    df,
    target_col="top3",
    split="random",
    test_size=0.3,
    random_state=42,
    feature_cols=None,
    lgb_params=None,
):
    """LightGBMで二値分類を学習し、評価指標一式とテスト結果を返す。

    split="random"なら層化ランダム7:3分割(既定)、split="time"なら日付でソートし
    後方をテストにする時系列分割。
    """
    working = df[df[target_col].notna()].copy()
    y_all = working[target_col].astype(int)

    feature_cols = feature_cols or get_feature_columns(working)
    x_all = _to_numeric_matrix(working, feature_cols)

    train_idx, test_idx = _split_indices(working, split, test_size, random_state, y_all)
    x_train, x_test = x_all.loc[train_idx], x_all.loc[test_idx]
    y_train, y_test = y_all.loc[train_idx], y_all.loc[test_idx]

    params = {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": random_state,
        "verbosity": -1,
    }
    if lgb_params:
        params.update(lgb_params)

    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train)

    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    test_df = working.loc[test_idx, ["race_id", "馬番", "オッズ"]].copy()
    test_df["y_true"] = y_test.values
    test_df["y_pred_proba"] = y_pred_proba

    return TrainResult(
        model=model,
        metrics=metrics,
        importance=importance,
        test_df=test_df,
        feature_cols=feature_cols,
    )


def print_metrics(metrics, title="評価結果"):
    print(f"--- {title} ---")
    print(f"学習件数: {metrics['n_train']} / 評価件数: {metrics['n_test']}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"AUC      : {metrics['auc']:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    print("混同行列 (行=実際, 列=予測) [[TN, FP], [FN, TP]]:")
    print(cm)


def plot_feature_importance(importance_df, output_path, top_n=20, title="LightGBM 特徴量重要度 (Top20)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 既定フォント(DejaVu Sans)は日本語グリフを持たないため、特徴量名が文字化けする。
    # Windows標準搭載のゴシック体フォントを明示指定する。
    matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

    top_df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_df))))
    ax.barh(top_df["feature"], top_df["importance"], color="#4C72B0")
    ax.set_xlabel("importance (gain)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

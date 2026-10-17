"""Step3: SHAPで芝・ダート×距離区分ごとの特徴量重要度の違いを可視化する(アプローチ3)。

model/train_base.pyで区分ごとにモデルを分割する前段階として、今の統合モデルに
SHAPを適用し、テスト集合を芝・ダート×距離区分でグループ化した上で「同じ特徴量でも
区分によって重要度がどう変わるか」を確認する。区分間の重要度ランキングの相関が低いほど、
区分ごとにモデルを分ける価値が高いと判断できる。

例:
    python -m model.shap_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model.common import _split_indices, _to_numeric_matrix, get_feature_columns, train_and_evaluate

SURFACE_LABELS = {0.0: "芝", 1.0: "ダート"}
DISTANCE_LABELS = {
    0.0: "短距離(-1400)",
    1.0: "マイル(1401-1800)",
    2.0: "中距離(1801-2200)",
    3.0: "長距離(2201-)",
}
MIN_GROUP_SIZE = 20


def compute_group_shap_importance(
    df,
    target_col="top3",
    split="random",
    test_size=0.3,
    random_state=42,
):
    """テスト集合のSHAP値を計算し、区分(芝・ダート×距離区分)ごとの平均|SHAP|を返す。

    戻り値: (group_importance, group_sizes)
    group_importance: index=グループ名, columns=特徴量 の平均|SHAP|表
    """
    import shap

    working = df[df[target_col].notna()].copy()
    feature_cols = get_feature_columns(working)
    x_all = _to_numeric_matrix(working, feature_cols)
    y_all = working[target_col].astype(int)

    result = train_and_evaluate(
        df, target_col=target_col, split=split, test_size=test_size, random_state=random_state
    )

    _, test_idx = _split_indices(working, split, test_size, random_state, y_all)
    x_test = x_all.loc[test_idx]

    explainer = shap.TreeExplainer(result.model.booster_)
    shap_values = explainer.shap_values(x_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    abs_shap = pd.DataFrame(np.abs(shap_values), columns=feature_cols, index=test_idx)
    groups = working.loc[test_idx, ["芝・ダート", "距離区分"]]

    rows = {}
    sizes = {}
    for surface_code, surface_name in SURFACE_LABELS.items():
        for dist_code, dist_name in DISTANCE_LABELS.items():
            mask = (groups["芝・ダート"] == surface_code) & (groups["距離区分"] == dist_code)
            n = int(mask.sum())
            if n < MIN_GROUP_SIZE:
                continue
            group_name = f"{surface_name}/{dist_name}"
            rows[group_name] = abs_shap.loc[mask[mask].index].mean()
            sizes[group_name] = n

    group_importance = pd.DataFrame(rows).T
    group_sizes = pd.Series(sizes, name="n")
    return group_importance, group_sizes


def plot_group_heatmap(group_importance, output_path, top_n=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

    # グループ間でばらつきが大きい(=区分依存が強い)特徴量を優先して表示する。
    normalized = group_importance / group_importance.max(axis=0).replace(0, np.nan)
    top_features = normalized.var(axis=0).sort_values(ascending=False).head(top_n).index
    view = normalized[top_features].T

    fig, ax = plt.subplots(figsize=(1.1 * len(view.columns) + 3, 0.35 * len(view.index) + 2))
    im = ax.imshow(view.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(view.columns)))
    ax.set_xticklabels(view.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(view.index)))
    ax.set_yticklabels(view.index)
    ax.set_title("区分別 特徴量重要度(|SHAP|、区分内最大値=1に正規化) Top{}".format(top_n))
    fig.colorbar(im, ax=ax, label="正規化重要度")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_group_correlation(group_importance, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

    # 特徴量重要度の「順位」のスピアマン相関で、区分ごとの重要度パターンの近さを測る。
    corr = group_importance.rank(axis=1, ascending=False).T.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(0.6 * len(corr) + 3, 0.6 * len(corr) + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title("区分間の特徴量重要度ランキング相関(Spearman)")
    fig.colorbar(im, ax=ax, label="相関係数")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return corr


def main():
    parser = argparse.ArgumentParser(description="SHAPで芝・ダート×距離区分ごとの特徴量重要度の違いを可視化する")
    parser.add_argument("--features-csv", default="data/processed/features.csv")
    parser.add_argument("--split", choices=["random", "time"], default="random")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--output-dir", default="data/models")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str}, low_memory=False)
    print(f"読み込み: {df.shape[0]}行 / {df.shape[1]}列 (features-csv: {args.features_csv})")

    group_importance, group_sizes = compute_group_shap_importance(
        df, split=args.split, test_size=args.test_size
    )

    print("\n--- 区分ごとのテスト件数 ---")
    print(group_sizes.to_string())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_out = output_dir / "shap_group_importance.csv"
    group_importance.to_csv(csv_out, encoding="utf-8-sig")
    print(f"\n区分別重要度表 保存: {csv_out}")

    print("\n--- 区分ごとの重要度 Top5 ---")
    for group_name, row in group_importance.iterrows():
        top5 = row.sort_values(ascending=False).head(5)
        print(f"[{group_name}] (n={group_sizes[group_name]})")
        for feat, val in top5.items():
            print(f"  {feat}: {val:.4f}")

    heatmap_out = output_dir / "shap_group_heatmap.png"
    plot_group_heatmap(group_importance, heatmap_out)
    print(f"\n区分別重要度ヒートマップ 保存: {heatmap_out}")

    corr_out = output_dir / "shap_group_correlation.png"
    corr = plot_group_correlation(group_importance, corr_out)
    print(f"区分間相関ヒートマップ 保存: {corr_out}")

    print("\n--- 区分間の重要度ランキング相関(Spearman、1に近いほど似た重要度パターン) ---")
    print(corr.round(2).to_string())

    off_diag = corr.values[~np.eye(len(corr), dtype=bool)]
    print(f"\n区分間相関の平均: {off_diag.mean():.3f} (最小: {off_diag.min():.3f})")
    if off_diag.mean() < 0.7:
        print("→ 区分間で重要度パターンの差が大きく、区分ごとのモデル分割(アプローチ1)に価値がある可能性が高い。")
    else:
        print("→ 区分間で重要度パターンは概ね似ており、モデル分割による上乗せは限定的な可能性がある。")


if __name__ == "__main__":
    main()

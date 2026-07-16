"""Step5: 穴馬モデル(オッズ15倍以上かつ3着以内=1)の学習・評価CLI。

ベースモデル(Step2)と同じ評価・回収率計算を行い、両モデルを比較する。

例:
    python -m model.train_anauma
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.roi import compute_roi_table, load_payouts
from model.common import plot_feature_importance, print_metrics, train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="穴馬モデル(オッズ15倍以上×3着以内)を学習し、ベースモデルと比較する")
    parser.add_argument("--features-csv", default="data/processed/features.csv")
    parser.add_argument("--payouts-csv", default="data/processed/payouts.csv")
    parser.add_argument("--split", choices=["random", "time"], default="random")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--model-out", default="data/models/anauma_model.txt")
    parser.add_argument("--importance-out", default="data/models/anauma_model_importance.png")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str})
    payouts_by_race = load_payouts(args.payouts_csv)

    print("=== ベースモデル (3着以内) ===")
    base_result = train_and_evaluate(df, target_col="top3", split=args.split, test_size=args.test_size)
    print_metrics(base_result.metrics, title="ベースモデル")
    base_roi = compute_roi_table(base_result.test_df, payouts_by_race)
    print(base_roi.to_string(index=False))

    print("\n=== 穴馬モデル (オッズ15倍以上 かつ 3着以内) ===")
    anauma_target_count = df["top3_anauma"].value_counts(dropna=False)
    print(f"目的変数の分布:\n{anauma_target_count.to_string()}")

    anauma_result = train_and_evaluate(df, target_col="top3_anauma", split=args.split, test_size=args.test_size)
    print_metrics(anauma_result.metrics, title="穴馬モデル")
    anauma_roi = compute_roi_table(anauma_result.test_df, payouts_by_race)
    print(anauma_roi.to_string(index=False))

    print("\n=== 比較サマリー(ベースモデル vs 穴馬モデル) ===")
    compare_rows = []
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        compare_rows.append({
            "指標": key,
            "ベースモデル": round(base_result.metrics[key], 4),
            "穴馬モデル": round(anauma_result.metrics[key], 4),
        })
    print(pd.DataFrame(compare_rows).to_string(index=False))

    roi_compare = base_roi[["閾値", "単勝回収率(%)", "複勝回収率(%)"]].merge(
        anauma_roi[["閾値", "単勝回収率(%)", "複勝回収率(%)"]],
        on="閾値",
        suffixes=("_ベース", "_穴馬"),
    )
    print(roi_compare.to_string(index=False))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    anauma_result.model.booster_.save_model(args.model_out)
    plot_feature_importance(anauma_result.importance, args.importance_out, title="LightGBM 特徴量重要度 (Top20, 穴馬モデル)")
    print(f"\n穴馬モデル保存: {args.model_out}")
    print(f"穴馬モデル 重要度グラフ保存: {args.importance_out}")

    metrics_out = str(Path(args.model_out).with_suffix(".metrics.json"))
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(anauma_result.metrics, f, ensure_ascii=False, indent=2)
    print(f"評価指標保存: {metrics_out}")


if __name__ == "__main__":
    main()

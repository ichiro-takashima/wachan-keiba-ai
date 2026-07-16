"""Step2: ベースモデル(3着以内の二値分類)の学習・評価CLI。

例:
    python -m model.train_base
    python -m model.train_base --split time
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from model.common import plot_feature_importance, print_metrics, train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="ベースモデル(3着以内二値分類)を学習・評価する")
    parser.add_argument("--features-csv", default="data/processed/features.csv")
    parser.add_argument("--split", choices=["random", "time"], default="random",
                         help="random: 層化ランダム7:3分割(既定) / time: 日付順に後方30%%をテストにする時系列分割")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--model-out", default="data/models/base_model.txt")
    parser.add_argument("--importance-out", default="data/models/base_model_importance.png")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str})
    print(f"読み込み: {df.shape[0]}行 / {df.shape[1]}列 (features-csv: {args.features_csv})")

    result = train_and_evaluate(df, split=args.split, test_size=args.test_size)
    print_metrics(result.metrics, title=f"ベースモデル (split={args.split})")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    result.model.booster_.save_model(args.model_out)
    print(f"モデル保存: {args.model_out}")

    plot_feature_importance(result.importance, args.importance_out)
    print(f"特徴量重要度グラフ保存: {args.importance_out}")
    print("--- 特徴量重要度 Top10 ---")
    print(result.importance.head(10).to_string(index=False))

    metrics_out = str(Path(args.model_out).with_suffix(".metrics.json"))
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2)
    print(f"評価指標保存: {metrics_out}")


if __name__ == "__main__":
    main()

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
    parser.add_argument("--surface", choices=["all", "turf", "dirt"], default="all",
                         help="all: 芝・ダート問わず1モデル(既定) / turf: 芝のみで学習 / dirt: ダートのみで学習")
    parser.add_argument("--model-out", default=None,
                         help="既定値は--surfaceに応じて data/models/base_model[_turf|_dirt].txt")
    parser.add_argument("--importance-out", default=None,
                         help="既定値は--surfaceに応じて data/models/base_model_importance[_turf|_dirt].png")
    args = parser.parse_args()

    surface_suffix = {"all": "", "turf": "_turf", "dirt": "_dirt"}[args.surface]
    model_out = args.model_out or f"data/models/base_model{surface_suffix}.txt"
    importance_out = args.importance_out or f"data/models/base_model_importance{surface_suffix}.png"

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str})
    print(f"読み込み: {df.shape[0]}行 / {df.shape[1]}列 (features-csv: {args.features_csv})")

    if args.surface != "all":
        surface_code = {"turf": 0.0, "dirt": 1.0}[args.surface]
        before = len(df)
        df = df[df["芝・ダート"] == surface_code].reset_index(drop=True)
        print(f"芝・ダートで絞り込み({args.surface}): {before}行 → {df.shape[0]}行")

    result = train_and_evaluate(df, split=args.split, test_size=args.test_size)
    print_metrics(result.metrics, title=f"ベースモデル (split={args.split}, surface={args.surface})")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    result.model.booster_.save_model(model_out)
    print(f"モデル保存: {model_out}")

    plot_feature_importance(result.importance, importance_out)
    print(f"特徴量重要度グラフ保存: {importance_out}")
    print("--- 特徴量重要度 Top10 ---")
    print(result.importance.head(10).to_string(index=False))

    metrics_out = str(Path(model_out).with_suffix(".metrics.json"))
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2)
    print(f"評価指標保存: {metrics_out}")


if __name__ == "__main__":
    main()

"""Step4: optuna(LightGBMTunerCV)によるハイパーパラメータチューニング。

チューニング後のパラメータでStep2と同じ評価・Step3と同じ回収率計算を再実行し、
チューニング前後を比較する。

例:
    python -m model.tune_optuna
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import optuna_integration.lightgbm as opt_lgb
import pandas as pd

from evaluation.roi import compute_roi_table, load_payouts
from model.common import (
    _split_indices,
    _to_numeric_matrix,
    get_feature_columns,
    plot_feature_importance,
    print_metrics,
    train_and_evaluate,
)

optuna_integration_logger = __import__("optuna").logging
optuna_integration_logger.set_verbosity(optuna_integration_logger.WARNING)


def tune_hyperparameters(
    df,
    target_col="top3",
    split="random",
    test_size=0.3,
    random_state=42,
    num_boost_round=100,
    nfold=4,
):
    """LightGBMTunerCVで訓練データ内CVによりハイパーパラメータを探索し、best_paramsを返す。"""
    working = df[df[target_col].notna()].copy()
    y_all = working[target_col].astype(int)
    feature_cols = get_feature_columns(working)
    x_all = _to_numeric_matrix(working, feature_cols)

    train_idx, _ = _split_indices(working, split, test_size, random_state, y_all)
    x_train, y_train = x_all.loc[train_idx], y_all.loc[train_idx]

    dtrain = lgb.Dataset(x_train, label=y_train)
    base_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    tuner = opt_lgb.LightGBMTunerCV(
        base_params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=True,
        shuffle=True,
        seed=random_state,
        callbacks=[lgb.early_stopping(30, verbose=False)],
        show_progress_bar=False,
        optuna_seed=random_state,
    )
    tuner.run()
    return tuner.best_params, feature_cols


def main():
    parser = argparse.ArgumentParser(description="optunaでLightGBMをチューニングし、Step2/3をbefore/after比較する")
    parser.add_argument("--features-csv", default="data/processed/features.csv")
    parser.add_argument("--payouts-csv", default="data/processed/payouts.csv")
    parser.add_argument("--split", choices=["random", "time"], default="random")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--num-boost-round", type=int, default=100)
    parser.add_argument("--nfold", type=int, default=4)
    parser.add_argument("--model-out", default="data/models/tuned_model.txt")
    parser.add_argument("--importance-out", default="data/models/tuned_model_importance.png")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str})
    payouts_by_race = load_payouts(args.payouts_csv)

    print("=== チューニング前(ベースモデル) ===")
    before_result = train_and_evaluate(df, split=args.split, test_size=args.test_size)
    print_metrics(before_result.metrics, title="チューニング前")
    before_roi = compute_roi_table(before_result.test_df, payouts_by_race)
    print(before_roi.to_string(index=False))

    print("\n=== optunaでハイパーパラメータ探索中 ===")
    best_params, _ = tune_hyperparameters(
        df,
        split=args.split,
        test_size=args.test_size,
        num_boost_round=args.num_boost_round,
        nfold=args.nfold,
    )
    print("ベストパラメータ:", json.dumps(best_params, ensure_ascii=False, indent=2))

    print("\n=== チューニング後 ===")
    after_result = train_and_evaluate(df, split=args.split, test_size=args.test_size, lgb_params=best_params)
    print_metrics(after_result.metrics, title="チューニング後")
    after_roi = compute_roi_table(after_result.test_df, payouts_by_race)
    print(after_roi.to_string(index=False))

    print("\n=== 比較サマリー(チューニング前 → 後) ===")
    compare_rows = []
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        compare_rows.append({
            "指標": key,
            "チューニング前": round(before_result.metrics[key], 4),
            "チューニング後": round(after_result.metrics[key], 4),
        })
    print(pd.DataFrame(compare_rows).to_string(index=False))

    roi_compare = before_roi[["閾値", "単勝回収率(%)", "複勝回収率(%)"]].merge(
        after_roi[["閾値", "単勝回収率(%)", "複勝回収率(%)"]],
        on="閾値",
        suffixes=("_前", "_後"),
    )
    print(roi_compare.to_string(index=False))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    after_result.model.booster_.save_model(args.model_out)
    plot_feature_importance(after_result.importance, args.importance_out, title="LightGBM 特徴量重要度 (Top20, チューニング後)")
    print(f"\nチューニング後モデル保存: {args.model_out}")
    print(f"チューニング後 重要度グラフ保存: {args.importance_out}")

    params_out = str(Path(args.model_out).with_suffix(".best_params.json"))
    with open(params_out, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"ベストパラメータ保存: {params_out}")


if __name__ == "__main__":
    main()

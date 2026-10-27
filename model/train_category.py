"""芝・ダート×距離帯の5カテゴリごとに別々のベースモデル(3着以内の二値分類)を学習・評価するCLI。

全カテゴリ共通の1つの分割(random/time)を作り、
  - カテゴリ別モデル: そのカテゴリの学習行だけで学習
  - 全体モデル    : 全カテゴリの学習行で学習
を同じテスト行で比較して、分割したことで精度・回収率が良くなったかを確認できるようにする。

例:
    python -m model.train_category --split time
    python -m model.train_category --categories turf_long --split time
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model.categories import CATEGORY_KEYS, CATEGORY_LABELS, CATEGORY_LGB_PARAMS, assign_categories
from model.common import _split_indices, get_feature_columns, plot_feature_importance, print_metrics, train_and_evaluate
from model.win_prob import fit_blend, save_calibration

DEFAULT_OUT_DIR = "data/models/category"


def _top1_stats(test_df, payouts_by_race):
    """各レースで予測確率1位の馬について、3着以内率と複勝回収率(100円ずつ)を返す。"""
    if test_df.empty:
        return {"races": 0, "top1_place_rate": float("nan"), "top1_fukusho_roi": float("nan")}
    top1 = test_df.sort_values("y_pred_proba", ascending=False).groupby("race_id").head(1)
    stats = {"races": int(len(top1)), "top1_place_rate": float(top1["y_true"].mean())}

    if payouts_by_race is None:
        stats["top1_fukusho_roi"] = float("nan")
        return stats

    from tickets import calculate_return

    bet_total = return_total = 0
    for _, row in top1.iterrows():
        payouts = payouts_by_race.get(row["race_id"])
        if not payouts or pd.isna(row["馬番"]):
            continue
        bets = [{"type": "複勝", "combo": str(int(round(row["馬番"]))), "amount": 100}]
        bet, ret, _, _ = calculate_return(bets, payouts)
        bet_total += bet
        return_total += ret
    stats["top1_fukusho_roi"] = round(return_total / bet_total * 100, 1) if bet_total else float("nan")
    return stats


def _auc(test_df):
    if test_df["y_true"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(test_df["y_true"], test_df["y_pred_proba"]))


def main():
    parser = argparse.ArgumentParser(description="芝・ダート×距離帯の5カテゴリ別にベースモデルを学習・評価する")
    parser.add_argument("--features-csv", default="data/processed/features.csv")
    parser.add_argument("--payouts-csv", default="data/processed/payouts.csv",
                         help="複勝回収率の算出に使う(ファイルが無ければ回収率はスキップ)")
    parser.add_argument("--split", choices=["random", "time"], default="time",
                         help="time: 日付順に後方をテストにする時系列分割(既定) / random: 層化ランダム分割")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--categories", nargs="+", choices=CATEGORY_KEYS, default=CATEGORY_KEYS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str})
    df = df[df["top3"].notna()].reset_index(drop=True)
    df["_category"] = assign_categories(df)
    print(f"読み込み: {df.shape[0]}行 (features-csv: {args.features_csv})")
    print(df["_category"].fillna("対象外(障害など)").value_counts().to_string())

    payouts_by_race = None
    if Path(args.payouts_csv).exists():
        from evaluation.roi import load_payouts
        payouts_by_race = load_payouts(args.payouts_csv)

    # 全カテゴリで共通の分割を作る(全体モデルとカテゴリ別モデルを同じテスト行で比べるため)。
    train_idx, test_idx = _split_indices(df, args.split, args.test_size, 42, df["top3"].astype(int))
    feature_cols = [c for c in get_feature_columns(df) if c != "_category"]

    print("\n=== 全体モデル(比較用) ===")
    global_result = train_and_evaluate(
        df, split=args.split, feature_cols=feature_cols, train_idx=train_idx, test_idx=test_idx,
    )
    global_test = global_result.test_df.join(df["_category"])

    summary_rows = []
    category_tests = []
    for key in args.categories:
        cat_df = df[df["_category"] == key]
        cat_train = cat_df.index.intersection(train_idx)
        cat_test = cat_df.index.intersection(test_idx)
        print(f"\n=== {key}: {CATEGORY_LABELS[key]} (学習{len(cat_train)}行 / 評価{len(cat_test)}行) ===")
        if len(cat_train) < 100 or len(cat_test) == 0:
            print("学習・評価データが不足しているためスキップします。")
            continue

        result = train_and_evaluate(
            cat_df,
            split=args.split,
            feature_cols=feature_cols,
            lgb_params=CATEGORY_LGB_PARAMS.get(key),
            train_idx=cat_train,
            test_idx=cat_test,
        )
        print_metrics(result.metrics, title=f"{key} モデル")
        category_tests.append(result.test_df)

        model_path = out_dir / f"{key}.txt"
        result.model.booster_.save_model(str(model_path))
        result.importance.to_csv(out_dir / f"{key}_importance.csv", index=False, encoding="utf-8-sig")
        plot_feature_importance(
            result.importance, out_dir / f"{key}_importance.png",
            title=f"{CATEGORY_LABELS[key]} 特徴量重要度 (Top20)",
        )

        cat_stats = _top1_stats(result.test_df, payouts_by_race)
        global_cat_test = global_test[global_test["_category"] == key]
        global_stats = _top1_stats(global_cat_test, payouts_by_race)

        metrics = dict(result.metrics)
        metrics.update({
            "category": key,
            "label": CATEGORY_LABELS[key],
            "split": args.split,
            "lgb_params": CATEGORY_LGB_PARAMS.get(key, {}),
            "top1_place_rate": cat_stats["top1_place_rate"],
            "top1_fukusho_roi": cat_stats["top1_fukusho_roi"],
            "global_model_auc": _auc(global_cat_test),
            "global_top1_place_rate": global_stats["top1_place_rate"],
            "global_top1_fukusho_roi": global_stats["top1_fukusho_roi"],
        })
        with open(out_dir / f"{key}.metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"モデル保存: {model_path}")
        print("--- 特徴量重要度 Top5 ---")
        print(result.importance.head(5).to_string(index=False))

        summary_rows.append({
            "category": key,
            "学習行数": metrics["n_train"],
            "評価行数": metrics["n_test"],
            "評価レース数": cat_stats["races"],
            "AUC(カテゴリ別)": round(metrics["auc"], 4),
            "AUC(全体)": round(metrics["global_model_auc"], 4),
            "1位馬3着内率(カテゴリ別)": round(cat_stats["top1_place_rate"], 3),
            "1位馬3着内率(全体)": round(global_stats["top1_place_rate"], 3),
            "1位馬複勝回収率%(カテゴリ別)": cat_stats["top1_fukusho_roi"],
            "1位馬複勝回収率%(全体)": global_stats["top1_fukusho_roi"],
        })

    # 推定勝率の補正係数: カテゴリ別モデルの検証データ(学習に使っていない行)で、
    # モデル確率と単勝オッズの組み合わせ方を推定する(model.win_prob)。
    if category_tests:
        tests = pd.concat(category_tests)
        calibration = fit_blend(
            tests["y_pred_proba"], tests["オッズ"], (df.loc[tests.index, "着順_数値"] == 1).astype(float), tests["race_id"],
        )
        if calibration:
            save_calibration(calibration, out_dir / "win_calibration.json")
            print("\n=== 推定勝率の補正係数 (q ∝ p_model^a × p_market^b) ===")
            print(f"a(モデル)={calibration['a']:.3f}, b(オッズ)={calibration['b']:.3f} / {calibration['races']}レース")
            print(f"1着馬の平均対数損失: 補正後 {calibration['nll_blend']:.4f} / モデルのみ {calibration['nll_model_only']:.4f}"
                  f" / オッズのみ {calibration['nll_market_only']:.4f} (小さいほど良い)")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / "comparison.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n=== カテゴリ別モデル vs 全体モデル (同一テスト行) ===")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary.to_string(index=False))
        print(f"比較表保存: {summary_path}")
        if (summary["評価レース数"] < 50).any():
            print("※評価レース数が50未満のカテゴリは指標のぶれが大きいので参考程度にしてください。")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

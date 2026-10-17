"""季節適性特徴量の効果検証(アブレーション)。

既存特徴量のみのモデル(baseline)と、季節適性特徴量を加えたモデル(season)を、
日付単位の時系列分割(ウォークフォワード)で学習・評価して比較する。
モデルのハイパーパラメータは model.common.train_and_evaluate と同じ。保存済みモデルは上書きしない。

例:
    python -m model.season_ablation
    python -m model.season_ablation --folds 4 --bootstrap 1000
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from model.common import _to_numeric_matrix, get_feature_columns
from preprocessing.build_features import build_feature_dataset
from preprocessing.season import SEASON_FEATURE_COLUMNS

SEASON_NAMES = {0.0: "春", 1.0: "夏", 2.0: "秋", 3.0: "冬"}
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1,
}


def _date_folds(dates, n_folds, test_share):
    """ユニーク日付の後方test_share分をn_folds個に分け、各ブロックをテスト・それより前を学習にする。
    同じ日のレースが学習とテストに分かれないよう、日付単位で区切る。"""
    unique = np.sort(dates.dropna().unique())
    start = int(len(unique) * (1 - test_share))
    blocks = np.array_split(unique[start:], n_folds)
    for block in blocks:
        yield dates < block[0], dates.isin(block)


def _metrics(y, p, frame):
    out = {
        "n": int(len(y)),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
        "mean_pred": float(np.mean(p)),
        "mean_true": float(np.mean(y)),
    }
    # レースごとに予測確率最大の馬を本命とした場合の単勝的中率・複勝(3着以内)率
    tmp = frame.assign(p=p)
    top = tmp.loc[tmp.groupby("race_id")["p"].idxmax()]
    out["races"] = int(len(top))
    out["top_pick_win"] = float((top["着順_数値"] == 1).mean())
    out["top_pick_top3"] = float((top["着順_数値"] <= 3).mean())
    return out


def _experience_bucket(df):
    total_prior = df[[f"着順_{i}" for i in range(1, 6)]].notna().sum(axis=1)
    n = df["季節_出走数"]
    return np.select(
        [total_prior == 0, n == 0, df["季節_前年以前出走数"] > 0, n == 1],
        ["初出走(履歴なし)", "当季節の履歴なし", "前年以前の同季節あり", "当季節1走"],
        default="当季節2走以上(同年のみ)",
    )


def _paired_bootstrap(pred, n_boot, seed=0):
    """レース単位の対応ありブートストラップで、logloss/Brierの差(season - baseline)の95%区間を求める。"""
    rng = np.random.default_rng(seed)
    eps = 1e-15
    y = pred["y"].to_numpy()
    ll = {}
    br = {}
    for name in ("baseline", "season"):
        p = np.clip(pred[name].to_numpy(), eps, 1 - eps)
        ll[name] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        br[name] = (p - y) ** 2
    per_race = pd.DataFrame({
        "race_id": pred["race_id"].to_numpy(),
        "d_ll": ll["season"] - ll["baseline"],
        "d_br": br["season"] - br["baseline"],
    }).groupby("race_id").agg(["sum", "count"])
    sums_ll = per_race[("d_ll", "sum")].to_numpy()
    sums_br = per_race[("d_br", "sum")].to_numpy()
    counts = per_race[("d_ll", "count")].to_numpy()
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    boot_ll = sums_ll[idx].sum(1) / counts[idx].sum(1)
    boot_br = sums_br[idx].sum(1) / counts[idx].sum(1)
    return {
        "logloss_diff": float(sums_ll.sum() / counts.sum()),
        "logloss_diff_ci95": [float(np.percentile(boot_ll, 2.5)), float(np.percentile(boot_ll, 97.5))],
        "brier_diff": float(sums_br.sum() / counts.sum()),
        "brier_diff_ci95": [float(np.percentile(boot_br, 2.5)), float(np.percentile(boot_br, 97.5))],
    }


def run_ablation(df, n_folds=4, test_share=0.5, n_boot=1000, target_col="top3"):
    working = df[df[target_col].notna()].reset_index(drop=True)
    y_all = working[target_col].astype(int)
    dates = pd.to_datetime(working["日付"], format="%Y/%m/%d", errors="coerce")

    all_cols = get_feature_columns(working)
    feature_sets = {
        "baseline": [c for c in all_cols if c not in SEASON_FEATURE_COLUMNS],
        "season": all_cols,
    }
    missing = [c for c in SEASON_FEATURE_COLUMNS if c not in all_cols]
    if missing:
        raise ValueError(f"季節特徴量が見つかりません: {missing}")

    preds = []
    importances = []
    for fold, (train_mask, test_mask) in enumerate(_date_folds(dates, n_folds, test_share), start=1):
        fold_pred = working.loc[test_mask, ["race_id", "日付", "季節", "着順_数値"]].copy()
        fold_pred["y"] = y_all[test_mask].to_numpy()
        fold_pred["fold"] = fold
        fold_pred["bucket"] = _experience_bucket(working.loc[test_mask])
        for name, cols in feature_sets.items():
            x = _to_numeric_matrix(working, cols)
            model = lgb.LGBMClassifier(**LGB_PARAMS)
            model.fit(x[train_mask], y_all[train_mask])
            fold_pred[name] = model.predict_proba(x[test_mask])[:, 1]
            if name == "season":
                importances.append(pd.Series(
                    model.booster_.feature_importance(importance_type="gain"), index=cols))
        print(f"fold {fold}: train {int(train_mask.sum())}行 (〜{dates[train_mask].max():%Y-%m-%d}) / "
              f"test {int(test_mask.sum())}行 ({dates[test_mask].min():%Y-%m-%d}〜{dates[test_mask].max():%Y-%m-%d})")
        preds.append(fold_pred)
    pred = pd.concat(preds, ignore_index=True)

    def table(group_col=None):
        rows = []
        groups = [("全体", pred)] if group_col is None else pred.groupby(group_col)
        for key, g in groups:
            for name in feature_sets:
                m = _metrics(g["y"], g[name], g)
                rows.append({"group": key, "model": name, **m})
        return pd.DataFrame(rows)

    imp = pd.concat(importances, axis=1).mean(axis=1)
    imp_share = (imp / imp.sum()).sort_values(ascending=False)
    season_imp = imp_share[SEASON_FEATURE_COLUMNS].sort_values(ascending=False)

    return {
        "overall": table(),
        "by_fold": table("fold"),
        "by_season": table("季節").assign(group=lambda t: t["group"].map(SEASON_NAMES)),
        "by_bucket": table("bucket"),
        "bootstrap": _paired_bootstrap(pred, n_boot),
        "season_importance_share": season_imp,
        "season_importance_total": float(season_imp.sum()),
        "predictions": pred,
    }


def _print_table(title, t):
    cols = ["group", "model", "n", "logloss", "brier", "auc", "top_pick_win", "top_pick_top3"]
    print(f"\n--- {title} ---")
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser(description="季節適性特徴量の有無でモデル性能を比較する")
    parser.add_argument("--results-csv", default="data/processed/race_results.csv")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--test-share", type=float, default=0.5,
                        help="ユニーク開催日のうち後方何割をウォークフォワードのテスト期間にするか")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-json", default="data/models/season_ablation.json")
    args = parser.parse_args()

    df = build_feature_dataset(args.results_csv, include_season=True)
    print(f"特徴量データ: {df.shape[0]}行 / {df.shape[1]}列")
    result = run_ablation(df, n_folds=args.folds, test_share=args.test_share, n_boot=args.bootstrap)

    _print_table("全テスト期間(ウォークフォワード合算)", result["overall"])
    _print_table("fold別", result["by_fold"])
    _print_table("レース季節別", result["by_season"])
    _print_table("出走履歴別", result["by_bucket"])
    b = result["bootstrap"]
    print("\n--- season - baseline の差 (負なら季節特徴量ありが良い, レース単位ブートストラップ95%区間) ---")
    print(f"logloss: {b['logloss_diff']:+.5f}  CI95 [{b['logloss_diff_ci95'][0]:+.5f}, {b['logloss_diff_ci95'][1]:+.5f}]")
    print(f"brier  : {b['brier_diff']:+.5f}  CI95 [{b['brier_diff_ci95'][0]:+.5f}, {b['brier_diff_ci95'][1]:+.5f}]")
    print(f"\n季節特徴量のgain重要度シェア合計: {result['season_importance_total']:.3%}")
    print(result["season_importance_share"].to_string(float_format=lambda v: f"{v:.3%}"))

    out = {
        key: (val.to_dict(orient="records") if isinstance(val, pd.DataFrame)
              else val.to_dict() if isinstance(val, pd.Series) else val)
        for key, val in result.items() if key != "predictions"
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n結果保存: {args.output_json}")


if __name__ == "__main__":
    main()

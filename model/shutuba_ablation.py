"""出馬表モデル(model.shutuba_model)の追加特徴量の効果検証(アブレーション)。

既存特徴量のみ(baseline)と、FEATURE_GROUPS の各グループを足したモデルを、本番と同じ手順
(shutuba_model.fit_model)で学習して比べる。保存済みモデルは上書きしない。

評価期間は2つ:
  - テスト期間        : data/test のレース(モデルの採用判定に使っている期間)
  - 学習期間の最後1年 : data/train の最後の1年をテスト、それより前を学習にした場合(レース数が多く差を見やすい)

LightGBMは乱数(行・列のサンプリング)で結果が少し揺れるため、複数シードの平均で比べる。
baselineとの差は、レース単位の対応ありブートストラップで95%区間を出す(0をまたぐなら誤差の範囲)。

例:
    python -m model.shutuba_ablation
    python -m model.shutuba_ablation --seeds 5 --bootstrap 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from model.shutuba_model import (
    ALL_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    FEATURE_GROUPS,
    _pick_stats,
    build_training_table,
    fit_model,
)

BASELINE = "baseline"
ALL_GROUPS = "+全グループ"
PICK_COLUMNS = ["単勝的中率", "単勝回収率", "複勝的中率", "複勝回収率"]


def feature_sets():
    sets = {BASELINE: BASE_FEATURE_COLUMNS}
    for name, cols in FEATURE_GROUPS.items():
        sets[f"+{name}"] = BASE_FEATURE_COLUMNS + cols
    sets[ALL_GROUPS] = ALL_FEATURE_COLUMNS
    return sets


def evaluation_periods(df):
    """{期間名: (学習行, テスト行)}。人気が欠けている行はテストから外す(採用判定と同じ扱い)。"""
    is_train = (df["split"] == "train").to_numpy()
    has_popularity = pd.to_numeric(df["人気"], errors="coerce").notna().to_numpy()
    last_year = (df["_date"] >= df.loc[is_train, "_date"].max() - pd.Timedelta(days=365)).to_numpy()
    return {
        "テスト期間": (is_train, ~is_train & has_popularity),
        "学習期間の最後1年": (is_train & ~last_year, is_train & last_year & has_popularity),
    }


def _auc(y, score):
    ranks = rankdata(score)
    n_pos = y.sum()
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos))


def _logloss_rows(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _metrics(test, proba):
    y = test["top3"].to_numpy()
    picks = _pick_stats(test.assign(_proba=proba), "_proba")
    return {"AUC": _auc(y, proba), "logloss": _logloss_rows(y, proba).mean(), **{c: picks[c] for c in PICK_COLUMNS}}


def _paired_bootstrap(test, base_proba, new_proba, n_boot, seed=0):
    """レース単位で復元抽出し、(追加あり - baseline) の AUC・logloss の差の95%区間を返す。"""
    rng = np.random.default_rng(seed)
    y = test["top3"].to_numpy()
    codes, _ = pd.factorize(test["race_id"])
    sizes = np.bincount(codes)
    race_rows = np.split(np.argsort(codes, kind="stable"), np.cumsum(sizes)[:-1])
    d_ll = np.bincount(codes, weights=_logloss_rows(y, new_proba) - _logloss_rows(y, base_proba))

    d_auc_boot, d_ll_boot = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(sizes), len(sizes))
        rows = np.concatenate([race_rows[i] for i in pick])
        d_auc_boot.append(_auc(y[rows], new_proba[rows]) - _auc(y[rows], base_proba[rows]))
        d_ll_boot.append(d_ll[pick].sum() / sizes[pick].sum())
    return {
        "AUC差": float(_auc(y, new_proba) - _auc(y, base_proba)),
        "AUC差_CI95": [float(v) for v in np.percentile(d_auc_boot, [2.5, 97.5])],
        "logloss差": float(d_ll.sum() / sizes.sum()),
        "logloss差_CI95": [float(v) for v in np.percentile(d_ll_boot, [2.5, 97.5])],
    }


def run_ablation(df, x, seeds=(42, 43, 44), n_boot=1000):
    y = df["top3"]
    sets = feature_sets()
    results = {}
    for period, (train_mask, test_mask) in evaluation_periods(df).items():
        test = df[test_mask]
        print(f"\n[{period}] 学習 {df.loc[train_mask, 'race_id'].nunique()}レース "
              f"(〜{df.loc[train_mask, '_date'].max():%Y-%m-%d}) / テスト {test['race_id'].nunique()}レース "
              f"({test['_date'].min():%Y-%m-%d}〜{test['_date'].max():%Y-%m-%d})", flush=True)

        rows, mean_proba, importance = [], {}, None
        for name, cols in sets.items():
            per_seed, probas, trees, gains = [], [], [], []
            for seed in seeds:
                model, best_iter = fit_model(x[cols], y, df["_date"], train_mask, seed=seed)
                proba = model.predict_proba(x.loc[test_mask, cols])[:, 1]
                per_seed.append(_metrics(test, proba))
                probas.append(proba)
                trees.append(best_iter)
                gains.append(pd.Series(model.booster_.feature_importance("gain"), index=cols))
            mean_proba[name] = np.mean(probas, axis=0)
            per_seed = pd.DataFrame(per_seed)
            rows.append({
                "特徴量セット": name, "特徴量数": len(cols), "木の本数": float(np.mean(trees)),
                **per_seed.mean().to_dict(), "AUC_シード間SD": float(per_seed["AUC"].std(ddof=0)),
            })
            if name == ALL_GROUPS:
                gain = pd.concat(gains, axis=1).mean(axis=1)
                importance = (gain / gain.sum() * 100).sort_values(ascending=False)
            print(f"  {name}: AUC {rows[-1]['AUC']:.4f} / logloss {rows[-1]['logloss']:.4f}", flush=True)

        for row in rows[1:]:
            row.update(_paired_bootstrap(test, mean_proba[BASELINE], mean_proba[row["特徴量セット"]], n_boot))

        popularity_score = -pd.to_numeric(test["人気"], errors="coerce")
        results[period] = {
            "table": pd.DataFrame(rows),
            "1番人気": {"AUC": float(_auc(test["top3"].to_numpy(), popularity_score.to_numpy())),
                      **_pick_stats(test.assign(_score=popularity_score), "_score")},
            "importance": importance,
            "レース数": int(test["race_id"].nunique()),
        }
    return results


def _ci(values):
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def print_results(results, seeds):
    for period, res in results.items():
        table = res["table"]
        print(f"\n=== {period} ({res['レース数']}レース, シード{len(seeds)}個の平均) ===")
        show = table[["特徴量セット", "特徴量数", "AUC", "logloss"] + PICK_COLUMNS].copy()
        print(show.to_string(index=False, formatters={
            "AUC": "{:.4f}".format, "logloss": "{:.4f}".format,
            "単勝的中率": "{:.1%}".format, "複勝的中率": "{:.1%}".format,
            "単勝回収率": "{:.1f}".format, "複勝回収率": "{:.1f}".format,
        }))
        fav = res["1番人気"]
        print(f"(参考) 1番人気: AUC {fav['AUC']:.4f} / 単勝 {fav['単勝的中率']:.1%}・{fav['単勝回収率']}% / "
              f"複勝 {fav['複勝的中率']:.1%}・{fav['複勝回収率']}%")
        print("--- baselineとの差 (AUCは正・loglossは負なら追加ありが良い。[]は95%区間) ---")
        for _, row in table.iloc[1:].iterrows():
            print(f"  {row['特徴量セット']}: AUC {row['AUC差']:+.4f} {_ci(row['AUC差_CI95'])} / "
                  f"logloss {row['logloss差']:+.4f} {_ci(row['logloss差_CI95'])}")

    importance = results["テスト期間"]["importance"]
    added = importance.drop(BASE_FEATURE_COLUMNS)
    print(f"\n--- {ALL_GROUPS} モデルでの追加特徴量の重要度(gain%, 合計 {added.sum():.1f}%) ---")
    print(added.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description="出馬表モデルの追加特徴量の効果を比較する")
    parser.add_argument("--seeds", type=int, default=3, help="乱数シードの個数(42から連番)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-json", default="data/models/shutuba_ablation.json")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    df, x = build_training_table()
    print(f"学習表: {len(df)}頭 / 特徴量候補 {x.shape[1]}個")
    results = run_ablation(df, x, seeds=seeds, n_boot=args.bootstrap)
    print_results(results, seeds)

    out = {"seeds": seeds, "bootstrap": args.bootstrap, "特徴量グループ": FEATURE_GROUPS}
    for period, res in results.items():
        out[period] = {
            "レース数": res["レース数"],
            "比較": res["table"].to_dict(orient="records"),
            "1番人気": res["1番人気"],
            "特徴量重要度(gain%)": res["importance"].round(2).to_dict(),
        }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n結果保存: {args.output_json}")


if __name__ == "__main__":
    main()

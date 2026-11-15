"""出馬表モデル(model.shutuba_model)のノイズ特徴量の選別。

特徴量候補すべて(ALL_FEATURE_COLUMNS)から始めて、効いていない特徴量を外していく。保存済みモデルは上書きしない。

手順:
  1. 学習期間の最後3年を1年ずつテストにするウォークフォワードで学習する(テスト期間のレースは選別に使わない)
  2. 各特徴量の値をテスト側でシャッフルし、loglossがどれだけ悪化するか(permutation importance)を測る
  3. 悪化しない特徴量(平均が0以下、または過半数の年で0以下)をノイズ候補にして外し、学習し直す
  4. 外した後が、全特徴量のモデルより有意に悪くなっていなければ採用(悪ければ候補を半分に減らしてやり直す)
  5. 候補がなくなるまで 2〜4 を繰り返す
  6. 最後に、選別に使っていないテスト期間で「既存26個 / 候補すべて / 選別後」を比べて答え合わせをする

結果(外す特徴量)は shutuba_model.DROPPED_FEATURES に手で反映する。

例:
    python -m model.shutuba_feature_selection
    python -m model.shutuba_feature_selection --seeds 3 --repeats 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.shutuba_ablation import PICK_COLUMNS, _logloss_rows, _metrics, _paired_bootstrap, evaluation_periods
from model.shutuba_model import ALL_FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, build_training_table, fit_model


def walk_forward_folds(df, n_folds=3):
    """学習期間(split=train)の最後 n_folds 年を、古い順に1年ずつテストにする [(学習行, テスト行)]。"""
    is_train = (df["split"] == "train").to_numpy()
    has_popularity = pd.to_numeric(df["人気"], errors="coerce").notna().to_numpy()
    end = df.loc[is_train, "_date"].max()
    folds = []
    for k in range(n_folds, 0, -1):
        start, stop = end - pd.Timedelta(days=365 * k), end - pd.Timedelta(days=365 * (k - 1))
        in_block = ((df["_date"] > start) & (df["_date"] <= stop)).to_numpy()
        folds.append((is_train & (df["_date"] <= start).to_numpy(), is_train & in_block & has_popularity))
    return folds


def fit_folds(df, x, cols, folds, seeds, n_repeats, rng):
    """各foldで学習し、(テスト行を連結した予測確率[シード平均], permutation importance[特徴量×fold])を返す。

    importance は「その特徴量をテスト側でシャッフルしたときの logloss の悪化量」。0以下なら効いていない。
    """
    y = df["top3"]
    probas, importances = [], []
    for train_mask, test_mask in folds:
        x_test = x.loc[test_mask, cols].to_numpy(dtype=float)
        y_test = y[test_mask].to_numpy()
        proba = np.zeros(len(y_test))
        importance = np.zeros(len(cols))
        for seed in seeds:
            model, _ = fit_model(x[cols], y, df["_date"], train_mask, seed=seed)
            booster = model.booster_
            seed_proba = booster.predict(x_test)
            proba += seed_proba / len(seeds)
            base = _logloss_rows(y_test, seed_proba).mean()
            for j in range(len(cols)):
                original = x_test[:, j].copy()
                for _ in range(n_repeats):
                    x_test[:, j] = rng.permutation(original)
                    importance[j] += _logloss_rows(y_test, booster.predict(x_test)).mean() - base
                x_test[:, j] = original
        probas.append(proba)
        importances.append(pd.Series(importance / (len(seeds) * n_repeats), index=cols))
    return np.concatenate(probas), pd.concat(importances, axis=1)


def noise_candidates(importance):
    """シャッフルしても悪化しない特徴量を、効いていない順に返す。"""
    mean = importance.mean(axis=1)
    positive_folds = (importance > 0).sum(axis=1)
    is_noise = (mean <= 0) | (positive_folds * 2 < importance.shape[1])
    return list(mean[is_noise].sort_values().index)


def select_features(df, x, seeds=(42, 43), n_repeats=3, n_boot=1000, max_rounds=8):
    folds = walk_forward_folds(df)
    pooled = pd.concat([df[test_mask] for _, test_mask in folds])
    y = pooled["top3"].to_numpy()
    rng = np.random.default_rng(0)
    for i, (train_mask, test_mask) in enumerate(folds, start=1):
        print(f"fold {i}: 学習 〜{df.loc[train_mask, '_date'].max():%Y-%m-%d} "
              f"({df.loc[train_mask, 'race_id'].nunique()}レース) / テスト {df.loc[test_mask, '_date'].min():%Y-%m-%d}"
              f"〜{df.loc[test_mask, '_date'].max():%Y-%m-%d} ({df.loc[test_mask, 'race_id'].nunique()}レース)")

    current = list(ALL_FEATURE_COLUMNS)
    full_proba, importance = fit_folds(df, x, current, folds, seeds, n_repeats, rng)
    first_importance = importance
    print(f"全{len(current)}特徴量: logloss {_logloss_rows(y, full_proba).mean():.5f}", flush=True)

    rounds = []
    for round_no in range(1, max_rounds + 1):
        drop = noise_candidates(importance)
        accepted = False
        while drop and not accepted:
            trial = [c for c in current if c not in drop]
            trial_proba, trial_importance = fit_folds(df, x, trial, folds, seeds, n_repeats, rng)
            diff = _paired_bootstrap(pooled, full_proba, trial_proba, n_boot)
            # 全特徴量モデルより有意に悪い(logloss差の95%区間が0より上)ときだけ却下する
            accepted = diff["logloss差_CI95"][0] <= 0
            rounds.append({"round": round_no, "外した特徴量": drop, "残り": len(trial),
                           "logloss": float(_logloss_rows(y, trial_proba).mean()), "採用": accepted, **diff})
            print(f"round {round_no}: {len(drop)}個を外す → 残り{len(trial)}個 / 全特徴量との差 "
                  f"logloss {diff['logloss差']:+.5f} [{diff['logloss差_CI95'][0]:+.5f}, {diff['logloss差_CI95'][1]:+.5f}] "
                  f"AUC {diff['AUC差']:+.5f} → {'採用' if accepted else '却下'}\n    {drop}", flush=True)
            if accepted:
                current, importance = trial, trial_importance
            else:
                drop = drop[:len(drop) // 2]
        if not accepted:
            break
    return {
        "selected": current,
        "dropped": [c for c in ALL_FEATURE_COLUMNS if c not in current],
        "rounds": rounds,
        "first_importance": first_importance,
        "final_importance": importance,
    }


def confirm_on_test(df, x, selected, seeds=(42, 43, 44), n_boot=1000):
    """選別に使っていないテスト期間で、既存26個・候補すべて・選別後を比べる。"""
    train_mask, test_mask = evaluation_periods(df)["テスト期間"]
    test = df[test_mask]
    sets = {"既存": BASE_FEATURE_COLUMNS, "候補すべて": ALL_FEATURE_COLUMNS, "選別後": selected}
    rows, mean_proba = [], {}
    for name, cols in sets.items():
        per_seed, probas = [], []
        for seed in seeds:
            model, _ = fit_model(x[cols], df["top3"], df["_date"], train_mask, seed=seed)
            probas.append(model.predict_proba(x.loc[test_mask, cols])[:, 1])
            per_seed.append(_metrics(test, probas[-1]))
        mean_proba[name] = np.mean(probas, axis=0)
        rows.append({"特徴量セット": name, "特徴量数": len(cols), **pd.DataFrame(per_seed).mean().to_dict()})
    diffs = {f"選別後 - {name}": _paired_bootstrap(test, mean_proba[name], mean_proba["選別後"], n_boot)
             for name in ("既存", "候補すべて")}
    return {"table": pd.DataFrame(rows), "diffs": diffs, "レース数": int(test["race_id"].nunique())}


def _print_importance(title, importance):
    table = pd.DataFrame({"平均": importance.mean(axis=1), "悪化した年数": (importance > 0).sum(axis=1)})
    print(f"\n--- {title} (シャッフル時のlogloss悪化量 ×1e4。0以下は効いていない) ---")
    print((table.assign(平均=table["平均"] * 1e4).sort_values("平均", ascending=False)
           .to_string(formatters={"平均": "{:+.2f}".format})))


def main():
    parser = argparse.ArgumentParser(description="出馬表モデルのノイズ特徴量を選別する")
    parser.add_argument("--seeds", type=int, default=2, help="選別時の乱数シードの個数(42から連番)")
    parser.add_argument("--repeats", type=int, default=3, help="1特徴量あたりのシャッフル回数")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-json", default="data/models/shutuba_feature_selection.json")
    args = parser.parse_args()

    df, x = build_training_table()
    seeds = list(range(42, 42 + args.seeds))
    result = select_features(df, x, seeds=seeds, n_repeats=args.repeats, n_boot=args.bootstrap)
    _print_importance("選別前の重要度", result["first_importance"])
    _print_importance("選別後の重要度", result["final_importance"])
    print(f"\n外す特徴量 ({len(result['dropped'])}個): {result['dropped']}")
    print(f"残す特徴量 ({len(result['selected'])}個): {result['selected']}")

    confirm = confirm_on_test(df, x, result["selected"], n_boot=args.bootstrap)
    print(f"\n=== 答え合わせ: テスト期間 ({confirm['レース数']}レース, 選別には未使用, シード3個の平均) ===")
    print(confirm["table"].to_string(index=False, formatters={
        "AUC": "{:.4f}".format, "logloss": "{:.4f}".format,
        "単勝的中率": "{:.1%}".format, "複勝的中率": "{:.1%}".format,
        "単勝回収率": "{:.1f}".format, "複勝回収率": "{:.1f}".format,
    }))
    for name, d in confirm["diffs"].items():
        print(f"  {name}: AUC {d['AUC差']:+.4f} [{d['AUC差_CI95'][0]:+.4f}, {d['AUC差_CI95'][1]:+.4f}] / "
              f"logloss {d['logloss差']:+.4f} [{d['logloss差_CI95'][0]:+.4f}, {d['logloss差_CI95'][1]:+.4f}]")

    out = {
        "seeds": seeds, "repeats": args.repeats,
        "外す特徴量": result["dropped"], "残す特徴量": result["selected"], "rounds": result["rounds"],
        "選別前の重要度(logloss悪化量)": result["first_importance"].mean(axis=1).sort_values(ascending=False).to_dict(),
        "選別後の重要度(logloss悪化量)": result["final_importance"].mean(axis=1).sort_values(ascending=False).to_dict(),
        "テスト期間": {"比較": confirm["table"].to_dict(orient="records"), "差": confirm["diffs"]},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n結果保存: {args.output_json}")


if __name__ == "__main__":
    main()

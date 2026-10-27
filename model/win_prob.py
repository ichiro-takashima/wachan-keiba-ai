"""3着以内確率(モデル)と単勝オッズ(市場)を組み合わせて、1着になる確率を推定する。

モデルは3着以内の二値分類なので、そのまま全頭合計で割ると人気薄の勝率を過大評価し、
期待値(勝率×オッズ)が実際の回収率より大きく出てしまう。そこで
    q_i ∝ p_model_i^a × p_market_i^b   (p_market = 1/オッズ をレース内で正規化)
の形(条件付きロジット)で、検証データの実際の1着馬に対する尤度が最大になる a, b を推定する。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CALIBRATION_PATH = Path("data/models/category/win_calibration.json")


def _market_prob(odds, race_ids):
    inv = 1.0 / odds
    return inv / inv.groupby(race_ids).transform("sum")


def blend_win_probability(p_model, odds, race_ids, a, b):
    p_model = pd.Series(p_model).clip(1e-6, 1 - 1e-6)
    odds = pd.to_numeric(pd.Series(odds, index=p_model.index), errors="coerce")
    race_ids = pd.Series(race_ids, index=p_model.index)
    log_score = a * np.log(p_model) + b * np.log(_market_prob(odds, race_ids))
    score = np.exp(log_score - log_score.groupby(race_ids).transform("max"))
    return score / score.groupby(race_ids).transform("sum")


def fit_blend(p_model, odds, is_win, race_ids):
    """1着馬が1頭だけ確定しているレースで a, b を最尤推定し、評価値と一緒に返す。"""
    d = pd.DataFrame({"p": p_model, "odds": pd.to_numeric(odds, errors="coerce"), "win": is_win, "race": race_ids})
    d = d[d["p"].notna() & d["odds"].notna() & (d["odds"] > 0)]
    d = d[d.groupby("race")["win"].transform("sum") == 1]
    if d["race"].nunique() < 30:
        return None

    def nll(w, frame):
        q = blend_win_probability(frame["p"], frame["odds"], frame["race"], *w)
        return float(-np.log(q[frame["win"] == 1].clip(1e-12)).mean())

    res = minimize(lambda w: nll(w, d), x0=[1.0, 1.0], method="Nelder-Mead")
    # aが負(モデル評価が高いほど勝率を下げる)になるのは不自然なので0で止める。
    a, b = max(float(res.x[0]), 0.0), float(res.x[1])
    return {
        "a": a,
        "b": b,
        "races": int(d["race"].nunique()),
        "nll_blend": nll((a, b), d),
        "nll_model_only": nll((1.0, 0.0), d),
        "nll_market_only": nll((0.0, 1.0), d),
    }


def load_calibration(path=CALIBRATION_PATH):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_calibration(calibration, path=CALIBRATION_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)

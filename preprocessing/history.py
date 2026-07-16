"""馬ごとの直近5走履歴・騎手勝率・平均斤量などの特徴量を作る。

すべて「対象レースより前の情報のみ」を参照するため、リーク(未来の情報を使うこと)は起きない。
"""
import numpy as np
import pandas as pd

from preprocessing.encode import parse_rank

HISTORY_SOURCE_COLUMNS = [
    "馬番", "騎手", "斤量", "オッズ", "体重", "体重変化", "上がり", "通過順",
    "着順", "距離", "クラス", "走破時間", "芝・ダート", "天気", "馬場",
]
HISTORY_WINDOW = 5


def _prepare_sorted(df):
    working = df.copy()
    working["_date"] = pd.to_datetime(working["日付"], format="%Y/%m/%d", errors="coerce")
    working["_rank_num"] = working["着順"].map(parse_rank)
    working = working.sort_values(["馬", "_date"]).reset_index(drop=True)
    return working


def add_last5_history(df):
    """直近5走の馬番/騎手/斤量/オッズ/体重/体重変化/上がり/通過順/着順/距離/クラス/
    走破時間/芝・ダート/天気/馬場/日付差/距離差 を _1〜_5 のカラムとして追加する。

    馬ごとに日付順へ並べた上でgroupby().shift()を使うベクトル化実装。
    行ごとのループより大幅に高速で、全レース分の本格データにも耐えられる。
    """
    working = _prepare_sorted(df)
    grouped = working.groupby("馬", sort=False)

    for col in HISTORY_SOURCE_COLUMNS:
        for i in range(1, HISTORY_WINDOW + 1):
            working[f"{col}_{i}"] = grouped[col].shift(i)

    for i in range(1, HISTORY_WINDOW + 1):
        # i=1のときは対象レース自身、i>=2のときは1つ前の過去走が基準になる
        base_date = working["_date"] if i == 1 else grouped["_date"].shift(i - 1)
        base_distance = working["距離"] if i == 1 else grouped["距離"].shift(i - 1)
        slot_date = grouped["_date"].shift(i)
        slot_distance = grouped["距離"].shift(i)
        working[f"日付差_{i}"] = (base_date - slot_date).dt.days
        working[f"距離差_{i}"] = base_distance - slot_distance

    return working.drop(columns=["_date", "_rank_num"])


def add_jockey_win_rate(df):
    """騎手ごとの勝率(対象レースより前のみで集計)を追加する。"""
    working = df.copy()
    working["_date"] = pd.to_datetime(working["日付"], format="%Y/%m/%d", errors="coerce")
    working["_rank_num"] = working["着順"].map(parse_rank)
    working["_is_win"] = (working["_rank_num"] == 1).astype(float)

    working = working.sort_values(["騎手", "_date"]).reset_index(drop=True)
    win_rates = pd.Series(np.nan, index=working.index)
    for _, group in working.groupby("騎手", sort=False):
        wins_so_far = group["_is_win"].cumsum().shift(1)
        races_so_far = pd.Series(range(len(group)), index=group.index)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = wins_so_far / races_so_far
        win_rates.loc[group.index] = rate

    working["騎手勝率"] = win_rates
    return working.drop(columns=["_date", "_rank_num", "_is_win"])


def add_avg_weight_carried(df):
    """直近5走(斤量_1〜_5)の平均斤量を追加する。history.add_last5_historyの後に呼ぶこと。"""
    working = df.copy()
    weight_cols = [f"斤量_{i}" for i in range(1, HISTORY_WINDOW + 1) if f"斤量_{i}" in working.columns]
    working["平均斤量"] = working[weight_cols].mean(axis=1) if weight_cols else np.nan
    return working

"""馬ごとの季節適性(季節別成績)の特徴量を作る。

すべて「対象レースより前の日付に行われたレースの成績のみ」から集計する(リーク防止)。
全期間で季節別成績を先に集計して結合する方式は採らず、馬ごとに日付順の累積和を
1つずらして(shift)使う。

注意:
- 季節は開催月からの暦上の分類であり、気温・湿度などの観測値ではない。
- 季節別成績の差は観察データ上の差であり、季節が原因で能力が変化したことを意味しない。
  人気との差(市場評価との乖離)や頭数補正で交絡をある程度緩和するが、因果効果の推定ではない。
"""
import numpy as np
import pandas as pd

from preprocessing.encode import parse_rank

# 季節の分類(設定で変更可)。キーの並び順が季節コード(0,1,2,...)になる。
# 各タプルは「季節内の月を時系列順」に並べること(年をまたぐ季節の年度判定に使う)。
DEFAULT_SEASON_MONTHS = {
    "春": (3, 4, 5),
    "夏": (6, 7, 8),
    "秋": (9, 10, 11),
    "冬": (12, 1, 2),
}

# 平滑化の強さ(擬似出走数)。季節別成績はk走分だけ馬自身の通年成績に、
# 通年成績はk走分だけ「平均的な馬(期待値との差0)」に引き寄せる。
DEFAULT_PRIOR_STRENGTH = 3.0

# 直近成績として使う過去エントリ数
RECENT_WINDOW = 3

# 着順に含まれる「出走したが完走・入線扱いにならなかった」印(最下位扱い)と、
# 「出走していない」印(集計から除外)。
DNF_MARKS = ("中", "失")
NOT_STARTED_MARKS = ("取", "除")

SEASON_FEATURE_COLUMNS = [
    "季節",
    "季節_出走数",
    "季節_勝率",
    "季節_連対率",
    "季節_複勝率",
    "季節_平均着順比",
    "季節_着順比_標準偏差",
    "季節_複勝超過_相対",
    "季節_人気差_相対",
    "季節_信頼度",
    "季節_前年以前出走数",
    "季節_前年以前_人気差_相対",
    "季節_直近複合",
]


def build_month_lookup(season_months=None):
    """月→(季節コード, 季節内の開始月)の辞書を返す。1〜12月がちょうど1回ずつ現れることを検証する。"""
    season_months = season_months or DEFAULT_SEASON_MONTHS
    lookup = {}
    for code, (name, months) in enumerate(season_months.items()):
        if not months:
            raise ValueError(f"季節 {name} に月が設定されていません")
        for month in months:
            if not isinstance(month, (int, np.integer)) or not 1 <= month <= 12:
                raise ValueError(f"季節 {name} の月 {month!r} が不正です")
            if month in lookup:
                raise ValueError(f"{month}月が複数の季節に設定されています")
            lookup[int(month)] = (code, int(months[0]))
    missing = sorted(set(range(1, 13)) - set(lookup))
    if missing:
        raise ValueError(f"どの季節にも属さない月があります: {missing}")
    return lookup


def month_to_season(month, season_months=None):
    """月(1〜12)を季節コードへ変換する。不正値はNaN。"""
    lookup = build_month_lookup(season_months)
    if pd.isna(month):
        return np.nan
    entry = lookup.get(int(month)) if float(month).is_integer() else None
    return float(entry[0]) if entry else np.nan


def season_year(year, month, season_months=None):
    """季節の「年度」を返す。冬(12,1,2)なら2026年1月は2025年度の冬(=2025年12月と同じ冬)。"""
    lookup = build_month_lookup(season_months)
    if pd.isna(year) or pd.isna(month) or int(month) not in lookup:
        return np.nan
    _, start_month = lookup[int(month)]
    return float(int(year) - 1 if int(month) < start_month else int(year))


def _season_columns(dates, season_months):
    lookup = build_month_lookup(season_months)
    month = dates.dt.month
    code = month.map(lambda m: lookup[int(m)][0] if pd.notna(m) else np.nan)
    start = month.map(lambda m: lookup[int(m)][1] if pd.notna(m) else np.nan)
    year = dates.dt.year
    s_year = np.where(month < start, year - 1, year).astype(float)
    s_year[dates.isna().to_numpy()] = np.nan
    return code.astype(float), pd.Series(s_year, index=dates.index)


def _place_slots(field_size):
    """JRAの複勝の払戻対象着順数: 8頭以上→3着まで、5〜7頭→2着まで、4頭以下→発売なし(1着のみ扱い)。"""
    return np.select([field_size >= 8, field_size >= 5], [3, 2], default=1).astype(float)


def _per_race_outcomes(df):
    """各エントリの成績を、頭数を考慮した比較可能な量へ変換する。"""
    raw_rank = df["着順"].astype(str)
    rank = df["着順"].map(parse_rank)
    dnf = raw_rank.str.contains("|".join(DNF_MARKS), na=False) & rank.isna()
    not_started = raw_rank.str.contains("|".join(NOT_STARTED_MARKS), na=False) & rank.isna()
    started = (rank.notna() | dnf) & ~not_started

    field = started.groupby(df["race_id"]).transform("sum").astype(float)
    rank_eff = rank.where(~dnf, field)  # 競走中止・失格は最下位扱い
    rank_eff = rank_eff.where(started)

    slots = _place_slots(field)
    popularity = pd.to_numeric(df["人気"], errors="coerce")
    denom = (field - 1).where(field > 1)

    out = pd.DataFrame(index=df.index)
    out["started"] = started.astype(float)
    out["win"] = (rank_eff == 1).astype(float).where(started)
    out["top2"] = (rank_eff <= 2).astype(float).where(started)
    out["place"] = (rank_eff <= slots).astype(float).where(started)
    # 複勝超過: 実際の複勝圏入り − ランダムに着順が決まった場合の期待値(払戻枠数/頭数)
    out["place_excess"] = out["place"] - (slots / field).where(started)
    # 着順比: 1着=0, 最下位=1(頭数の違いを吸収)
    out["rank_ratio"] = ((rank_eff - 1) / denom).where(started)
    # 人気差: 人気(レース前の市場評価)より何頭分上の着順に来たかを頭数で割ったもの(正=人気以上に好走)
    out["pop_diff"] = ((popularity - rank_eff) / denom).where(started)
    return out


_SUM_COLUMNS = ["started", "win", "top2", "place", "place_excess", "rank_ratio", "rank_ratio_sq",
                "pop_diff", "pop_n", "ratio_n"]


def _prior_sums(frame, keys):
    """keys単位で、各エントリより「前の日付」のエントリの合計を返す(同日は含めない)。"""
    daily = frame.groupby(keys + ["_date"], sort=True)[_SUM_COLUMNS].sum()
    cum = daily.groupby(level=list(range(len(keys))), sort=False).cumsum()
    prior = cum.groupby(level=list(range(len(keys))), sort=False).shift(1).fillna(0.0)
    merged = frame[keys + ["_date"]].merge(
        prior.reset_index(), on=keys + ["_date"], how="left", validate="many_to_one"
    )
    merged.index = frame.index
    return merged[_SUM_COLUMNS]


def _shrunk_relative(season_sum, season_n, overall_sum, overall_n, k):
    """季節別平均を通年平均へ、通年平均を0(平均的な馬)へ縮約し、その差を返す。"""
    overall = overall_sum / (overall_n + k)
    season = (season_sum + k * overall) / (season_n + k)
    return season - overall


def add_season_aptitude(df, season_months=None, prior_strength=DEFAULT_PRIOR_STRENGTH):
    """季節適性の特徴量(SEASON_FEATURE_COLUMNS)を追加して返す。行の並び順は入力のまま。

    必要な列: race_id, 馬, 日付(YYYY/MM/DD), 着順(生テキスト), 人気
    日付が解析できない行は季節系の特徴量がNaNになり、他の行の集計にも使わない。
    """
    k = float(prior_strength)
    if k <= 0:
        raise ValueError("prior_strength は正の値にしてください")

    working = df.copy()
    dates = pd.to_datetime(working["日付"], format="%Y/%m/%d", errors="coerce")
    season_code, s_year = _season_columns(dates, season_months)

    frame = _per_race_outcomes(working)
    frame["rank_ratio_sq"] = frame["rank_ratio"] ** 2
    frame["pop_n"] = frame["pop_diff"].notna().astype(float)
    frame["ratio_n"] = frame["rank_ratio"].notna().astype(float)
    frame = frame.fillna(0.0)
    frame["馬"] = working["馬"].fillna("不明").astype(str)
    frame["_date"] = dates
    frame["_season"] = season_code
    frame["_season_year"] = s_year

    valid = dates.notna()
    vf = frame[valid]
    overall = _prior_sums(vf, ["馬"])
    season = _prior_sums(vf, ["馬", "_season"])
    block = _prior_sums(vf, ["馬", "_season", "_season_year"])  # 当該季節(同じ年度)のみ
    prev = season - block  # 前年以前の同季節

    n = season["started"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = lambda col: (season[col] / n).where(n > 0)
        ratio_n = season["ratio_n"]
        ratio_mean = (season["rank_ratio"] / ratio_n).where(ratio_n > 0)
        ratio_var = ((season["rank_ratio_sq"] - ratio_n * ratio_mean ** 2) / (ratio_n - 1)).where(ratio_n > 1)

        feats = pd.DataFrame(index=vf.index)
        feats["季節"] = vf["_season"]
        feats["季節_出走数"] = n
        feats["季節_勝率"] = rate("win")
        feats["季節_連対率"] = rate("top2")
        feats["季節_複勝率"] = rate("place")
        feats["季節_平均着順比"] = ratio_mean
        feats["季節_着順比_標準偏差"] = np.sqrt(ratio_var.clip(lower=0))
        feats["季節_複勝超過_相対"] = _shrunk_relative(
            season["place_excess"], n, overall["place_excess"], overall["started"], k)
        feats["季節_人気差_相対"] = _shrunk_relative(
            season["pop_diff"], season["pop_n"], overall["pop_diff"], overall["pop_n"], k)
        feats["季節_信頼度"] = n / (n + k)
        feats["季節_前年以前出走数"] = prev["started"]
        feats["季節_前年以前_人気差_相対"] = _shrunk_relative(
            prev["pop_diff"], prev["pop_n"], overall["pop_diff"], overall["pop_n"], k)

    # 直近RECENT_WINDOW走の人気差(縮約済み)+季節の相対差。対象レース自身は含めない。
    sorted_vf = vf.sort_values(["馬", "_date"], kind="stable")
    g = sorted_vf.groupby("馬", sort=False)
    recent_sum = g["pop_diff"].transform(lambda s: s.rolling(RECENT_WINDOW, min_periods=1).sum().shift(1))
    recent_n = g["pop_n"].transform(lambda s: s.rolling(RECENT_WINDOW, min_periods=1).sum().shift(1))
    recent = (recent_sum.fillna(0.0) / (recent_n.fillna(0.0) + k)).reindex(vf.index)
    feats["季節_直近複合"] = recent + feats["季節_人気差_相対"]

    for col in SEASON_FEATURE_COLUMNS:
        working[col] = feats[col].reindex(working.index)
    return working

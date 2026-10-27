"""芝・ダート×距離帯による5カテゴリの定義と判定。

カテゴリごとに別のLightGBMモデルを学習し(model.train_category)、
予想時はレース条件から担当モデルを選ぶ(model.live_predict)。
"""
import numpy as np
import pandas as pd

# (キー, 表示名, 馬場コード(encode.SURFACE_MAP), 距離下限, 距離上限)
# 範囲外の距離(ダート1000m・1150m、芝3400m など)は get_category_key で最寄りのカテゴリへ寄せる。
CATEGORIES = [
    ("dirt_short", "ダート短距離 (1200〜1600m)", 1, 1200, 1600),
    ("dirt_long", "ダート中長距離 (1700〜2500m)", 1, 1700, 2500),
    ("turf_short", "芝短距離 (1200〜1600m)", 0, 1200, 1600),
    ("turf_mid", "芝中距離 (1700〜2400m)", 0, 1700, 2400),
    ("turf_long", "芝長距離 (2500〜3200m)", 0, 2500, 3200),
]
CATEGORY_KEYS = [c[0] for c in CATEGORIES]
CATEGORY_LABELS = {c[0]: c[1] for c in CATEGORIES}

# turf_longは学習データが約800行と少ないため、木を浅く・葉の最小サンプル数を大きくして過学習を抑える。
CATEGORY_LGB_PARAMS = {
    "turf_long": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 40,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.7,
    },
}


def _surface_code(surface):
    """「芝」「ダ」「ダート」や数値コード(0=芝, 1=ダート)を 0/1 に揃える。障害・不明はNone。"""
    if surface is None or (isinstance(surface, float) and np.isnan(surface)):
        return None
    if isinstance(surface, (int, float, np.integer, np.floating)):
        return int(surface) if int(surface) in (0, 1) else None
    text = str(surface).strip()
    if text.startswith("障"):
        return None
    if text.startswith("ダ"):
        return 1
    if text.startswith("芝"):
        return 0
    return None


def get_category_key(surface, distance):
    """レース条件(馬場・距離)から担当モデルのキーを返す。障害戦や判定不能ならNone。"""
    code = _surface_code(surface)
    distance = pd.to_numeric(distance, errors="coerce")
    if code is None or pd.isna(distance):
        return None
    if code == 1:
        return "dirt_short" if distance <= 1600 else "dirt_long"
    if distance <= 1600:
        return "turf_short"
    if distance <= 2400:
        return "turf_mid"
    return "turf_long"


def assign_categories(df, surface_col="芝・ダート", distance_col="距離"):
    """特徴量DataFrameの各行にカテゴリキーを割り当てたSeriesを返す。"""
    return pd.Series(
        [get_category_key(s, d) for s, d in zip(df[surface_col], df[distance_col])],
        index=df.index,
        dtype="object",
    )

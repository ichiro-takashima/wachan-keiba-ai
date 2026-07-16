"""出馬情報CSVの各列を数値・カテゴリコードへ変換するルール一式。

Step0で保存した生テキスト(クラス・走破時間・通過順・性齢・芝ダート・回り・馬場・天気・日付)を、
学習に使える数値表現へ変換する。
"""
import re

import numpy as np
import pandas as pd

# 部分一致の優先順位はこの並び順のまま判定する(先に一致した方を採用)。
# 実データのグレード表記は「G1」ではなく「(GI)」形式のため、ローマ数字表記も対象に含める。
# "GI"は"GII"/"GIII"の部分文字列でもあるため、その3つだけは長い表記から先に判定する
# (G1/G2/G3はローマ数字表記と衝突しないため元の指定順のまま)。それ以外のキーワードの
# 優先順位はユーザー指定の並びをそのまま踏襲する(例: 障害戦は「オープン」等のクラス語も
# 含むため、この並びでは障害より先にオープンが一致しうる)。
CLASS_RULES = [
    ("GIII", 8), ("GII", 9), ("GI", 10),
    ("G1", 10), ("G2", 9), ("G3", 8),
    ("(L)", 7),
    ("オープン", 7),
    ("3勝", 6), ("1600", 6),
    ("2勝", 5), ("1000", 5),
    ("1勝", 4), ("500", 4),
    ("新馬", 3),
    ("未勝利", 1),
    ("障害", 0),
]

SEX_MAP = {"牡": 0, "牝": 1, "セ": 2, "せ": 2, "騙": 2}
SURFACE_MAP = {"芝": 0, "ダート": 1, "障害": 2}
TURN_MAP = {"右": 0, "左": 1, "芝": 2, "直": 2}
WEATHER_RULES = [("晴", 0), ("曇", 1), ("小", 2), ("雨", 3), ("雪", 4)]
GOING_RULES = [("良", 0), ("稍", 1), ("重", 2), ("不", 3)]


def encode_class(text):
    text = "" if pd.isna(text) else str(text)
    for keyword, code in CLASS_RULES:
        if keyword in text:
            return code
    return np.nan


def parse_rank(value):
    """着順を数値化する。中止・除外・取消・失格などはNaNを返す。"""
    if pd.isna(value):
        return np.nan
    text = str(value)
    if any(mark in text for mark in ["中", "除", "取", "失"]):
        return np.nan
    match = re.search(r"\d+", text)
    return float(match.group(0)) if match else np.nan


def parse_time_to_seconds(value):
    """「1:31.5」→91.5(秒)のように走破時間を秒へ変換する。"""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.match(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def parse_corner_passage_avg(value):
    """「10-8-5」→(10+8+5)/3 のように通過順の平均を求める。"""
    if pd.isna(value):
        return np.nan
    numbers = [float(n) for n in re.findall(r"\d+", str(value))]
    return float(np.mean(numbers)) if numbers else np.nan


def encode_sex(value):
    text = "" if pd.isna(value) else str(value).strip()
    return SEX_MAP.get(text[:1], np.nan)


def encode_surface(value):
    text = "" if pd.isna(value) else str(value).strip()
    return SURFACE_MAP.get(text, np.nan)


def encode_turn(value):
    text = "" if pd.isna(value) else str(value).strip()
    return TURN_MAP.get(text, np.nan)


def _encode_by_rules(value, rules):
    text = "" if pd.isna(value) else str(value)
    for keyword, code in rules:
        if keyword in text:
            return code
    return np.nan


def encode_going(value):
    return _encode_by_rules(value, GOING_RULES)


def encode_weather(value):
    return _encode_by_rules(value, WEATHER_RULES)


def split_date(value):
    """「2024/05/12」→(2024, 5, 12) を返す。パースできない場合は(NaN,NaN,NaN)。"""
    if pd.isna(value):
        return np.nan, np.nan, np.nan
    match = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", str(value))
    if not match:
        return np.nan, np.nan, np.nan
    y, m, d = match.groups()
    return int(y), int(m), int(d)

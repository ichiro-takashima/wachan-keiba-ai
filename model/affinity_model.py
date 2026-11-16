"""血統・戦績の「レースとの相性」と馬体重の変化から、各馬の勝率を出すモデル(相性モデル)。

オッズ・人気は今回のレース分も過去走分も一切使わない(市場の評価に頼らず、馬の中身だけで勝率を出す)。
単勝オッズは画面で「期待値 = 勝率 × オッズ」を出すときにだけ使う。

特徴量は3つの柱 + 基本情報:
- 血統相性: 父・母父の産駒の「実力」(全体成績)と「適性」(条件ごとの成績が、その種牡馬の普段の成績より
            どれだけ上か)。使い方は予想モードで変わる(下記)
- 戦績相性: その馬自身が同じ条件で残した成績 + 近走の内容(着順率・タイム差・上がり・位置取り)
- 馬体重:   体重・増減・増減率・好走時の体重との差・平均体重との差
- 基本情報: 枠・斤量・性齢・騎手・レース条件
成績はすべて「そのレースの日より前」の走だけで数え、件数の少ない組み合わせは全体平均に寄せて(縮約)使う。

予想モード(血統の使い方だけが違う3つのモデル。3シード平均の比較で割り当てた):
- 回収率重視: 血統を使わない(勝率1位馬の単勝回収率が最も高かった)
- 標準:       血統を全馬に使い、キャリアが浅い・初めての条件ほど強く効かせる(精度・回収率とも中間)
- 確実性重視: 血統はキャリア0〜2走の馬と初めて走る条件だけに使い、それ以外は欠損にする(精度が最も高く安定)。
              父・母父の得意・苦手条件(model.sire_aptitude の判定を、その日より前の産駒成績だけで出したもの)も加える

どのモードでも、予想結果には「血統の得意・苦手」(今回の条件で父・母父が得意・苦手と判定されたもの)を表示する。

目的変数は1着。LightGBMの出力をレース内で正規化し、温度(a)を検証データで合わせて勝率にする。
    勝率_i = exp(a × logit(p_i)) / Σ_j exp(a × logit(p_j))

血統は scraping.collect_pedigree で取得した data/processed/pedigree.csv を使う(未取得の馬は欠損扱い)。

使い方:
    python -m scraping.collect_pedigree   # 血統を取得(初回は時間がかかる。途中で止めても再開できる)
    python -m model.affinity_model        # 3モードを学習 → テスト期間で検証 → data/models/affinity/<モード>/ に保存
    python -m model.affinity_model --mode 標準   # 1モードだけ学習
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from model import shutuba_model as sm
from model import sire_aptitude as sa
from preprocessing import encode

MODEL_DIR = Path("data/models/affinity")
PEDIGREE_RUNS_PATH = MODEL_DIR / "pedigree_runs.csv"  # 予想時に父・母父の成績を数えるための全走データ

# 予想モード → 保存先のフォルダ名・血統の使い方(pedigree_features の style)・種牡馬の得意・苦手判定を使うか
# aptitude: 3シード平均の比較で、確実性重視は判定あり(対数尤度 2.1038→2.1025、条件替わり 2.1907→2.1849)、
#           標準は判定なしのほうが良かった(2.1056→2.1111 に悪化)
MODES = {
    "回収率重視": {"key": "roi", "pedigree": "none", "aptitude": False,
                "説明": "血統を使わず戦績・馬体重・基本情報だけで判断。人気になりやすい良血馬を過大評価しにくい"},
    "標準": {"key": "standard", "pedigree": "weighted", "aptitude": False,
           "説明": "血統を全馬に使い、キャリアが浅い・初めての条件ほど強く効かせる"},
    "確実性重視": {"key": "accuracy", "pedigree": "gated", "aptitude": True,
              "説明": "血統はキャリア0〜2走の馬と初めて走る条件だけに使い、父・母父の得意・苦手条件も加味する。勝率の精度が最も高く安定"},
}
DEFAULT_MODE = "標準"


def mode_paths(mode):
    d = MODEL_DIR / MODES[mode]["key"]
    return {"dir": d, "model": d / "affinity_model.txt", "noweight": d / "affinity_model_noweight.txt",
            "metrics": d / "affinity_model.metrics.json", "temperature": d / "temperature.json"}

PLACE_IDS = {"札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5, "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10}

# 縮約の強さ(この件数ぶんの全体平均を足してから率を出す)と、寄せ先の全体平均
SHRINK_HORSE = 3.0
SHRINK_PEDIGREE = 30.0
CAREER_MAX = 2  # 確実性重視: この走数以下を「キャリアが浅い」とみなし、血統(実力・全適性)を使う
EXPERIENCE_K = 3.0  # 標準: 血統の効き K / (K + 自身の出走数)。未経験なら1、経験を積むほど0に近づく
PRIOR = {"win": 0.073, "top3": 0.215, "rr": 0.53}
MARGIN_PRIOR = 1.5  # 勝ち馬とのタイム差(秒)の寄せ先
COUNT_CAP = 20

# ---------------------------------------------------------------- 特徴量の定義

# 基本情報・近走(shutuba_model と同じ計算)。オッズ・人気から作られる列は入れない。
BASE_FEATURES = [
    "枠番", "馬番", "頭数", "斤量", "性", "齢", "芝・ダート", "距離", "回り", "馬場", "天気", "場id", "クラス",
    "騎手勝率", "騎手3着内率", "斤量_対平均", "騎手3着内率_対平均",
]
FORM_FEATURES = [
    "前走着順率", "近走平均着順率", "近走3着内率", "近走数", "前走間隔",
    "距離変化", "芝ダ替わり", "クラス変化", "斤量変化",
    "前走タイム差", "近走平均タイム差", "近走最小タイム差", "前走上がり", "前走4角位置率", "近走平均4角位置率",
    "近走平均着順率_対平均", "近走平均タイム差_対平均",
]
MARKET_COLUMNS = {"単勝オッズ", "人気", "人気比", "市場確率", "前走人気", "前走単勝オッズ", "近走平均人気", "近走人気超過"}

# (名前, キー) : その馬自身の成績を数える条件
HORSE_CONDITIONS = [
    ("全体", ["_horse"]),
    ("同芝ダ", ["_horse", "_surf"]),
    ("同距離帯", ["_horse", "_surf", "_band"]),
    ("同場", ["_horse", "_place", "_surf"]),
    ("同馬場状態", ["_horse", "_surf", "_wet"]),
]
# 父・母父の産駒の全体成績(=種牡馬の実力)
PEDIGREE_OVERALL = [("父", "sire"), ("母父", "bms")]
# (名前, 条件のキー, 比べる元のキー, その馬自身の経験を数えるキー)
# 適性 = 条件の3着内率(元の3着内率に寄せて縮約) − 元の3着内率。
# 芝ダ適性は全体と、距離帯・馬場状態・競馬場の適性は同じ芝ダの成績と比べる(芝ダの得意不得意を二重に数えない)。
PEDIGREE_CONDITIONS = [
    ("父_芝ダ", ["sire", "_surf"], ["sire"], ["_horse", "_surf"]),
    ("父_距離帯", ["sire", "_surf", "_band"], ["sire", "_surf"], ["_horse", "_surf", "_band"]),
    ("父_馬場状態", ["sire", "_surf", "_wet"], ["sire", "_surf"], ["_horse", "_surf", "_wet"]),
    ("父_競馬場", ["sire", "_place", "_surf"], ["sire", "_surf"], ["_horse", "_place", "_surf"]),
    ("母父_芝ダ", ["bms", "_surf"], ["bms"], ["_horse", "_surf"]),
    ("母父_距離帯", ["bms", "_surf", "_band"], ["bms", "_surf"], ["_horse", "_surf", "_band"]),
]

HORSE_AFFINITY_FEATURES = (
    [f"戦績_{n}_3着内率" for n, _ in HORSE_CONDITIONS]
    + [f"戦績_{n}_着順率" for n, _ in HORSE_CONDITIONS]
    + ["戦績_同芝ダ_出走数", "戦績_同距離帯_出走数", "戦績_同距離帯_平均タイム差", "戦績_好走距離差",
       "戦績_同距離帯_3着内率_対平均"]
)
_PEDIGREE_COMMON = ["血統_父_距離帯_出走数", "血統_父_好走距離差", "血統_父_距離帯_適性_対平均"]
# sire_aptitude の判定(今回の条件での父・母父の自身比のz値と、得意・苦手の数)。その日より前の産駒成績だけで数える。
# 種類 → その馬自身の経験を数えるキー(確実性重視では、初めて走る条件のときだけ使う)
APTITUDE_HORSE_KEYS = {"芝ダ": ["_horse", "_surf"], "距離": ["_horse", "_surf", "_band"],
                       "馬場": ["_horse", "_surf", "_wet"], "競馬場": ["_horse", "_place", "_surf"]}
APTITUDE_FEATURES = ([f"血統_{who}_{kind}_適性z" for who in sa.ASOF_BY for kind in sa.ASOF_KINDS]
                     + ["血統_適性判定"])
PEDIGREE_FEATURES_BY_STYLE = {
    "none": [],
    "gated": (["血統_父_全体_3着内率", "血統_父_全体_勝率", "血統_母父_全体_3着内率"]
              + [f"血統_{c[0]}_適性" for c in PEDIGREE_CONDITIONS] + _PEDIGREE_COMMON + APTITUDE_FEATURES),
    "weighted": (["血統_父_全体_3着内率", "血統_父_全体_勝率", "血統_母父_全体_3着内率", "血統_父_全体_3着内率_キャリア少"]
                 + [f"血統_{c[0]}_適性" for c in PEDIGREE_CONDITIONS]
                 + [f"血統_{c[0]}_適性_未経験" for c in PEDIGREE_CONDITIONS] + _PEDIGREE_COMMON + APTITUDE_FEATURES),
}
PEDIGREE_FEATURES = PEDIGREE_FEATURES_BY_STYLE["weighted"]  # 全方式の血統列(要素別の集計用)
WEIGHT_FEATURES = ["体重", "体重変化", "体重変化率", "体重_対好走時", "体重_対平均"]

FEATURE_GROUPS = {
    "血統相性": PEDIGREE_FEATURES,
    "戦績相性": HORSE_AFFINITY_FEATURES + FORM_FEATURES,
    "馬体重": WEIGHT_FEATURES,
    "基本情報": BASE_FEATURES,
}
FEATURE_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]  # 全方式の列


def feature_columns(style, weight=True, aptitude=True):
    """血統の使い方 style のモデルが使う列。weight=False なら馬体重の列、aptitude=False なら種牡馬の得意・苦手判定の列を除く。"""
    ped = set(PEDIGREE_FEATURES_BY_STYLE[style])
    return [c for c in FEATURE_COLUMNS
            if (c in ped or c not in PEDIGREE_FEATURES) and (weight or c not in WEIGHT_FEATURES)
            and (aptitude or c not in APTITUDE_FEATURES)]


assert not MARKET_COLUMNS & set(FEATURE_COLUMNS)
assert not sm.POST_RACE_COLUMNS & set(FEATURE_COLUMNS)

_CONDITION_LABELS = {
    "全体": "全成績", "同芝ダ": "同じ芝ダ", "同距離帯": "同じ距離帯", "同場": "同じ競馬場", "同馬場状態": "同じ馬場状態",
    "父_芝ダ": "父産駒の芝ダ", "父_距離帯": "父産駒の距離帯", "父_馬場状態": "父産駒の馬場状態",
    "父_競馬場": "父産駒の競馬場", "母父_芝ダ": "母父産駒の芝ダ", "母父_距離帯": "母父産駒の距離帯",
    "父_全体": "父産駒全体", "母父_全体": "母父産駒全体",
}


def _display_name(col):
    """根拠の表示用の日本語名。"""
    fixed = {
        "戦績_同芝ダ_出走数": "同じ芝ダの出走経験", "戦績_同距離帯_出走数": "同じ距離帯の出走経験",
        "戦績_同距離帯_平均タイム差": "同じ距離帯でのタイム差", "戦績_好走距離差": "好走した距離との近さ",
        "戦績_同距離帯_3着内率_対平均": "同じ距離帯の成績(出走馬内比較)",
        "血統_父_距離帯_出走数": "父産駒の距離帯での実績数", "血統_父_好走距離差": "父産駒の得意距離との近さ",
        "血統_父_距離帯_適性_対平均": "父産駒の距離適性(出走馬内比較)",
        "血統_父_全体_3着内率_キャリア少": "キャリアの浅い馬の父産駒の実力",
        "体重": "馬体重", "体重変化": "馬体重の増減", "体重変化率": "馬体重の増減率",
        "体重_対好走時": "好走時の馬体重との差", "体重_対平均": "平均馬体重との差",
        "血統_適性判定": "父・母父の得意条件(sire_aptitude)",
    }
    if col in fixed:
        return fixed[col]
    m = re.match(r"血統_(父|母父)_(芝ダ|距離|馬場|競馬場)_適性z$", col)
    if m:
        return f"{m.group(1)}産駒の{m.group(2)}の得意度"
    m = re.match(r"血統_(.+)_適性(_未経験)?$", col)
    if m:
        label = f"{_CONDITION_LABELS.get(m.group(1), m.group(1))}適性"
        return f"初めての条件での{label}" if m.group(2) else label
    m = re.match(r"(戦績|血統)_(.+)_(3着内率|着順率|勝率)$", col)
    if m:
        return f"{_CONDITION_LABELS.get(m.group(2), m.group(2))}の{m.group(3)}"
    return sm.FEATURE_DISPLAY_NAMES.get(col, col)


# ---------------------------------------------------------------- 1走=1行の表(学習・予想で共通)

def distance_band(distance):
    """距離帯: 0=~1300 / 1=1301-1500 / 2=1501-1800 / 3=1801-2200 / 4=2201~。"""
    d = pd.to_numeric(distance, errors="coerce")
    return pd.cut(d, [0, 1300, 1500, 1800, 2200, 10000], labels=False).astype(float)


def _finish_runs(runs):
    """_rank などの生の値から、集計に使う派生列(距離帯・道悪・勝ち・3着内…)を付ける。"""
    r = runs.copy()
    r["_band"] = distance_band(r["_dist"])
    r["_wet"] = (r["_going"] >= 1).astype(float).where(r["_going"].notna())
    valid = r["_rank"].notna() & r["_field"].notna() & (r["_field"] > 0)
    r["n"] = valid.astype(float)
    r["win"] = ((r["_rank"] == 1) & valid).astype(float)
    r["top3"] = ((r["_rank"] <= 3) & valid).astype(float)
    r["rr"] = (r["_rank"] / r["_field"]).where(valid, 0.0)
    margin = r["_margin"].clip(upper=sm.MARGIN_CLIP)
    r["m_n"] = (margin.notna() & valid).astype(float)
    r["m_sum"] = margin.where(r["m_n"] > 0, 0.0)
    r["top3_dist"] = (r["top3"] * r["_dist"]).fillna(0.0)
    w_ok = r["_weight"].notna() & valid
    r["w_n"] = w_ok.astype(float)
    r["w_sum"] = r["_weight"].where(w_ok, 0.0)
    r["t3w_n"] = (w_ok & (r["top3"] > 0)).astype(float)
    r["t3w_sum"] = r["_weight"].where(r["t3w_n"] > 0, 0.0)
    return r


def _prior_sums(runs, keys, values):
    """各行と同じキーで、その行の日付より前の走の values の合計。キーが欠けている行は NaN。"""
    valid = runs[keys].notna().all(axis=1)
    daily = runs.loc[valid, keys + ["_date"] + values].groupby(keys + ["_date"], sort=True)[values].sum()
    daily = daily.reset_index()
    daily[values] = daily.groupby(keys, sort=False)[values].cumsum() - daily[values]
    merged = runs[keys + ["_date"]].merge(daily, on=keys + ["_date"], how="left")
    merged.index = runs.index
    out = merged[values].copy()
    out.loc[~valid] = np.nan
    return out


def _shrunk(total, n, prior, strength):
    return (total + prior * strength) / (n + strength)


def horse_affinity_features(runs):
    """その馬自身が今回と同じ条件で残した成績(戦績相性)。"""
    out = pd.DataFrame(index=runs.index)
    for name, keys in HORSE_CONDITIONS:
        s = _prior_sums(runs, keys, ["n", "top3", "rr"])
        out[f"戦績_{name}_3着内率"] = _shrunk(s["top3"], s["n"], PRIOR["top3"], SHRINK_HORSE)
        out[f"戦績_{name}_着順率"] = _shrunk(s["rr"], s["n"], PRIOR["rr"], SHRINK_HORSE)
        if name in ("同芝ダ", "同距離帯"):
            out[f"戦績_{name}_出走数"] = s["n"].clip(upper=COUNT_CAP)
    s = _prior_sums(runs, ["_horse", "_surf", "_band"], ["m_n", "m_sum"])
    out["戦績_同距離帯_平均タイム差"] = _shrunk(s["m_sum"], s["m_n"], MARGIN_PRIOR, SHRINK_HORSE)
    s = _prior_sums(runs, ["_horse", "_surf"], ["top3", "top3_dist"])
    out["戦績_好走距離差"] = runs["_dist"] - s["top3_dist"] / s["top3"].where(s["top3"] > 0)
    return out


def own_experience(horse_runs, keys, index):
    """その馬自身が同じ条件を走った回数(当日より前の走だけ)。"""
    return _prior_sums(horse_runs, keys, ["n"])["n"].reindex(index).fillna(0)


def pedigree_features(runs, horse_runs=None, style="gated", aptitude=None):
    """父・母父の産駒の実力と条件適性(血統相性)。style で使い方を切り替える。

    - "gated"(確実性重視): 実力はキャリア CAREER_MAX 走以下の馬だけ、条件適性はキャリアが浅い馬か、その条件
      (芝ダ・距離帯・馬場状態・競馬場)を初めて走る馬だけに使い、それ以外は欠損にする
    - "weighted"(標準): 全馬に使い、K/(K+自身の出走数) を掛けた「経験が浅いほど効く」列も足す
    - "none"(回収率重視): 血統を使わない(空の表)

    horse_runs はその馬自身の経験を数える表(省略時は runs)。予想時は産駒成績の表と馬の戦績の表が別になるので
    渡し分ける。両者で、特徴量を使う行(今回の出走)のインデックスがそろっていること。
    aptitude は sire_aptitude.asof_aptitude(runs) の結果(計算済みなら渡すと使い回す)。
    """
    out = pd.DataFrame(index=runs.index)
    if style == "none":
        return out
    horse_runs = runs if horse_runs is None else horse_runs
    gated = style == "gated"
    rate = {}  # キー → 縮約済みの3着内率
    career = own_experience(horse_runs, ["_horse"], runs.index)
    young = career <= CAREER_MAX

    def gate(values, use):
        return values.where(use) if gated else values

    def top3_rate(keys, prior):
        s = _prior_sums(runs, keys, ["n", "top3"])
        rate[tuple(keys)] = _shrunk(s["top3"], s["n"], prior, SHRINK_PEDIGREE)
        return s

    for name, key in PEDIGREE_OVERALL:
        top3_rate([key], PRIOR["top3"])
        out[f"血統_{name}_全体_3着内率"] = gate(rate[(key,)], young)
    w = _prior_sums(runs, ["sire"], ["n", "win"])
    out["血統_父_全体_勝率"] = gate(_shrunk(w["win"], w["n"], PRIOR["win"], SHRINK_PEDIGREE), young)
    if not gated:
        out["血統_父_全体_3着内率_キャリア少"] = ((rate[("sire",)] - PRIOR["top3"])
                                           * EXPERIENCE_K / (EXPERIENCE_K + career))

    # 芝ダ(元=全体)を先に計算してから、距離帯など(元=同じ芝ダ)を計算する
    for name, keys, parent, horse_keys in sorted(PEDIGREE_CONDITIONS, key=lambda c: len(c[2])):
        base = rate[tuple(parent)]
        s = top3_rate(keys, base.fillna(PRIOR["top3"]))
        diff = rate[tuple(keys)] - base
        tried = own_experience(horse_runs, horse_keys, runs.index)
        use = young | (tried == 0)
        out[f"血統_{name}_適性"] = gate(diff, use)
        if not gated:
            out[f"血統_{name}_適性_未経験"] = diff * EXPERIENCE_K / (EXPERIENCE_K + tried)
        if name == "父_距離帯":
            out["血統_父_距離帯_出走数"] = gate(np.log1p(s["n"]), use)
            s2 = _prior_sums(runs, ["sire", "_surf"], ["top3", "top3_dist"])
            out["血統_父_好走距離差"] = gate(runs["_dist"] - s2["top3_dist"] / s2["top3"].where(s2["top3"] > 0), use)

    # sire_aptitude の判定: 今回の条件での自身比のz値(父・母父 × 芝ダ・距離・馬場・競馬場)と、得意(+1)・苦手(-1)の合計
    apt = (sa.asof_aptitude(runs) if aptitude is None else aptitude).reindex(runs.index)
    judge = pd.Series(0.0, index=runs.index)
    used = pd.Series(False, index=runs.index)
    for kind, horse_keys in APTITUDE_HORSE_KEYS.items():
        use = young | (own_experience(horse_runs, horse_keys, runs.index) == 0)
        used |= use
        for who in sa.ASOF_BY:
            out[f"血統_{who}_{kind}_適性z"] = gate(apt[f"{who}_{kind}_z値"], use)
            judge += gate(apt[f"{who}_{kind}_判定"], use).fillna(0)
    out["血統_適性判定"] = gate(judge, used)
    return out


def weight_features(runs):
    """馬体重の変化率と、好走したときの体重・普段の体重との差(体重・体重変化そのものは base_features にある)。"""
    out = pd.DataFrame(index=runs.index)
    w, dw = runs["_weight"], runs["_wchange"]
    out["体重変化率"] = dw / (w - dw)
    s = _prior_sums(runs, ["_horse"], ["w_n", "w_sum", "t3w_n", "t3w_sum"])
    out["体重_対好走時"] = w - s["t3w_sum"] / s["t3w_n"].where(s["t3w_n"] > 0)
    out["体重_対平均"] = w - s["w_sum"] / s["w_n"].where(s["w_n"] > 0)
    return out


def add_relative(feats, race_id):
    feats = feats.copy()
    for col in ["戦績_同距離帯_3着内率", "血統_父_距離帯_適性"]:
        if col not in feats.columns:
            continue
        feats[f"{col}_対平均"] = feats[col] - feats[col].groupby(race_id).transform("mean")
    return feats


# ---------------------------------------------------------------- 学習データ

def _pedigree_by_race_horse():
    """(race_id, 馬名) → horse_id, sire, bms。"""
    from scraping.collect_pedigree import build_horse_ids, load_pedigree

    ids = build_horse_ids()
    ped = load_pedigree()[["horse_id", "sire", "broodmare_sire"]].rename(columns={"broodmare_sire": "bms"})
    ped = ped.drop_duplicates("horse_id", keep="last")
    return ids.merge(ped, on="horse_id", how="left")


def training_runs(df, feats):
    """学習用: shutuba_model.training_features の結果から1走=1行の表を作る。"""
    ped = _pedigree_by_race_horse()
    keys = pd.DataFrame({"race_id": df["race_id"].astype(str), "馬名": df["馬名"].map(sm.clean_horse_name)})
    joined = keys.merge(ped, on=["race_id", "馬名"], how="left")
    joined.index = df.index
    weight, change = sm._body_weight(df["馬体重"])
    runs = pd.DataFrame({
        "_horse": joined["horse_id"].fillna("name:" + keys["馬名"]),
        "_date": df["_date"],
        "_rank": df["_rank"],
        "_field": feats["頭数"],
        "_dist": feats["距離"],
        "_surf": feats["芝・ダート"],
        "_place": feats["場id"],
        "_going": feats["馬場"],
        "_margin": sm._winner_margin(df),
        "_weight": weight,
        "_wchange": change,
        "sire": joined["sire"],
        "bms": joined["bms"],
    }, index=df.index)
    return _finish_runs(runs)


def build_training_table(rebuild=False, styles=("none", "weighted", "gated")):
    """学習用の表。x は血統の使い方(style)ごとの特徴量の表の辞書。"""
    train = sm.load_collected_rows("train", rebuild=rebuild).assign(split="train")
    test = sm.load_collected_rows("test", rebuild=rebuild).assign(split="test")
    df, feats = sm.training_features(pd.concat([train, test], ignore_index=True))
    runs = training_runs(df, feats)
    common = feats.join(horse_affinity_features(runs)).join(weight_features(runs))

    keep = df["_rank"].notna()
    x = {}
    for style in styles:
        f = add_relative(common.join(pedigree_features(runs, style=style)), df["race_id"])
        x[style] = f.loc[keep, feature_columns(style)].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    df = df[keep].reset_index(drop=True)
    df["win"] = (df["_rank"] == 1).astype(int)
    df["top3"] = (df["_rank"] <= 3).astype(int)
    df["体重変化"] = x[styles[0]]["体重変化"]
    df["血統あり"] = runs.loc[keep, "sire"].notna().to_numpy()
    df["_segment"] = experience_segment(runs).loc[keep].to_numpy()
    return df, x, runs


SEGMENTS = ["キャリア0〜2走", "条件替わり(初めての芝ダ・距離帯・馬場状態・競馬場)", "経験十分(血統を使わない)"]


def experience_segment(runs):
    """各走を、その馬の経験で3つに分ける。pedigree_features が血統を使う場面と一致させている。"""
    seg = pd.Series(SEGMENTS[2], index=runs.index)
    horse_keys = {tuple(c[3]) for c in PEDIGREE_CONDITIONS}
    untried = np.logical_or.reduce([own_experience(runs, list(k), runs.index) == 0 for k in horse_keys])
    seg[untried] = SEGMENTS[1]
    seg[own_experience(runs, ["_horse"], runs.index) <= CAREER_MAX] = SEGMENTS[0]
    return seg


# ---------------------------------------------------------------- 勝率への変換

def race_softmax(p, race_ids, a):
    """2値分類の出力 p をレース内で正規化して勝率にする(a は温度)。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    s = pd.Series(a * np.log(p / (1 - p)), index=getattr(race_ids, "index", None))
    race_ids = pd.Series(np.asarray(race_ids), index=s.index)
    e = np.exp(s - s.groupby(race_ids).transform("max"))
    return (e / e.groupby(race_ids).transform("sum")).to_numpy()


def _winner_nll(q, race_ids, win):
    d = pd.DataFrame({"q": q, "race": np.asarray(race_ids), "win": np.asarray(win)})
    d = d[d.groupby("race")["win"].transform("sum") == 1]
    return float(-np.log(d.loc[d["win"] == 1, "q"].clip(1e-12)).mean())


def fit_temperature(p, race_ids, win):
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda a: _winner_nll(race_softmax(p, race_ids, a), race_ids, win),
                          bounds=(0.2, 3.0), method="bounded")
    return float(res.x)


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=80,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbosity=-1,
)


def fit(df, x, features, seed=42):
    """学習期間の最後15%で木の本数と温度を決め、学習期間全体で学習し直す。(モデル, 温度, 木の本数)"""
    import lightgbm as lgb

    is_train = (df["split"] == "train").to_numpy()
    dates = df["_date"]
    valid_start = dates[is_train].quantile(0.85)
    fit_mask = is_train & (dates < valid_start).to_numpy()
    valid_mask = is_train & ~fit_mask
    params = dict(LGB_PARAMS, random_state=seed)
    xf = x[features]

    probe = lgb.LGBMClassifier(n_estimators=3000, **params)
    probe.fit(xf[fit_mask], df["win"][fit_mask], eval_set=[(xf[valid_mask], df["win"][valid_mask])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = max(int(probe.best_iteration_ or 0), 50)
    a = fit_temperature(probe.predict_proba(xf[valid_mask])[:, 1], df.loc[valid_mask, "race_id"],
                        df.loc[valid_mask, "win"])

    model = lgb.LGBMClassifier(n_estimators=best_iter, **params)
    model.fit(xf[is_train], df["win"][is_train])
    return model, a, best_iter


# ---------------------------------------------------------------- 検証

def _test_scores(df, q):
    from sklearn.metrics import roc_auc_score

    test = df.assign(q=q)
    return {
        "勝ち馬の対数尤度(低いほど良い)": round(_winner_nll(test["q"], test["race_id"], test["win"]), 4),
        "AUC(1着)": round(float(roc_auc_score(test["win"], test["q"])), 4),
        "AUC(3着内)": round(float(roc_auc_score(test["top3"], test["q"])), 4),
    }


def _segment_scores(test, q):
    """場面別(キャリア・条件替わり)の成績。勝率はレース全体で正規化したものをそのまま使う。"""
    from sklearn.metrics import roc_auc_score

    t = test.assign(q=q)
    out = {}
    for seg in SEGMENTS:
        g = t[t["_segment"] == seg]
        if g["win"].nunique() < 2:
            continue
        out[seg] = {
            "頭数": int(len(g)), "勝ち馬数": int(g["win"].sum()),
            "勝ち馬の対数尤度": round(_winner_nll(g["q"], g["race_id"], g["win"]), 4),
            "AUC(1着)": round(float(roc_auc_score(g["win"], g["q"])), 4),
            "AUC(3着内)": round(float(roc_auc_score(g["top3"], g["q"])), 4),
        }
    return out


def _pedigree_effect(with_ped, without_ped):
    """場面ごとに、血統ありモデルと血統なしモデルの成績を並べる(改善 = 対数尤度は下がる・AUCは上がる)。"""
    out = {}
    for seg, w in with_ped.items():
        wo = without_ped.get(seg)
        if not wo:
            continue
        out[seg] = {
            "頭数": w["頭数"],
            "対数尤度(血統あり/なし)": [w["勝ち馬の対数尤度"], wo["勝ち馬の対数尤度"]],
            "AUC1着(血統あり/なし)": [w["AUC(1着)"], wo["AUC(1着)"]],
            "AUC3着内(血統あり/なし)": [w["AUC(3着内)"], wo["AUC(3着内)"]],
            "対数尤度の改善": round(wo["勝ち馬の対数尤度"] - w["勝ち馬の対数尤度"], 4),
        }
    return out


def _calibration_table(test):
    bins = [0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 1.0]
    labels = ["~3%", "3-6%", "6-10%", "10-15%", "15-20%", "20-30%", "30%~"]
    grp = test.groupby(pd.cut(test["q"], bins, labels=labels), observed=False)
    return [
        {"予測勝率": str(k), "頭数": int(len(g)), "予測の平均": round(float(g["q"].mean()), 4) if len(g) else None,
         "実際の勝率": round(float(g["win"].mean()), 4) if len(g) else None}
        for k, g in grp
    ]


WEIGHT_CHANGE_BINS = [-100, -11, -5, -1, 1, 5, 11, 100]
WEIGHT_CHANGE_LABELS = ["-12kg以下", "-10〜-6kg", "-4〜-2kg", "±0kg", "+2〜+4kg", "+6〜+10kg", "+12kg以上"]


def weight_change_effect(test, contrib_weight):
    """馬体重の増減ごとに、実際の成績と、モデルが馬体重から見積もった影響(対数オッズ)をまとめる。"""
    t = test.assign(_c=contrib_weight)
    t = t[t["体重変化"].notna()]
    grp = t.groupby(pd.cut(t["体重変化"], WEIGHT_CHANGE_BINS, labels=WEIGHT_CHANGE_LABELS), observed=False)
    base_win = float(t["win"].mean())
    rows = []
    for k, g in grp:
        if not len(g):
            continue
        rows.append({
            "体重変化": str(k), "頭数": int(len(g)),
            "実際の勝率": round(float(g["win"].mean()), 4),
            "実際の3着内率": round(float(g["top3"].mean()), 4),
            "勝率の全体比": round(float(g["win"].mean()) / base_win, 2) if base_win else None,
            "モデルの馬体重寄与(平均)": round(float(g["_c"].mean()), 3),
        })
    return rows


def train_and_validate(rebuild=False, ablation=True, modes=None):
    """予想モードごとに学習・検証して data/models/affinity/<モード>/ に保存する。{モード: 検証成績}"""
    modes = list(modes or MODES)
    styles = tuple(dict.fromkeys(MODES[m]["pedigree"] for m in modes))
    df, xs, runs = build_training_table(rebuild=rebuild, styles=styles)
    results = {}
    for mode in modes:
        print(f"[{mode}] 学習中...", flush=True)
        results[mode] = _train_mode(df, xs[MODES[mode]["pedigree"]], mode, ablation=ablation)

    # 予想時に父・母父の産駒成績を数え直せるよう、血統の分かる走だけを保存する
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ped_runs = runs[runs["sire"].notna() | runs["bms"].notna()]
    ped_runs[["_date", "sire", "bms", "_surf", "_band", "_wet", "_place", "_dist", "n", "win", "top3", "top3_dist"]] \
        .to_csv(PEDIGREE_RUNS_PATH, index=False, encoding="utf-8-sig")
    if not sm.JOCKEY_STATS_PATH.exists():
        sm._save_jockey_stats()
    return results


def _train_mode(df, x, mode, ablation=True):
    style, apt = MODES[mode]["pedigree"], MODES[mode].get("aptitude", True)
    cols, cols_nw = feature_columns(style, aptitude=apt), feature_columns(style, weight=False, aptitude=apt)
    groups = {g: [c for c in gcols if c in cols] for g, gcols in FEATURE_GROUPS.items()}
    is_test = (df["split"] == "test").to_numpy()
    test = df[is_test].copy()

    model, a, best_iter = fit(df, x, cols)
    q = race_softmax(model.predict_proba(x[cols][is_test])[:, 1], test["race_id"], a)
    test["q"] = q
    scores = _test_scores(test, q)

    model_nw, a_nw, best_iter_nw = fit(df, x, cols_nw)
    q_nw = race_softmax(model_nw.predict_proba(x[cols_nw][is_test])[:, 1], test["race_id"], a_nw)

    # 比較用: 一様(1/頭数) と 単勝オッズ(市場)。モデル自体はオッズを使っていない。
    field = test.groupby("race_id")["race_id"].transform("size")
    odds = pd.to_numeric(test["単勝"], errors="coerce")
    inv = (1 / odds.where(odds > 0))
    market_q = inv / inv.groupby(test["race_id"]).transform("sum")
    popularity = -pd.to_numeric(test["人気"], errors="coerce")

    contrib = model.booster_.predict(x[cols][is_test], pred_contrib=True)[:, :-1]
    contrib = pd.DataFrame(contrib, columns=cols)
    group_share = {g: float(contrib[gc].abs().sum(axis=1).mean()) if gc else 0.0 for g, gc in groups.items()}
    total_share = sum(group_share.values())
    importance = pd.Series(model.booster_.feature_importance("gain"), index=cols)

    train_dates = df.loc[~is_test, "_date"]
    metrics = {
        "予想モード": mode,
        "モードの説明": MODES[mode]["説明"],
        "学習レース数": int(df.loc[~is_test, "race_id"].nunique()),
        "テストレース数": int(test["race_id"].nunique()),
        "学習期間": [str(train_dates.min().date()), str(train_dates.max().date())],
        "テスト期間": [str(test["_date"].min().date()), str(test["_date"].max().date())],
        "血統の取得率": round(float(df["血統あり"].mean()), 3),
        "木の本数": best_iter,
        "温度a": round(a, 3),
        "テスト成績": scores,
        "テスト成績(馬体重なしモデル)": _test_scores(test, q_nw),
        "場面別のテスト成績": _segment_scores(test, q),
        "比較: 一様(1/頭数)の対数尤度": round(_winner_nll(1 / field, test["race_id"], test["win"]), 4),
        "比較: 単勝オッズの対数尤度": round(_winner_nll(market_q, test["race_id"], test["win"]), 4),
        "勝率1位馬": sm._pick_stats(test.assign(_rank=test["_rank"]), "q"),
        "1番人気(参考)": sm._pick_stats(test.assign(score=popularity).dropna(subset=["score"]), "score"),
        "勝率の当てはまり": _calibration_table(test),
        "要素別の影響度(%)": {g: round(v / total_share * 100, 1) for g, v in group_share.items() if groups[g]},
        "馬体重の増減と成績": weight_change_effect(test, contrib[WEIGHT_FEATURES].sum(axis=1).to_numpy()),
        "特徴量": cols,
        "特徴量重要度(gain%)": (importance / importance.sum() * 100).round(1).sort_values(ascending=False).to_dict(),
        "馬体重なしモデル": {"木の本数": best_iter_nw, "温度a": round(a_nw, 3)},
    }

    if ablation:
        # 要素を1つずつ外して学習し直し、テスト期間の成績がどれだけ落ちるか(=その要素の貢献)を見る
        rows = []
        for group, gcols in groups.items():
            if group == "基本情報" or not gcols:
                continue
            feats = [c for c in cols if c not in gcols]
            m, a_g, _ = fit(df, x, feats)
            q_g = race_softmax(m.predict_proba(x[feats][is_test])[:, 1], test["race_id"], a_g)
            rows.append({"外した要素": group, **_test_scores(test, q_g)})
            if group == "血統相性":
                metrics["血統の効果(場面別)"] = _pedigree_effect(_segment_scores(test, q), _segment_scores(test, q_g))
        metrics["要素を外したときの成績"] = rows

    paths = mode_paths(mode)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]))
    model_nw.booster_.save_model(str(paths["noweight"]))
    paths["temperature"].write_text(json.dumps({"a": a, "a_noweight": a_nw}), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------- 予想(app.pyから使う)

def load_model(mode=DEFAULT_MODE):
    """学習済みの相性モデル一式(予想モード mode)。未学習なら None。"""
    paths = mode_paths(mode)
    if not paths["model"].exists():
        return None
    import lightgbm as lgb

    calib = json.loads(paths["temperature"].read_text(encoding="utf-8"))
    ped_runs = pd.read_csv(PEDIGREE_RUNS_PATH, encoding="utf-8-sig", dtype={"sire": str, "bms": str})
    ped_runs["_date"] = pd.to_datetime(ped_runs["_date"], errors="coerce")
    ped_runs = sa.add_expected(ped_runs)  # 種牡馬の得意・苦手判定に使う期待3着内数は、読み込み時に一度だけ数える
    return {
        "mode": mode,
        "pedigree_style": MODES[mode]["pedigree"],
        "booster": lgb.Booster(model_file=str(paths["model"])),
        "booster_noweight": lgb.Booster(model_file=str(paths["noweight"])) if paths["noweight"].exists() else None,
        "a": calib["a"],
        "a_noweight": calib.get("a_noweight", calib["a"]),
        "metrics": json.loads(paths["metrics"].read_text(encoding="utf-8")) if paths["metrics"].exists() else {},
        "pedigree_runs": ped_runs,
        "jockey_daily": sm._load_jockey_daily(),
    }


def _place_id_from_meeting(text):
    for name, pid in PLACE_IDS.items():
        if name in str(text):
            return float(pid)
    return np.nan


def _past_runs(key, results, race_date):
    """戦績ページの表(scraper.scrape_race_results_dedicated)を1走=1行の表にする(当日より前の走だけ)。"""
    if results is None or results.empty or "日付" not in results.columns:
        return None

    def col(name):
        return results[name] if name in results.columns else pd.Series(np.nan, index=results.index)

    course = col("距離").astype(str).str.extract(r"(芝|ダ|障)\D*(\d{3,4})")
    weight, change = sm._body_weight(col("馬体重"))
    runs = pd.DataFrame({
        "_horse": key,
        "_date": pd.to_datetime(results["日付"], errors="coerce"),
        "_rank": col("着順").map(encode.parse_rank),
        "_field": sm._num(col("頭数")),
        "_dist": pd.to_numeric(course[1], errors="coerce"),
        "_surf": course[0].map({"芝": "芝", "ダ": "ダート", "障": "障害"}).map(encode.encode_surface),
        "_place": col("開催").map(_place_id_from_meeting),
        "_going": col("馬場").map(encode.encode_going),
        "_margin": sm._num(col("着差")),
        "_weight": weight,
        "_wchange": change,
    })
    runs = runs[runs["_date"] < race_date]
    return runs if len(runs) else None


def _jra_only(all_horse_data):
    """戦績から中央競馬(JRA10場)以外の走を除く。学習データは中央のレースだけなので、地方・海外の走を
    混ぜると、地方で好走を重ねた馬の近走成績が中央の成績と同じ重みで数えられてしまう。"""
    out = []
    for horse in all_horse_data or []:
        results = horse.get("results")
        if results is not None and not results.empty and "開催" in results.columns:
            results = results[results["開催"].map(_place_id_from_meeting).notna()]
        out.append({**horse, "results": results})
    return out


def predict(bundle, race_id, race_info, shutuba_df, all_horse_data, race_date, use_weight=None):
    """出馬表の各馬の勝率と、血統相性・戦績相性・馬体重の評価を返す。

    use_weight=None なら、出馬表に馬体重が出ていれば馬体重ありのモデル、なければ馬体重なしのモデルを使う。
    """
    race_date = pd.Timestamp(race_date)
    all_horse_data = _jra_only(all_horse_data)
    entries, feats, names = sm.build_entry_features(
        bundle["jockey_daily"], race_id, race_info, shutuba_df, all_horse_data, race_date)
    n = len(feats)
    horse_data = {sm.clean_horse_name(h["pedigree"].get("name")): h for h in (all_horse_data or [])}
    peds = [horse_data.get(name, {}).get("pedigree", {}) for name in names]

    def known(value):
        return value if value and value != "不明" else np.nan

    weight, change = sm._body_weight(entries["馬体重"].astype(str))
    current = pd.DataFrame({
        "_horse": [f"entry{i}" for i in range(n)],
        "_date": race_date,
        "_rank": np.nan,
        "_field": feats["頭数"].to_numpy(dtype=float),
        "_dist": feats["距離"].to_numpy(dtype=float),
        "_surf": feats["芝・ダート"].to_numpy(dtype=float),
        "_place": feats["場id"].to_numpy(dtype=float),
        "_going": feats["馬場"].to_numpy(dtype=float),
        "_margin": np.nan,
        "_weight": weight.to_numpy(dtype=float),
        "_wchange": change.to_numpy(dtype=float),
        "sire": [known(p.get("sire")) for p in peds],
        "bms": [known(p.get("broodmare_sire")) for p in peds],
    })

    # 馬自身の成績: 今回の出走 + 戦績ページの過去走
    past = [_past_runs(f"entry{i}", horse_data.get(name, {}).get("results"), race_date) for i, name in enumerate(names)]
    horse_runs = _finish_runs(pd.concat([current] + [p for p in past if p is not None], ignore_index=True))
    cur_idx = horse_runs.index[:n]
    affinity = horse_affinity_features(horse_runs).loc[cur_idx].join(weight_features(horse_runs).loc[cur_idx])

    # 血統: 今回の出走 + 学習データの全走(当日より前)
    hist = bundle["pedigree_runs"]
    hist = hist[hist["_date"] < race_date].assign(_horse="hist")
    cur = _finish_runs(current)
    ped_runs = pd.concat([cur, hist], ignore_index=True)
    apt = sa.asof_aptitude(ped_runs, targets=cur_idx)  # 父・母父の得意・苦手(どの予想モードでも表示に使う)
    affinity = affinity.join(pedigree_features(ped_runs, horse_runs, style=bundle.get("pedigree_style", "gated"),
                                               aptitude=apt).loc[cur_idx])
    aptitude_notes = sa.aptitude_notes(apt.loc[cur_idx], ped_runs)

    affinity.index = feats.index
    feats = add_relative(feats.join(affinity), entries["race_id"])

    has_weight = bool(weight.notna().any())
    if use_weight is None:
        use_weight = has_weight
    if not use_weight and bundle.get("booster_noweight") is not None:
        booster, a = bundle["booster_noweight"], bundle["a_noweight"]
    else:
        booster, a = bundle["booster"], bundle["a"]
    names_used = booster.feature_name()
    x = feats.reindex(columns=names_used).apply(pd.to_numeric, errors="coerce")
    p = booster.predict(x)
    q = race_softmax(p, entries["race_id"], a)
    contrib = pd.DataFrame(booster.predict(x, pred_contrib=True)[:, :-1], columns=names_used, index=feats.index)

    def deviation(values):
        values = pd.Series(values, index=feats.index)
        sd = values.std(ddof=0)
        return (50 + 10 * (values - values.mean()) / sd).round(0) if sd > 1e-9 else pd.Series(50.0, index=feats.index)

    group_contrib = {g: contrib[[c for c in cols if c in names_used]].sum(axis=1) for g, cols in FEATURE_GROUPS.items()}
    # 馬体重の影響: 馬体重の寄与を除いた場合の勝率との差(ポイント)
    weight_cols = [c for c in WEIGHT_FEATURES if c in names_used]
    if weight_cols:
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        p_without = 1 / (1 + np.exp(-(logit - group_contrib["馬体重"].to_numpy())))
        weight_effect = (q - race_softmax(p_without, entries["race_id"], a)) * 100
    else:
        weight_effect = np.full(n, np.nan)

    reasons = []
    for _, row in contrib.iterrows():
        top = row.sort_values(ascending=False)
        reasons.append([_display_name(c) for c, v in top.head(3).items() if v > 0])

    odds = sm._num(entries["単勝"]).where(lambda s: s > 0)
    jockeys = shutuba_df.loc[entries.index, "騎手"] if "騎手" in shutuba_df.columns else ""
    out = pd.DataFrame({
        "馬番": sm._num(entries["馬番"]).astype(int),
        "馬名": names,
        "騎手": jockeys,
        "オッズ": odds,
        "人気": sm._num(entries["人気"]),
        "推定勝率": q,
        "血統相性": deviation(group_contrib["血統相性"]),
        "戦績相性": deviation(group_contrib["戦績相性"]),
        "馬体重": entries["馬体重"].astype(str).where(weight.notna(), "未発表"),
        "馬体重影響": np.round(weight_effect, 1),
        "血統": [f"父{p.get('sire', '不明')} / 母父{p.get('broodmare_sire', '不明')}" for p in peds],
        "血統の得意・苦手": aptitude_notes,
        "根拠": reasons,
        "過去走数": [0 if r is None else len(r) for r in past],
    }, index=feats.index)
    out["期待値"] = out["推定勝率"] * out["オッズ"]
    from model.live_predict import MARKS, _grade

    out["期待値ランク"] = out["期待値"].map(_grade)
    out = out.sort_values("推定勝率", ascending=False).reset_index(drop=True)
    out.insert(0, "AI順位", range(1, len(out) + 1))
    out.insert(1, "印", [MARKS[i] if i < len(MARKS) else "" for i in range(len(out))])
    return {
        "predictions": out,
        "mode": bundle.get("mode"),
        "uses_weight": bool(weight_cols),
        "weight_missing": not has_weight,
        "pedigree_missing": int(out["血統"].str.contains("不明").sum()),
    }


def format_for_prompt(result, metrics):
    """AIプロンプトに渡す文字列。"""
    if not result:
        return ""
    pred = result["predictions"]
    lines = ["【相性モデル(血統・戦績・馬体重から算出、オッズ不使用)の推定勝率】"]
    if metrics and metrics.get("予想モード"):
        lines.append(f"予想モード: {metrics['予想モード']}({metrics.get('モードの説明', '')})")
    if metrics:
        pick = metrics.get("勝率1位馬", {})
        lines.append(f"別期間{metrics.get('テストレース数', '?')}レースで検証済み(勝率1位馬の単勝的中率"
                     f"{pick.get('単勝的中率', 0):.1%}・複勝的中率{pick.get('複勝的中率', 0):.1%})。")
    lines.append("血統相性・戦績相性は出走馬内の偏差値(50が平均)。馬体重影響は馬体重による勝率の増減(ポイント)。")
    lines.append("血統の得意・苦手は、父・母父の産駒が今回と同じ条件(芝ダ・距離・馬場・競馬場)で、その種牡馬の普段の水準より"
                 "明らかに走る(得意)・走らない(苦手)もの。( )内は普段の水準に対する3着内率の比。")
    if result.get("weight_missing"):
        lines.append("※馬体重が未発表のため、馬体重なしで学習したモデルで計算しています。")
    elif not result.get("uses_weight"):
        lines.append("※馬体重を使わないモデルで計算しています。")
    lines.append("順位|馬番|馬名|推定勝率|血統相性|戦績相性|馬体重|馬体重影響|血統|血統の得意・苦手|主な根拠")
    for _, r in pred.iterrows():
        effect = "-" if pd.isna(r["馬体重影響"]) else f"{r['馬体重影響']:+.1f}pt"
        notes = "・".join(r.get("血統の得意・苦手") or []) or "-"
        lines.append(f"{r['AI順位']}|{r['馬番']}|{r['馬名']}|{r['推定勝率']:.1%}|{r['血統相性']:.0f}|{r['戦績相性']:.0f}|"
                     f"{r['馬体重']}|{effect}|{r['血統']}|{notes}|{'・'.join(r['根拠'])}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="血統・戦績・馬体重の相性モデルを学習し、テスト期間で検証する")
    parser.add_argument("--rebuild", action="store_true", help="レースHTMLの解析をやり直す")
    parser.add_argument("--no-ablation", action="store_true", help="要素を外した比較学習を省く(速い)")
    parser.add_argument("--mode", choices=list(MODES), action="append", help="学習する予想モード(省略時は全モード)")
    args = parser.parse_args()
    results = train_and_validate(rebuild=args.rebuild, ablation=not args.no_ablation, modes=args.mode)
    skip = {"特徴量", "特徴量重要度(gain%)", "勝率の当てはまり", "馬体重の増減と成績"}
    for mode, metrics in results.items():
        print(f"===== {mode} =====")
        print(json.dumps({k: v for k, v in metrics.items() if k not in skip}, ensure_ascii=False, indent=2))
        print(f"保存: {mode_paths(mode)['dir']}")


if __name__ == "__main__":
    main()

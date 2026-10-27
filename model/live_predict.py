"""予想対象レース(未来レース含む)の特徴量を組み立て、カテゴリ別モデルで3着以内確率を出す。

app.pyが取得済みのデータ(出馬表・レース情報・各馬の戦績ページ)から、学習時の
features.csv(preprocessing.build_features)と同じ列を再現する。
  - 直近5走(_1〜_5)は各馬の戦績ページから作る
  - 騎手勝率はrace_results.csvのうち基準日より前の成績で集計する
  - 馬_id/騎手_idなどのラベルエンコード列はfeatures.csvの対応表を引く(未知の値はNaN)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.categories import CATEGORY_LABELS, get_category_key
from model.win_prob import blend_win_probability, load_calibration
from preprocessing import encode
from preprocessing.history import HISTORY_WINDOW

CATEGORY_MODEL_DIR = Path("data/models/category")
GLOBAL_MODEL_PATH = Path("data/models/base_model.txt")
FEATURES_CSV = Path("data/processed/features.csv")
RESULTS_CSV = Path("data/processed/race_results.csv")

PLACE_NAMES = {1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京", 6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"}
ID_LOOKUP_COLUMNS = ["馬", "騎手", "レース名", "開催", "場名"]
MARKS = ["◎", "○", "▲", "△", "☆"]

# 予測根拠の表示用に、特徴量名を読みやすい日本語へ置き換える(未登録の列はそのまま表示)。
FEATURE_DISPLAY_NAMES = {
    "騎手勝率": "騎手の勝率",
    "騎手_id": "騎手",
    "馬_id": "馬の個体傾向",
    "オッズ_1": "前走の単勝オッズ",
    "着順_1": "前走着順",
    "着順_2": "2走前着順",
    "上がり_1": "前走上がり3F",
    "通過順_1": "前走の位置取り",
    "走破時間_1": "前走タイム",
    "距離差_1": "前走からの距離変化",
    "日付差_1": "レース間隔",
    "クラス_1": "前走クラス",
    "体重": "馬体重",
    "体重変化": "馬体重増減",
    "斤量": "斤量",
    "齢": "年齢",
    "馬番": "馬番(枠順)",
    "平均斤量": "近走平均斤量",
}


@dataclass
class ReferenceData:
    """推論時に参照する学習データ由来の対応表。app側でキャッシュして使い回す。"""
    id_maps: dict = field(default_factory=dict)  # 列名 -> {値: id}
    jockey_history: pd.DataFrame = None  # 騎手, _date, _is_win


def load_reference_data(features_csv=FEATURES_CSV, results_csv=RESULTS_CSV):
    id_maps = {}
    if Path(features_csv).exists():
        usecols = ID_LOOKUP_COLUMNS + [f"{c}_id" for c in ID_LOOKUP_COLUMNS]
        feats = pd.read_csv(features_csv, encoding="utf-8-sig", usecols=usecols)
        for col in ID_LOOKUP_COLUMNS:
            pairs = feats[[col, f"{col}_id"]].dropna().drop_duplicates(col)
            id_maps[col] = dict(zip(pairs[col].astype(str), pairs[f"{col}_id"]))

    jockey_history = pd.DataFrame(columns=["騎手", "_date", "_is_win"])
    if Path(results_csv).exists():
        results = pd.read_csv(results_csv, encoding="utf-8-sig", usecols=["騎手", "日付", "着順"])
        jockey_history = pd.DataFrame({
            "騎手": results["騎手"].astype(str),
            "_date": pd.to_datetime(results["日付"], format="%Y/%m/%d", errors="coerce"),
            "_is_win": (results["着順"].map(encode.parse_rank) == 1).astype(float),
        })
    return ReferenceData(id_maps=id_maps, jockey_history=jockey_history)


def available_category_models(model_dir=CATEGORY_MODEL_DIR):
    return {p.stem: p for p in Path(model_dir).glob("*.txt") if p.stem in CATEGORY_LABELS}


def _load_booster(path):
    return lgb.Booster(model_file=str(path))


# ---------------------------------------------------------------- レース条件

def parse_race_conditions(race_info, race_id, race_date):
    """race_info(scraper.scrape_race_info の戻り値)から馬場・距離・天気などを取り出す。"""
    race_info = race_info or {}
    text = " ".join(str(race_info.get(k, "")) for k in ["course", "data", "detail"])

    surface, distance = None, np.nan
    m = re.search(r"(芝|ダ|障)\D{0,6}?(\d{3,4})\s*m", text)
    if m:
        surface = {"芝": "芝", "ダ": "ダート", "障": "障害"}[m.group(1)]
        distance = float(m.group(2))

    turn = None
    m = re.search(r"(?:\(|（|芝|ダート?)\s*(右|左|直)", text)
    if m:
        turn = m.group(1)

    weather = None
    m = re.search(r"天候\s*[:：]\s*(\S+)", text)
    if m:
        weather = m.group(1)

    going = None
    m = re.search(r"馬場\s*[:：]\s*(良|稍重|稍|重|不良)", text) or re.search(r"(?:芝|ダート)\s*[:：]\s*(良|稍重|稍|重|不良)", text)
    if m:
        going = m.group(1)

    place_id = int(race_id[4:6]) if race_id and len(race_id) >= 10 and race_id[4:6].isdigit() else None
    place_name = PLACE_NAMES.get(place_id)
    meeting = None
    if place_name and race_id[6:8].isdigit() and race_id[8:10].isdigit():
        meeting = f"{int(race_id[6:8])}回{place_name}{int(race_id[8:10])}日目"

    class_text = " ".join(str(race_info.get(k, "")) for k in ["race_class", "name", "detail"])
    race_date = pd.Timestamp(race_date)
    return {
        "surface": surface,
        "distance": distance,
        "turn": turn,
        "weather": weather,
        "going": going,
        "place_id": place_id if place_name else None,
        "place_name": place_name,
        "meeting": meeting,
        "race_name": str(race_info.get("name", "")).strip(),
        "class_code": encode.encode_class(class_text),
        "date": race_date,
        "category": get_category_key(surface, distance),
    }


# ---------------------------------------------------------------- 各馬の特徴量

def _num(value):
    return pd.to_numeric(str(value).replace(",", "").strip(), errors="coerce") if pd.notna(value) else np.nan


def _parse_body_weight(value):
    """「516(+6)」→(516, 6)。計不・空欄は(NaN, NaN)。"""
    m = re.search(r"(\d{3,4})\s*\(\s*([+-]?\d+)\s*\)", str(value or ""))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def _parse_course(value):
    """戦績ページの距離列「芝1800」「ダ1200」→(芝・ダートコード, 距離)。"""
    m = re.search(r"(芝|ダ|障)\D*(\d{3,4})", str(value or ""))
    if not m:
        return np.nan, np.nan
    surface = {"芝": "芝", "ダ": "ダート", "障": "障害"}[m.group(1)]
    return encode.encode_surface(surface), float(m.group(2))


def _clean_jockey(name):
    return re.sub(r"^[▲△☆★◇◆]+", "", str(name or "")).strip()


def _history_features(results_df, race_date, race_distance):
    feats = {}
    if results_df is None or results_df.empty or "日付" not in results_df.columns:
        return feats

    past = results_df.copy()
    past["_date"] = pd.to_datetime(past["日付"], errors="coerce")
    past = past[past["_date"] < race_date].sort_values("_date", ascending=False).head(HISTORY_WINDOW)

    base_date, base_distance = race_date, race_distance
    for i, (_, row) in enumerate(past.iterrows(), start=1):
        surface_code, distance = _parse_course(row.get("距離"))
        weight, weight_diff = _parse_body_weight(row.get("馬体重"))
        feats.update({
            f"馬番_{i}": _num(row.get("馬番")),
            f"斤量_{i}": _num(row.get("斤量")),
            f"オッズ_{i}": _num(row.get("オッズ")),
            f"体重_{i}": weight,
            f"体重変化_{i}": weight_diff,
            f"上がり_{i}": _num(row.get("上り")),
            f"通過順_{i}": encode.parse_corner_passage_avg(row.get("通過")),
            f"着順_{i}": encode.parse_rank(row.get("着順")),
            f"距離_{i}": distance,
            f"クラス_{i}": encode.encode_class(row.get("レース名")),
            f"走破時間_{i}": encode.parse_time_to_seconds(row.get("タイム")),
            f"芝・ダート_{i}": surface_code,
            f"天気_{i}": encode.encode_weather(row.get("天気")),
            f"馬場_{i}": encode.encode_going(row.get("馬場")),
            f"日付差_{i}": (base_date - row["_date"]).days,
            f"距離差_{i}": base_distance - distance,
        })
        base_date, base_distance = row["_date"], distance
    return feats


def _jockey_win_rate(reference, jockey, race_date):
    hist = reference.jockey_history
    if hist is None or hist.empty or not jockey:
        return np.nan
    rows = hist[(hist["騎手"] == jockey) & (hist["_date"] < race_date)]
    return float(rows["_is_win"].mean()) if len(rows) else np.nan


def build_live_features(conditions, shutuba_df, all_horse_data, reference):
    """出走馬1頭=1行の特徴量DataFrameを返す(学習時と同じ列名)。"""
    results_by_name = {
        re.sub(r"\s+", "", str(h["pedigree"].get("name", ""))): h.get("results")
        for h in (all_horse_data or [])
    }
    race_date = conditions["date"]
    surface_code = encode.encode_surface(conditions["surface"])
    distance_cat = encode.encode_distance_category(conditions["distance"])
    id_maps = reference.id_maps

    rows = []
    for _, entry in shutuba_df.iterrows():
        name = re.sub(r"\s+", "", str(entry.get("馬名", "")))
        jockey = _clean_jockey(entry.get("騎手"))
        sex_age = str(entry.get("性齢", "")).strip()
        age = re.search(r"\d+", sex_age)
        weight, weight_diff = _parse_body_weight(entry.get("馬体重"))

        feat = {
            "馬番": _num(entry.get("馬番")),
            "体重": weight,
            "体重変化": weight_diff,
            "性": encode.encode_sex(sex_age),
            "齢": float(age.group(0)) if age else np.nan,
            "斤量": _num(entry.get("斤量")),
            "クラス": conditions["class_code"],
            "芝・ダート": surface_code,
            "距離": conditions["distance"],
            "回り": encode.encode_turn(conditions["turn"]),
            "馬場": encode.encode_going(conditions["going"]),
            "天気": encode.encode_weather(conditions["weather"]),
            "場id": conditions["place_id"],
            "距離区分": distance_cat,
            "年": race_date.year,
            "月": race_date.month,
            "日": race_date.day,
            "騎手勝率": _jockey_win_rate(reference, jockey, race_date),
            "馬_id": id_maps.get("馬", {}).get(name, np.nan),
            "騎手_id": id_maps.get("騎手", {}).get(jockey, np.nan),
            "レース名_id": id_maps.get("レース名", {}).get(conditions["race_name"], np.nan),
            "開催_id": id_maps.get("開催", {}).get(conditions["meeting"], np.nan),
            "場名_id": id_maps.get("場名", {}).get(conditions["place_name"], np.nan),
        }
        feat.update(_history_features(results_by_name.get(name), race_date, conditions["distance"]))

        weights = [feat.get(f"斤量_{i}") for i in range(1, HISTORY_WINDOW + 1)]
        weights = [w for w in weights if pd.notna(w)]
        feat["平均斤量"] = float(np.mean(weights)) if weights else np.nan

        pass1 = feat.get("通過順_1", np.nan)
        lead_flag = np.nan if pd.isna(distance_cat) else float(distance_cat in (0, 3))
        feat.update({
            "芝ダート_距離区分": surface_code * 10 + distance_cat if pd.notna(surface_code) and pd.notna(distance_cat) else np.nan,
            "通過順1_距離区分": pass1 * distance_cat if pd.notna(distance_cat) else np.nan,
            "通過順1_芝ダート": pass1 * surface_code if pd.notna(surface_code) else np.nan,
            "距離帯_先行有利": lead_flag,
            "通過順1_先行有利": pass1 * lead_flag,
        })
        rows.append(feat)

    return pd.DataFrame(rows, index=shutuba_df.index)


# ---------------------------------------------------------------- 予測

def _grade(ev):
    if pd.isna(ev):
        return "-"
    if ev >= 1.3:
        return "S"
    if ev >= 1.0:
        return "A"
    if ev >= 0.8:
        return "B"
    return "C"


def _top_reasons(contrib_row, feature_names, feature_values, n=3):
    """pred_contribの値が大きい(3着以内方向へ押し上げた)特徴量を上位n個返す。"""
    pairs = sorted(zip(feature_names, contrib_row, feature_values), key=lambda x: x[1], reverse=True)
    reasons = []
    for name, value, raw in pairs:
        if value <= 0 or len(reasons) >= n:
            break
        if pd.isna(raw) and name not in ("馬_id", "騎手_id"):
            continue
        reasons.append(FEATURE_DISPLAY_NAMES.get(name, name))
    return reasons


def predict_race(race_id, race_info, shutuba_df, all_horse_data, race_date, reference, model_mode="category"):
    """レース全頭の3着以内確率・推定勝率・期待値・印を計算する。

    model_mode="category"ならレース条件に対応するカテゴリ別モデル、"global"なら全体モデルを使う。
    カテゴリ別モデルが未学習・対象外(障害など)の場合は全体モデルへフォールバックする。
    戻り値: {"conditions", "model_key", "model_label", "fallback_reason", "predictions"}
    """
    if shutuba_df is None or shutuba_df.empty:
        raise ValueError("出馬表がありません。")

    conditions = parse_race_conditions(race_info, race_id, race_date)
    category = conditions["category"]
    category_models = available_category_models()

    fallback_reason = None
    if model_mode == "category" and category in category_models:
        model_path, model_key, model_label = category_models[category], category, CATEGORY_LABELS[category]
    else:
        if model_mode == "category":
            fallback_reason = (
                "障害戦または馬場・距離を判定できないため" if category is None
                else f"{category} のモデルが未学習のため(python -m model.train_category で作成)"
            )
        if not GLOBAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"モデルファイルがありません: {GLOBAL_MODEL_PATH}")
        model_path, model_key, model_label = GLOBAL_MODEL_PATH, "global", "全体モデル(全条件共通)"

    booster = _load_booster(model_path)
    feature_names = booster.feature_name()
    features = build_live_features(conditions, shutuba_df, all_horse_data, reference)
    x = features.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce")

    proba = booster.predict(x)
    contrib = booster.predict(x, pred_contrib=True)[:, :-1]

    out = pd.DataFrame({
        "馬番": pd.to_numeric(shutuba_df.get("馬番"), errors="coerce"),
        "馬名": shutuba_df.get("馬名", ""),
        "騎手": shutuba_df.get("騎手", ""),
        "オッズ": pd.to_numeric(shutuba_df.get("オッズ"), errors="coerce"),
        "人気": pd.to_numeric(shutuba_df.get("人気"), errors="coerce"),
        "3着内確率": proba,
    }, index=shutuba_df.index)
    calibration = load_calibration()
    has_odds = out["オッズ"].notna() & (out["オッズ"] > 0)
    if calibration and has_odds.all():
        # モデル確率と単勝オッズを補正係数で組み合わせた勝率(model.win_prob)。
        out["推定勝率"] = blend_win_probability(
            out["3着内確率"], out["オッズ"], pd.Series(race_id, index=out.index), calibration["a"], calibration["b"],
        )
        win_prob_method = "blend"
    else:
        # オッズ未発表・補正係数なしのときは、3着以内確率の合計(≒3)で割った目安(人気薄を過大評価しやすい)。
        total = out["3着内確率"].sum()
        out["推定勝率"] = out["3着内確率"] / total if total > 0 else np.nan
        win_prob_method = "model_only"
    out["期待値"] = out["推定勝率"] * out["オッズ"]
    out["期待値ランク"] = out["期待値"].map(_grade)
    out["根拠"] = [
        _top_reasons(contrib[i], feature_names, x.iloc[i].tolist()) for i in range(len(out))
    ]
    out["過去走数"] = features[[c for c in features.columns if c.startswith("着順_")]].notna().sum(axis=1) if any(
        c.startswith("着順_") for c in features.columns) else 0

    out = out.sort_values("3着内確率", ascending=False).reset_index(drop=True)
    out.insert(0, "AI順位", range(1, len(out) + 1))
    out.insert(1, "印", [MARKS[i] if i < len(MARKS) else "" for i in range(len(out))])

    return {
        "conditions": conditions,
        "model_key": model_key,
        "model_label": model_label,
        "fallback_reason": fallback_reason,
        "win_prob_method": win_prob_method,
        "predictions": out,
    }


def load_category_importance(model_key, top_n=15):
    path = CATEGORY_MODEL_DIR / f"{model_key}_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig").head(top_n)


def load_category_comparison():
    path = CATEGORY_MODEL_DIR / "comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")

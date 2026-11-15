"""収集済みレースHTML(data/train, data/test)から「レース前に分かる情報だけ」で
各馬の3着以内確率を出すLightGBMモデル(出馬表モデル)。

- 学習: data/train/race_html のレースで学習する
- 検証: data/test/race_html のレース(学習より後の期間)で的中率・回収率を測り、
        1番人気を買い続けた場合より良いときだけ採用(data/models/shutuba_model.txt)する
- 予想: app.py の単一レース予想から predict_shutuba() を呼び出す

走破タイム・上がり3F・通過順・着順はレースが終わらないと分からないため、そのレースの特徴量には使わない
(着順は目的変数として、またこれらはその馬の「次走以降」の前走成績としてだけ使う)。
特徴量グループごとの効果は python -m model.shutuba_ablation で比較できる。

使い方:
    python -m model.shutuba_model            # データ作成 → 学習 → テスト検証 → 合格なら採用
    python -m model.shutuba_model --force    # 検証結果に関わらず採用
    python -m model.shutuba_model --rebuild  # HTMLの解析をやり直す(キャッシュCSVを作り直す)
    python -m model.shutuba_model --no-weight  # 馬体重なしモデルを前もって学習(アプリで選んだときも学習される)
"""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import encode

TRAIN_HTML_DIR = Path("data/train/race_html")
TEST_HTML_DIR = Path("data/test/race_html")
ROWS_CACHE = {"train": Path("data/processed/collected_train.csv"), "test": Path("data/processed/collected_test.csv")}

MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "shutuba_model.txt"  # 検証に合格した(アプリで使う)モデル
CANDIDATE_PATH = MODEL_DIR / "shutuba_model.candidate.txt"  # 直近に学習したモデル(不合格でも残す)
METRICS_PATH = MODEL_DIR / "shutuba_model.metrics.json"
JOCKEY_STATS_PATH = MODEL_DIR / "shutuba_jockey_daily.csv"
# アプリで説明変数を選び直したときに学習するモデル(特徴量の組み合わせごとに1つ保存して使い回す)
VARIANT_DIR = MODEL_DIR / "shutuba_variants"

HISTORY_WINDOW = 5

# レース後にしか分からない列。学習データには残るが、特徴量には絶対に入れない。
POST_RACE_COLUMNS = {"走破時間", "上がり", "通過順", "着順", "タイム", "通過", "上り"}
# そのうち、その馬の「次走以降」の前走成績(タイム差・位置取り・上がり)を作るのに使う列。
PAST_RUN_SOURCE_COLUMNS = ["タイム", "通過", "上り"]

BASE_FEATURE_COLUMNS = [
    "枠番", "馬番", "頭数", "斤量", "体重", "体重変化", "性", "齢",
    "単勝オッズ", "人気", "人気比", "市場確率",
    "芝・ダート", "距離", "回り", "馬場", "天気", "場id", "クラス",
    "騎手騎乗数", "騎手3着内率", "騎手勝率",
    "前走着順", "近走平均着順", "近走3着内率", "前走間隔",
]

# 追加特徴量の候補。グループ単位の効果は python -m model.shutuba_ablation で比較する。
# どれも出馬表と各馬の戦績(当日より前の走)だけから作れる。
FEATURE_GROUPS = {
    "近走の人気・着順率": [
        "前走人気", "前走単勝オッズ", "前走頭数", "前走着順率",
        "近走平均人気", "近走平均着順率", "近走人気超過", "近走数",
    ],
    "条件替わり": ["距離変化", "芝ダ替わり", "クラス変化", "斤量変化"],
    "タイム差・位置取り": [
        "前走タイム差", "近走平均タイム差", "近走最小タイム差",
        "前走上がり", "前走4角位置率", "近走平均4角位置率",
    ],
    "レース内相対": ["斤量_対平均", "騎手3着内率_対平均", "近走平均着順率_対平均", "近走平均タイム差_対平均"],
}
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [c for cols in FEATURE_GROUPS.values() for c in cols]

# 候補のうち、効いていない(ノイズ)と判定して学習・予想に使わない特徴量。
# python -m model.shutuba_feature_selection の結果(data/models/shutuba_feature_selection.json)を反映する。
# 外した7個は、値そのものより派生させた特徴量(斤量_対平均・クラス変化・前走着順率など)の方が効いていたもの。
# 派生特徴量の計算には引き続き使うので、特徴量表(feats)には残る。
DROPPED_FEATURES = ["斤量", "馬場", "クラス", "騎手騎乗数", "騎手3着内率", "前走着順", "前走人気"]
assert not set(DROPPED_FEATURES) - set(ALL_FEATURE_COLUMNS)

# 学習・予想に使う特徴量。
FEATURE_COLUMNS = [c for c in ALL_FEATURE_COLUMNS if c not in DROPPED_FEATURES]
assert not POST_RACE_COLUMNS & set(ALL_FEATURE_COLUMNS)

# 馬体重の特徴量。当日発表(発走の約1時間前)のため、未発表でも予想できるようアプリで個別に外せる。
WEIGHT_FEATURES = ["体重", "体重変化"]

FEATURE_DISPLAY_NAMES = {
    "単勝オッズ": "単勝オッズ", "市場確率": "オッズから見た勝率", "人気比": "人気(頭数比)",
    "騎手3着内率": "騎手の3着内率", "騎手勝率": "騎手の勝率", "前走着順": "前走着順",
    "近走平均着順": "近5走の平均着順", "近走3着内率": "近5走の3着内率", "前走間隔": "レース間隔",
    "前走人気": "前走の人気", "前走単勝オッズ": "前走の単勝オッズ", "前走頭数": "前走の頭数",
    "前走着順率": "前走着順(頭数比)", "近走平均人気": "近5走の平均人気", "近走平均着順率": "近5走の着順(頭数比)",
    "近走人気超過": "近5走で人気以上に走った度合い", "近走数": "近走の出走数",
    "距離変化": "前走からの距離変化", "芝ダ替わり": "芝・ダート替わり", "クラス変化": "前走からのクラス変化",
    "斤量変化": "前走からの斤量変化",
    "前走タイム差": "前走の勝ち馬とのタイム差", "近走平均タイム差": "近5走の平均タイム差",
    "近走最小タイム差": "近5走で最も小さいタイム差", "前走上がり": "前走の上がり3F",
    "前走4角位置率": "前走の4角位置", "近走平均4角位置率": "近5走の4角位置",
    "斤量_対平均": "斤量(出走馬平均との差)", "騎手3着内率_対平均": "騎手の3着内率(出走馬平均との差)",
    "近走平均着順率_対平均": "近走着順(出走馬平均との差)", "近走平均タイム差_対平均": "近走タイム差(出走馬平均との差)",
}


# ---------------------------------------------------------------- HTML → 1頭1行

def _read_html_text(path):
    content = Path(path).read_bytes()
    for encoding in ("euc-jp", "cp932", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _parse_payout(payouts, ticket_type):
    """{馬番: 払戻金(100円あたり)}。"""
    out = {}
    for item in payouts.get(ticket_type, []):
        if str(item.get("combo", "")).strip().isdigit():
            out[int(item["combo"])] = item["pay"]
    return out


def parse_race_file(path):
    """レース結果HTML1件から、出走馬1頭=1行の辞書リストを返す(払戻も各行に付ける)。"""
    import scraper
    from scraping.build_dataset import _extract_result_rows
    from scraping.race_metadata import clean_html_comments, parse_race_metadata

    race_id = Path(path).stem
    try:
        html_text = _read_html_text(path)
        soup = scraper.BeautifulSoup(clean_html_comments(html_text), "lxml")
        result_rows = _extract_result_rows(soup)
        if not result_rows:
            return []
        meta = parse_race_metadata(html_text, race_id)
        payouts = scraper._extract_payouts_from_soup(soup)
    except Exception:
        return []

    win_pay = _parse_payout(payouts, "単勝")
    place_pay = _parse_payout(payouts, "複勝")
    rows = []
    for r in result_rows:
        umaban = pd.to_numeric(r.get("馬番"), errors="coerce")
        rows.append({
            "race_id": race_id,
            "日付": meta["日付"],
            "馬名": r.get("馬名"),
            "騎手": r.get("騎手"),
            "枠番": r.get("枠番"),
            "馬番": r.get("馬番"),
            "性齢": r.get("性齢"),
            "斤量": r.get("斤量"),
            "馬体重": r.get("馬体重"),
            "単勝": r.get("単勝"),
            "人気": r.get("人気"),
            "着順": r.get("着順"),
            "タイム": r.get("タイム"),
            "通過": r.get("通過"),
            "上り": r.get("上り"),
            "芝・ダート": meta["芝・ダート"],
            "距離": meta["距離"],
            "回り": meta["回り"],
            "馬場": meta["馬場"],
            "天気": meta["天気"],
            "場id": meta["場id"],
            "クラス": meta["クラス"],
            "単勝払戻": win_pay.get(int(umaban), 0) if pd.notna(umaban) else 0,
            "複勝払戻": place_pay.get(int(umaban), 0) if pd.notna(umaban) else 0,
        })
    return rows


def load_collected_rows(split, rebuild=False, workers=None):
    """data/{split}/race_html を解析した表を返す。解析結果はCSVにキャッシュする。"""
    cache = ROWS_CACHE[split]
    if cache.exists() and not rebuild:
        df = pd.read_csv(cache, encoding="utf-8-sig", dtype={"race_id": str, "場id": str})
        missing = [c for c in PAST_RUN_SOURCE_COLUMNS if c not in df.columns]
        if not missing:
            return df
        print(f"{cache} に {missing} の列がないため、HTMLを解析し直します。")

    html_dir = TRAIN_HTML_DIR if split == "train" else TEST_HTML_DIR
    paths = sorted(html_dir.glob("*.html"))
    if not paths:
        raise FileNotFoundError(f"{html_dir} にHTMLがありません(python -m scraping.collect_dataset で取得)。")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, race_rows in enumerate(pool.map(parse_race_file, paths, chunksize=50), start=1):
            rows.extend(race_rows)
            if i % 1000 == 0:
                print(f"  {split}: {i}/{len(paths)}レース解析済み")

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"{split}: {df['race_id'].nunique()}レース / {len(df)}頭 → {cache}")
    return df


# ---------------------------------------------------------------- 特徴量

def _num(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def _body_weight(series):
    parts = series.astype(str).str.extract(r"(\d{3,4})\s*(?:\(\s*([+-]?\d+)\s*\))?")
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def clean_jockey(name):
    return re.sub(r"^[▲△☆★◇◆]+", "", str(name or "")).strip()


def clean_horse_name(name):
    return re.sub(r"\s+", "", str(name or ""))


def _encode_turn(value):
    text = "" if pd.isna(value) else str(value).strip()
    return encode.encode_turn(text[:1]) if text else np.nan


def base_features(df):
    """出馬表の列(枠番・馬番・性齢・斤量・馬体重・単勝・人気・騎手)とレース条件から特徴量を作る。

    df は1頭1行で race_id 列を持つ。学習時(結果HTML)と予想時(出馬表)の両方から使う。
    """
    out = pd.DataFrame(index=df.index)
    out["枠番"] = _num(df["枠番"])
    out["馬番"] = _num(df["馬番"])
    out["頭数"] = df.groupby("race_id")["馬番"].transform("size").astype(float)
    out["斤量"] = _num(df["斤量"])
    out["体重"], out["体重変化"] = _body_weight(df["馬体重"])
    sex_age = df["性齢"].astype(str)
    out["性"] = sex_age.map(encode.encode_sex)
    out["齢"] = _num(sex_age.str[1:])

    odds = _num(df["単勝"])
    odds = odds.where(odds > 0)
    out["単勝オッズ"] = odds
    out["人気"] = _num(df["人気"])
    out["人気比"] = out["人気"] / out["頭数"]
    inv = 1.0 / odds
    out["市場確率"] = inv / inv.groupby(df["race_id"]).transform("sum")

    out["芝・ダート"] = df["芝・ダート"].map(encode.encode_surface)
    out["距離"] = pd.to_numeric(df["距離"], errors="coerce")
    out["回り"] = df["回り"].map(_encode_turn)
    out["馬場"] = df["馬場"].map(encode.encode_going)
    out["天気"] = df["天気"].map(encode.encode_weather)
    out["場id"] = pd.to_numeric(df["場id"], errors="coerce")
    out["クラス"] = df["クラス"].map(encode.encode_class)
    return out


# ---- 馬の近走特徴量(学習・予想で共通の計算)
#
# 「1頭の1走=1行」の表(RUN_COLUMNS)を作り、horse_history_features() に渡す。
# 学習時は収集データ全体、予想時は「戦績ページの過去走+今回の出走」を渡すので、計算式は1か所だけ。

RUN_COLUMNS = [
    "_horse", "_date", "_rank",  # 馬のキー・開催日・着順(その走の結果)
    "頭数", "人気", "単勝オッズ", "距離", "芝・ダート", "クラス", "斤量",
    "_margin", "_last3f", "_corner",  # 勝ち馬との秒差・上がり3F・最終コーナー位置(頭数比)
]
MARGIN_CLIP = 5.0  # 大差負け(追うのをやめた走)が平均を歪めないよう、タイム差は5秒で頭打ちにする


def _last_corner(value):
    """通過順「10-8-5」の最後(最終コーナー)の順位。"""
    if pd.isna(value):
        return np.nan
    return pd.to_numeric(str(value).strip().split("-")[-1], errors="coerce")


def _winner_margin(df):
    """走破タイムから勝ち馬との秒差を作る(勝ち馬は2着馬との差で負の値)。戦績ページの「着差」と同じ定義。"""
    seconds = df["タイム"].map(encode.parse_time_to_seconds)
    best = seconds.groupby(df["race_id"]).transform("min")
    runner_up = seconds.where(seconds > best).groupby(df["race_id"]).transform("min")
    margin = seconds - best
    sole_winner = (margin == 0) & ((margin == 0).groupby(df["race_id"]).transform("sum") == 1)
    return margin.where(~sole_winner, best - runner_up)


def horse_history_features(runs):
    """RUN_COLUMNS の表から、各行の「その走より前」の成績だけで近走特徴量を作る(行の並びは入力のまま)。

    各行自身の結果(_rank, _margin, _last3f, _corner, 人気など)はその行の特徴量には使わず、
    1走ずらした値(前走)と直近 HISTORY_WINDOW 走の集計だけを使う。
    今回の条件(距離・芝ダ・クラス・斤量)は前走との差を取るためにだけ参照する。
    """
    r = runs.sort_values(["_horse", "_date"], kind="stable")
    horse = r["_horse"]
    grp = r.groupby("_horse", sort=False)

    def prev(series):
        return series.groupby(horse, sort=False).shift(1)

    def recent(series, how="mean"):
        roll = series.groupby(horse, sort=False).rolling(HISTORY_WINDOW, min_periods=1)
        return getattr(roll, how)().reset_index(level=0, drop=True)

    prev_rank = prev(r["_rank"])
    prev_field = prev(r["頭数"])
    prev_pop = prev(r["人気"])
    prev_rank_rate = prev_rank / prev_field
    prev_margin = prev(r["_margin"]).clip(upper=MARGIN_CLIP)
    prev_corner = prev(r["_corner"])
    prev_surface = prev(r["芝・ダート"])

    out = pd.DataFrame(index=r.index)
    out["前走着順"] = prev_rank
    out["近走平均着順"] = recent(prev_rank)
    out["近走3着内率"] = recent((prev_rank <= 3).astype(float).where(prev_rank.notna()))
    out["前走間隔"] = (r["_date"] - prev(r["_date"])).dt.days

    out["前走人気"] = prev_pop
    out["前走単勝オッズ"] = prev(r["単勝オッズ"])
    out["前走頭数"] = prev_field
    out["前走着順率"] = prev_rank_rate
    out["近走平均人気"] = recent(prev_pop)
    out["近走平均着順率"] = recent(prev_rank_rate)
    out["近走人気超過"] = recent(prev_pop - prev_rank)  # 正なら人気より上の着順に来ている
    out["近走数"] = grp.cumcount().clip(upper=HISTORY_WINDOW).astype(float)

    out["距離変化"] = r["距離"] - prev(r["距離"])
    out["芝ダ替わり"] = (r["芝・ダート"] != prev_surface).astype(float).where(
        r["芝・ダート"].notna() & prev_surface.notna())
    out["クラス変化"] = r["クラス"] - prev(r["クラス"])
    out["斤量変化"] = r["斤量"] - prev(r["斤量"])

    out["前走タイム差"] = prev_margin
    out["近走平均タイム差"] = recent(prev_margin)
    out["近走最小タイム差"] = recent(prev_margin, "min")
    out["前走上がり"] = prev(r["_last3f"])
    out["前走4角位置率"] = prev_corner
    out["近走平均4角位置率"] = recent(prev_corner)
    return out.reindex(runs.index)


RELATIVE_SOURCE_COLUMNS = ["斤量", "騎手3着内率", "近走平均着順率", "近走平均タイム差"]


def add_race_relative_features(feats, race_id):
    """同じレースの出走馬の平均との差(相手関係)を付ける。取消馬を外す前の全出走馬で計算する。"""
    feats = feats.copy()
    for col in RELATIVE_SOURCE_COLUMNS:
        feats[f"{col}_対平均"] = feats[col] - feats[col].groupby(race_id).transform("mean")
    return feats


def _training_runs(df, feats):
    """学習用: 収集データ(add_jockey_history 済み)と base_features から RUN_COLUMNS の表を作る。"""
    runs = pd.DataFrame({"_horse": df["_horse"], "_date": df["_date"], "_rank": df["_rank"]})
    for col in ["頭数", "人気", "単勝オッズ", "距離", "芝・ダート", "クラス", "斤量"]:
        runs[col] = feats[col]
    runs["_margin"] = _winner_margin(df)
    runs["_last3f"] = _num(df["上り"])
    runs["_corner"] = df["通過"].map(_last_corner) / feats["頭数"]
    return runs[RUN_COLUMNS]


def _encode_results_page_class(race_name):
    """戦績ページのレース名「○○S(OP)」「3歳未勝利」などをクラスコードにする。"""
    code = encode.encode_class(race_name)
    if pd.isna(code) and "OP" in str(race_name):
        return float(encode.encode_class("オープン"))
    return code


def _past_runs_from_results(key, results, race_date):
    """予想用: 戦績ページの表(scraper.scrape_race_results_dedicated)から race_date より前の走を RUN_COLUMNS にする。"""
    if results is None or results.empty or "日付" not in results.columns or "着順" not in results.columns:
        return None

    def col(name):
        return results[name] if name in results.columns else pd.Series(np.nan, index=results.index)

    course = col("距離").astype(str).str.extract(r"(芝|ダ|障)\D*(\d{3,4})")
    field = _num(col("頭数"))
    runs = pd.DataFrame({
        "_horse": key,
        "_date": pd.to_datetime(results["日付"], errors="coerce"),
        "_rank": results["着順"].map(encode.parse_rank),
        "頭数": field,
        "人気": _num(col("人気")),
        "単勝オッズ": _num(col("オッズ")),
        "距離": pd.to_numeric(course[1], errors="coerce"),
        "芝・ダート": course[0].map({"芝": "芝", "ダ": "ダート", "障": "障害"}).map(encode.encode_surface),
        "クラス": col("レース名").map(_encode_results_page_class),
        "斤量": _num(col("斤量")),
        "_margin": _num(col("着差")),
        "_last3f": _num(col("上り")),
        "_corner": col("通過").map(_last_corner) / field,
    })
    runs = runs[runs["_date"] < race_date]
    return runs[RUN_COLUMNS] if len(runs) else None


def entry_history_features(feats, results_list, race_date):
    """予想用: 出走馬ごとの戦績ページ(results_list は feats と同じ並び)から近走特徴量を作る。

    今回の出走を「最新の1走」として過去走の後ろに並べ、学習時と同じ horse_history_features() で計算する。
    """
    race_date = pd.Timestamp(race_date)
    current = pd.DataFrame({"_horse": range(len(feats)), "_date": race_date, "_rank": np.nan})
    for col in ["頭数", "人気", "単勝オッズ", "距離", "芝・ダート", "クラス", "斤量"]:
        current[col] = feats[col].to_numpy(dtype=float)
    for col in ["_margin", "_last3f", "_corner"]:
        current[col] = np.nan
    current = current[RUN_COLUMNS]

    past = [_past_runs_from_results(key, results, race_date) for key, results in enumerate(results_list)]
    frames = [current] + [p.astype({c: float for c in RUN_COLUMNS[2:]}) for p in past if p is not None]
    history = horse_history_features(pd.concat(frames, ignore_index=True)).iloc[:len(current)]
    history.index = feats.index
    return history


def jockey_daily_table(df):
    """騎手×開催日ごとの騎乗数・1着数・3着内数。予想時の騎手成績の元データになる。"""
    work = pd.DataFrame({
        "騎手": df["騎手"].map(clean_jockey),
        "日付": pd.to_datetime(df["日付"], format="%Y/%m/%d", errors="coerce"),
        "rank": df["着順"].map(encode.parse_rank),
    }).dropna(subset=["日付"])
    work["騎乗"] = 1
    work["1着"] = (work["rank"] == 1).astype(int)
    work["3着内"] = (work["rank"] <= 3).astype(int)
    return work.groupby(["騎手", "日付"], as_index=False)[["騎乗", "1着", "3着内"]].sum()


def add_jockey_history(df):
    """学習用: 同じデータ内の過去レースから騎手成績を付け、馬・開催日順に並べ替える(当日以降は使わない)。"""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["日付"], format="%Y/%m/%d", errors="coerce")
    df["_rank"] = df["着順"].map(encode.parse_rank)
    df["_jockey"] = df["騎手"].map(clean_jockey)
    df["_horse"] = df["馬名"].map(clean_horse_name)

    # 騎手: 開催日単位で累積し、1日ずらして「その日より前」だけの成績にする
    daily = jockey_daily_table(df).sort_values(["騎手", "日付"])
    cum = daily.groupby("騎手")[["騎乗", "1着", "3着内"]].cumsum() - daily[["騎乗", "1着", "3着内"]]
    daily = daily.assign(騎手騎乗数=cum["騎乗"], _j_win=cum["1着"], _j_top3=cum["3着内"])
    df = df.merge(
        daily[["騎手", "日付", "騎手騎乗数", "_j_win", "_j_top3"]].rename(columns={"騎手": "_jockey", "日付": "_date"}),
        on=["_jockey", "_date"], how="left",
    )
    rides = df["騎手騎乗数"].where(df["騎手騎乗数"] > 0)
    df["騎手3着内率"] = df["_j_top3"] / rides
    df["騎手勝率"] = df["_j_win"] / rides

    # 馬の近走特徴量(同じ馬名の当日より前の走)は horse_history_features() が馬・開催日順で計算する
    df = df.sort_values(["_horse", "_date"]).reset_index(drop=True)
    return df.drop(columns=["_j_win", "_j_top3"])


def training_features(rows):
    """収集データ(1頭1行)に特徴量候補すべてを付け、(馬・開催日順に並べた表, 特徴量)を返す。"""
    df = add_jockey_history(rows)
    feats = base_features(df)
    for col in ["騎手騎乗数", "騎手3着内率", "騎手勝率"]:
        feats[col] = df[col]
    feats = feats.join(horse_history_features(_training_runs(df, feats)))
    return df, add_race_relative_features(feats, df["race_id"])


def build_training_table(rebuild=False):
    """train/test を結合して履歴特徴量を付け、split 列付きで返す(特徴量は候補すべて=ALL_FEATURE_COLUMNS)。"""
    train = load_collected_rows("train", rebuild=rebuild).assign(split="train")
    test = load_collected_rows("test", rebuild=rebuild).assign(split="test")
    # 頭数・市場確率・レース内相対は出馬表と同じく全出走馬で計算してから、取消・除外・中止の馬を外す
    df, feats = training_features(pd.concat([train, test], ignore_index=True))
    keep = df["_rank"].notna()
    df = df[keep].reset_index(drop=True)
    feats = feats[keep].reset_index(drop=True)
    df["top3"] = (df["_rank"] <= 3).astype(int)
    return df, feats[ALL_FEATURE_COLUMNS]


# ---------------------------------------------------------------- 学習・検証

def _pick_stats(df, score_col):
    """各レースで score_col が最大の1頭を買った場合の単勝・複勝の的中率と回収率(%)。"""
    idx = df.groupby("race_id")[score_col].idxmax().dropna()
    picks = df.loc[idx]
    n = len(picks)
    return {
        "レース数": n,
        "単勝的中率": round(float((picks["_rank"] == 1).mean()), 4),
        "単勝回収率": round(float(picks["単勝払戻"].sum() / (n * 100) * 100), 1),
        "複勝的中率": round(float((picks["_rank"] <= 3).mean()), 4),
        "複勝回収率": round(float(picks["複勝払戻"].sum() / (n * 100) * 100), 1),
    }


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1,
)


def fit_model(x, y, dates, train_mask, seed=42):
    """train_mask(真偽の配列)の行で学習し、(モデル, 木の本数)を返す。

    木の本数は、学習データの最後の15%(日付順)を検証にした早期終了で決め、その本数で全学習データを学習し直す。
    """
    import lightgbm as lgb

    params = dict(LGB_PARAMS, random_state=seed)
    valid_start = dates[train_mask].quantile(0.85)
    fit_mask = train_mask & (dates < valid_start).to_numpy()
    valid_mask = train_mask & ~fit_mask

    probe = lgb.LGBMClassifier(n_estimators=3000, **params)
    probe.fit(x[fit_mask], y[fit_mask], eval_set=[(x[valid_mask], y[valid_mask])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = max(int(probe.best_iteration_ or 0), 50)

    model = lgb.LGBMClassifier(n_estimators=best_iter, **params)
    model.fit(x[train_mask], y[train_mask])
    return model, best_iter


def _fit_and_evaluate(df, x_all, features):
    """features で学習(train)→テスト期間で検証し、(モデル, 指標) を返す。"""
    from sklearn.metrics import log_loss, roc_auc_score

    features = list(features)
    x = x_all[features]
    is_train = (df["split"] == "train").to_numpy()
    y = df["top3"]
    train_dates = df.loc[is_train, "_date"]

    model, best_iter = fit_model(x, y, df["_date"], is_train)

    test = df[~is_train].copy()
    test["model_proba"] = model.predict_proba(x[~is_train])[:, 1]
    test["popularity_score"] = -pd.to_numeric(test["人気"], errors="coerce")
    test = test.dropna(subset=["popularity_score"])

    model_stats = _pick_stats(test, "model_proba")
    baseline_stats = _pick_stats(test, "popularity_score")
    auc = roc_auc_score(test["top3"], test["model_proba"])
    baseline_auc = roc_auc_score(test["top3"], test["popularity_score"])

    # 採用基準: 3着以内の当てやすさ(AUC)と、1位予想馬の複勝回収率の両方で1番人気買いに負けないこと
    passed = bool(auc >= baseline_auc and model_stats["複勝回収率"] >= baseline_stats["複勝回収率"])

    importance = pd.Series(model.booster_.feature_importance("gain"), index=features)
    metrics = {
        "学習レース数": int(df.loc[is_train, "race_id"].nunique()),
        "テストレース数": int(test["race_id"].nunique()),
        "学習期間": [str(train_dates.min().date()), str(train_dates.max().date())],
        "テスト期間": [str(test["_date"].min().date()), str(test["_date"].max().date())],
        "木の本数": best_iter,
        "AUC": round(float(auc), 4),
        "AUC(人気順)": round(float(baseline_auc), 4),
        "logloss": round(float(log_loss(test["top3"], test["model_proba"])), 4),
        "モデル1位馬": model_stats,
        "1番人気": baseline_stats,
        "検証合格": passed,
        "採用": None,  # 呼び出し側で決める
        "採用理由": None,
        "特徴量": features,
        "除外した結果系の列": sorted(POST_RACE_COLUMNS),
        "特徴量重要度(gain%)": (importance / importance.sum() * 100).round(1).sort_values(ascending=False).to_dict(),
    }
    return model, metrics


def _save_jockey_stats():
    all_rows = pd.concat([load_collected_rows("train"), load_collected_rows("test")], ignore_index=True)
    jockey_daily_table(all_rows).to_csv(JOCKEY_STATS_PATH, index=False, encoding="utf-8-sig")


def train_and_validate(rebuild=False, force=False):
    df, x = build_training_table(rebuild=rebuild)
    model, metrics = _fit_and_evaluate(df, x, FEATURE_COLUMNS)
    passed = metrics["検証合格"]
    adopted = passed or force
    metrics["採用"] = adopted
    metrics["採用理由"] = "検証合格" if passed else ("--force 指定" if force else "1番人気買いを上回れなかったため不採用")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(CANDIDATE_PATH))
    if adopted:
        model.booster_.save_model(str(MODEL_PATH))
        _save_jockey_stats()
        METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        CANDIDATE_PATH.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


# ---- 説明変数を選び直したモデル(アプリの「出馬表モデル」で選んで学習・予想する)

_TRAINING_TABLE = None  # 同じプロセスで何度も学習するとき、特徴量表を作り直さない


def _variant_paths(features):
    import hashlib

    key = hashlib.sha1("|".join(sorted(features)).encode("utf-8")).hexdigest()[:12]
    return VARIANT_DIR / f"{key}.txt", VARIANT_DIR / f"{key}.metrics.json"


def load_model_for_features(features, base_bundle=None):
    """選んだ説明変数で学習済みのモデル(load_adopted_model と同じ形)を返す。未学習なら None。

    採用済みモデルと同じ組み合わせなら base_bundle をそのまま返す。
    """
    import lightgbm as lgb

    features = list(features)
    if base_bundle is not None and set(features) == set(base_bundle["booster"].feature_name()):
        return base_bundle
    model_path, metrics_path = _variant_paths(features)
    if not model_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    jockey_daily = base_bundle["jockey_daily"] if base_bundle is not None else _load_jockey_daily()
    return {"booster": lgb.Booster(model_file=str(model_path)), "metrics": metrics, "jockey_daily": jockey_daily}


def train_model_for_features(features):
    """選んだ説明変数でモデルを学習・テスト期間で検証し、VARIANT_DIR に保存して指標を返す。

    採用基準による選別はしない(選んだ組み合わせのモデルを必ず保存し、検証成績は画面に出す)。
    """
    global _TRAINING_TABLE

    selected = set(features)
    features = [c for c in ALL_FEATURE_COLUMNS if c in selected]
    if not features:
        raise ValueError("説明変数を1つ以上選んでください。")
    if _TRAINING_TABLE is None:
        _TRAINING_TABLE = build_training_table()
    df, x = _TRAINING_TABLE
    model, metrics = _fit_and_evaluate(df, x, features)
    metrics["採用"] = True
    metrics["採用理由"] = "アプリで選んだ説明変数で学習" + ("" if metrics["検証合格"] else "(1番人気買いは上回れず)")

    model_path, metrics_path = _variant_paths(features)
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not JOCKEY_STATS_PATH.exists():
        _save_jockey_stats()
    return metrics


# ---------------------------------------------------------------- 予想(app.pyから使う)

def load_adopted_model():
    """採用済みモデル・指標・騎手成績を読み込む。未採用なら None。"""
    if not MODEL_PATH.exists():
        return None
    import lightgbm as lgb

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return {"booster": lgb.Booster(model_file=str(MODEL_PATH)), "metrics": metrics, "jockey_daily": _load_jockey_daily()}


def _load_jockey_daily():
    jockey_daily = pd.DataFrame(columns=["騎手", "日付", "騎乗", "1着", "3着内"])
    if JOCKEY_STATS_PATH.exists():
        jockey_daily = pd.read_csv(JOCKEY_STATS_PATH, encoding="utf-8-sig")
        jockey_daily["日付"] = pd.to_datetime(jockey_daily["日付"], errors="coerce")
    return jockey_daily


def predict_shutuba(bundle, race_id, race_info, shutuba_df, all_horse_data, race_date):
    """出馬表の各馬の3着以内確率を返す(馬番・馬名・3着内確率・根拠)。"""
    from model.live_predict import parse_race_conditions

    if shutuba_df is None or shutuba_df.empty:
        raise ValueError("出馬表がありません。")
    race_date = pd.Timestamp(race_date)
    cond = parse_race_conditions(race_info, race_id, race_date)
    class_text = " ".join(str((race_info or {}).get(k, "")) for k in ["race_class", "name", "detail"])

    entries = pd.DataFrame({
        "race_id": race_id,
        "枠番": shutuba_df.get("枠", shutuba_df.get("枠番")),
        "馬番": shutuba_df.get("馬番"),
        "性齢": shutuba_df.get("性齢", ""),
        "斤量": shutuba_df.get("斤量"),
        "馬体重": shutuba_df.get("馬体重", ""),
        "単勝": shutuba_df.get("オッズ", shutuba_df.get("単勝")),
        "人気": shutuba_df.get("人気"),
        "芝・ダート": cond["surface"],
        "距離": cond["distance"],
        "回り": cond["turn"],
        "馬場": cond["going"],
        "天気": cond["weather"],
        "場id": cond["place_id"],
        "クラス": class_text,
    }, index=shutuba_df.index)
    # 取消などで馬番が空の行は頭数に数えない
    entries = entries[_num(entries["馬番"]).notna()]
    feats = base_features(entries)

    daily = bundle["jockey_daily"]
    daily = daily[daily["日付"] < race_date]
    jockey_totals = daily.groupby("騎手")[["騎乗", "1着", "3着内"]].sum()
    jockeys = shutuba_df.loc[entries.index, "騎手"].map(clean_jockey) if "騎手" in shutuba_df else pd.Series("", index=entries.index)
    stats = jockey_totals.reindex(jockeys.to_numpy())
    rides = stats["騎乗"].to_numpy(dtype=float)
    feats["騎手騎乗数"] = np.nan_to_num(rides, nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["騎手3着内率"] = np.where(rides > 0, stats["3着内"].to_numpy(dtype=float) / rides, np.nan)
        feats["騎手勝率"] = np.where(rides > 0, stats["1着"].to_numpy(dtype=float) / rides, np.nan)

    results_by_name = {clean_horse_name(h["pedigree"].get("name")): h.get("results") for h in (all_horse_data or [])}
    names = shutuba_df.loc[entries.index, "馬名"].map(clean_horse_name)
    feats = feats.join(entry_history_features(feats, [results_by_name.get(name) for name in names], race_date))
    feats = add_race_relative_features(feats, entries["race_id"])

    booster = bundle["booster"]
    x = feats[booster.feature_name()].apply(pd.to_numeric, errors="coerce")
    proba = booster.predict(x)
    contrib = booster.predict(x, pred_contrib=True)[:, :-1]
    reasons = []
    for row in contrib:
        top = sorted(zip(booster.feature_name(), row), key=lambda t: t[1], reverse=True)[:3]
        reasons.append(" / ".join(FEATURE_DISPLAY_NAMES.get(n, n) for n, v in top if v > 0))

    out = pd.DataFrame({
        "馬番": _num(entries["馬番"]).astype(int),
        "馬名": names,
        "単勝オッズ": feats["単勝オッズ"],
        "人気": feats["人気"],
        "3着内確率": proba,
        "根拠": reasons,
    }, index=entries.index)
    out = out.sort_values("3着内確率", ascending=False).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return {
        "predictions": out,
        "odds_missing": bool(feats["単勝オッズ"].isna().all()),
        # 馬体重を使うモデルなのに出馬表に馬体重がない(未発表)とき。確率は参考値になる。
        "weight_missing": bool("体重" in booster.feature_name() and feats["体重"].isna().all()),
        "features": list(booster.feature_name()),
    }


def format_for_prompt(result, metrics):
    """AIプロンプトに渡す文字列。"""
    if not result:
        return ""
    pred = result["predictions"]
    lines = ["【学習済みモデル(出馬表モデル)の3着以内確率】"]
    if metrics:
        m, b = metrics.get("モデル1位馬", {}), metrics.get("1番人気", {})
        lines.append(
            f"過去{metrics.get('学習レース数', '?')}レースで学習し、別期間の{metrics.get('テストレース数', '?')}レースで検証済み"
            f"(1位予想馬の複勝的中率{m.get('複勝的中率', 0):.1%}・複勝回収率{m.get('複勝回収率', 0)}% / "
            f"1番人気は{b.get('複勝的中率', 0):.1%}・{b.get('複勝回収率', 0)}%)。"
        )
    if result.get("odds_missing"):
        lines.append("※オッズ未発表のため、オッズ・人気なしで計算した参考値です。")
    if result.get("weight_missing"):
        lines.append("※馬体重が未発表のため、馬体重なしで計算した参考値です。")
    elif result.get("features") and "体重" not in result["features"]:
        lines.append("※馬体重を使わないモデルで計算しています。")
    lines.append("順位|馬番|馬名|3着内確率|主な根拠")
    for _, r in pred.iterrows():
        lines.append(f"{r['順位']}|{r['馬番']}|{r['馬名']}|{r['3着内確率']:.1%}|{r['根拠']}")
    lines.append("この確率は数値データのみの評価です。血統・展開・当日の状態などと合わせて総合判断してください。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="収集したレースで出馬表モデルを学習し、テスト期間で検証する")
    parser.add_argument("--rebuild", action="store_true", help="HTMLの解析をやり直す")
    parser.add_argument("--force", action="store_true", help="検証結果に関わらずアプリで使うモデルとして採用する")
    parser.add_argument("--no-weight", action="store_true",
                        help="採用済みモデルから馬体重(体重・体重変化)を外したモデルを学習し、アプリ用に保存する")
    args = parser.parse_args()

    if args.no_weight:
        bundle = load_adopted_model()
        base = bundle["booster"].feature_name() if bundle else FEATURE_COLUMNS
        metrics = train_model_for_features([c for c in base if c not in WEIGHT_FEATURES])
        print(json.dumps({k: v for k, v in metrics.items() if k != "特徴量重要度(gain%)"}, ensure_ascii=False, indent=2))
        return

    metrics = train_and_validate(rebuild=args.rebuild, force=args.force)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("特徴量", "特徴量重要度(gain%)")},
                     ensure_ascii=False, indent=2))
    print("--- 特徴量重要度(gain%) Top10 ---")
    for name, pct in list(metrics["特徴量重要度(gain%)"].items())[:10]:
        print(f"  {name}: {pct}%")
    if metrics["採用"]:
        print(f"採用: {MODEL_PATH}(アプリの単一レース予想で使われます)")
    else:
        print(f"不採用: 候補モデルは {CANDIDATE_PATH} に保存しました。--force で採用できます。")


if __name__ == "__main__":
    main()

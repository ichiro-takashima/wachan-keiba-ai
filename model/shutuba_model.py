"""収集済みレースHTML(data/train, data/test)から「レース前に分かる情報だけ」で
各馬の3着以内確率を出すLightGBMモデル(出馬表モデル)。

- 学習: data/train/race_html のレースで学習する
- 検証: data/test/race_html のレース(学習より後の期間)で的中率・回収率を測り、
        1番人気を買い続けた場合より良いときだけ採用(data/models/shutuba_model.txt)する
- 予想: app.py の単一レース予想から predict_shutuba() を呼び出す

走破タイム・上がり3F・通過順・着順はレースが終わらないと分からないため特徴量に使わない
(着順は目的変数と、その馬の「次のレース」の前走成績としてだけ使う)。

使い方:
    python -m model.shutuba_model            # データ作成 → 学習 → テスト検証 → 合格なら採用
    python -m model.shutuba_model --force    # 検証結果に関わらず採用
    python -m model.shutuba_model --rebuild  # HTMLの解析をやり直す(キャッシュCSVを作り直す)
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

HISTORY_WINDOW = 5

# レース後にしか分からない列。学習データには残るが、特徴量には絶対に入れない。
POST_RACE_COLUMNS = {"走破時間", "上がり", "通過順", "着順"}

FEATURE_COLUMNS = [
    "枠番", "馬番", "頭数", "斤量", "体重", "体重変化", "性", "齢",
    "単勝オッズ", "人気", "人気比", "市場確率",
    "芝・ダート", "距離", "回り", "馬場", "天気", "場id", "クラス",
    "騎手騎乗数", "騎手3着内率", "騎手勝率",
    "前走着順", "近走平均着順", "近走3着内率", "前走間隔",
]
assert not POST_RACE_COLUMNS & set(FEATURE_COLUMNS)

FEATURE_DISPLAY_NAMES = {
    "単勝オッズ": "単勝オッズ", "市場確率": "オッズから見た勝率", "人気比": "人気(頭数比)",
    "騎手3着内率": "騎手の3着内率", "騎手勝率": "騎手の勝率", "前走着順": "前走着順",
    "近走平均着順": "近5走の平均着順", "近走3着内率": "近5走の3着内率", "前走間隔": "レース間隔",
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
        return pd.read_csv(cache, encoding="utf-8-sig", dtype={"race_id": str, "場id": str})

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


def _horse_history_from_ranks(dates, ranks, race_date):
    """過去走の(日付, 着順)から前走着順・近走平均着順・近走3着内率・前走間隔を作る。"""
    past = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates, dtype=object), errors="coerce"),
                         "rank": pd.to_numeric(pd.Series(ranks, dtype=object), errors="coerce")})
    past = past[past["date"] < race_date].sort_values("date", ascending=False).head(HISTORY_WINDOW)
    if past.empty:
        return {"前走着順": np.nan, "近走平均着順": np.nan, "近走3着内率": np.nan, "前走間隔": np.nan}
    ranks = past["rank"].dropna()
    return {
        "前走着順": past["rank"].iloc[0],
        "近走平均着順": ranks.mean() if len(ranks) else np.nan,
        "近走3着内率": (ranks <= 3).mean() if len(ranks) else np.nan,
        "前走間隔": float((race_date - past["date"].iloc[0]).days),
    }


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


def add_history_features(df):
    """学習用: 同じデータ内の過去レースから騎手成績・馬の近走成績を付ける(当日以降は使わない)。"""
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

    # 馬: 同じ馬名の直近5走(当日より前)
    df = df.sort_values(["_horse", "_date"]).reset_index(drop=True)
    grp = df.groupby("_horse")
    prev_rank = grp["_rank"].shift(1)
    df["前走着順"] = prev_rank
    df["前走間隔"] = (df["_date"] - grp["_date"].shift(1)).dt.days
    df["近走平均着順"] = prev_rank.groupby(df["_horse"]).transform(lambda s: s.rolling(HISTORY_WINDOW, min_periods=1).mean())
    prev_top3 = (prev_rank <= 3).astype(float).where(prev_rank.notna())
    df["近走3着内率"] = prev_top3.groupby(df["_horse"]).transform(lambda s: s.rolling(HISTORY_WINDOW, min_periods=1).mean())
    return df.drop(columns=["_j_win", "_j_top3"])


def build_training_table(rebuild=False):
    """train/test を結合して履歴特徴量を付け、split 列付きで返す。"""
    train = load_collected_rows("train", rebuild=rebuild).assign(split="train")
    test = load_collected_rows("test", rebuild=rebuild).assign(split="test")
    df = add_history_features(pd.concat([train, test], ignore_index=True))
    # 頭数・市場確率は出馬表と同じく全出走馬で計算してから、取消・除外・中止の馬を外す
    feats = base_features(df)
    for col in ["騎手騎乗数", "騎手3着内率", "騎手勝率", "前走着順", "近走平均着順", "近走3着内率", "前走間隔"]:
        feats[col] = df[col]
    keep = df["_rank"].notna()
    df = df[keep].reset_index(drop=True)
    feats = feats[keep].reset_index(drop=True)
    df["top3"] = (df["_rank"] <= 3).astype(int)
    return df, feats[FEATURE_COLUMNS]


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


def train_and_validate(rebuild=False, force=False):
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score

    df, x = build_training_table(rebuild=rebuild)
    is_train = (df["split"] == "train").to_numpy()
    y = df["top3"]

    # 学習データの最後の15%(日付順)で早期終了の回数を決める
    train_dates = df.loc[is_train, "_date"]
    valid_start = train_dates.quantile(0.85)
    fit_mask = is_train & (df["_date"] < valid_start).to_numpy()
    valid_mask = is_train & ~fit_mask

    params = dict(
        objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbosity=-1,
    )
    probe = lgb.LGBMClassifier(n_estimators=3000, **params)
    probe.fit(x[fit_mask], y[fit_mask], eval_set=[(x[valid_mask], y[valid_mask])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = max(int(probe.best_iteration_ or 0), 50)

    model = lgb.LGBMClassifier(n_estimators=best_iter, **params)
    model.fit(x[is_train], y[is_train])

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
    adopted = passed or force

    importance = pd.Series(model.booster_.feature_importance("gain"), index=FEATURE_COLUMNS)
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
        "採用": adopted,
        "採用理由": "検証合格" if passed else ("--force 指定" if force else "1番人気買いを上回れなかったため不採用"),
        "特徴量": FEATURE_COLUMNS,
        "除外した結果系の列": sorted(POST_RACE_COLUMNS),
        "特徴量重要度(gain%)": (importance / importance.sum() * 100).round(1).sort_values(ascending=False).to_dict(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(CANDIDATE_PATH))
    if adopted:
        model.booster_.save_model(str(MODEL_PATH))
        all_rows = pd.concat([load_collected_rows("train"), load_collected_rows("test")], ignore_index=True)
        jockey_daily_table(all_rows).to_csv(JOCKEY_STATS_PATH, index=False, encoding="utf-8-sig")
        METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        CANDIDATE_PATH.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------- 予想(app.pyから使う)

def load_adopted_model():
    """採用済みモデル・指標・騎手成績を読み込む。未採用なら None。"""
    if not MODEL_PATH.exists():
        return None
    import lightgbm as lgb

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    jockey_daily = pd.DataFrame(columns=["騎手", "日付", "騎乗", "1着", "3着内"])
    if JOCKEY_STATS_PATH.exists():
        jockey_daily = pd.read_csv(JOCKEY_STATS_PATH, encoding="utf-8-sig")
        jockey_daily["日付"] = pd.to_datetime(jockey_daily["日付"], errors="coerce")
    return {"booster": lgb.Booster(model_file=str(MODEL_PATH)), "metrics": metrics, "jockey_daily": jockey_daily}


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
    history = []
    for name in names:
        res = results_by_name.get(name)
        if res is None or res.empty or "日付" not in res.columns or "着順" not in res.columns:
            history.append(_horse_history_from_ranks([], [], race_date))
            continue
        dates = pd.to_datetime(res["日付"], errors="coerce")
        history.append(_horse_history_from_ranks(dates, res["着順"].map(encode.parse_rank), race_date))
    history = pd.DataFrame(history, index=entries.index)
    for col in history.columns:
        feats[col] = history[col]

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
    return {"predictions": out, "odds_missing": bool(feats["単勝オッズ"].isna().all())}


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
    lines.append("順位|馬番|馬名|3着内確率|主な根拠")
    for _, r in pred.iterrows():
        lines.append(f"{r['順位']}|{r['馬番']}|{r['馬名']}|{r['3着内確率']:.1%}|{r['根拠']}")
    lines.append("この確率は数値データのみの評価です。血統・展開・当日の状態などと合わせて総合判断してください。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="収集したレースで出馬表モデルを学習し、テスト期間で検証する")
    parser.add_argument("--rebuild", action="store_true", help="HTMLの解析をやり直す")
    parser.add_argument("--force", action="store_true", help="検証結果に関わらずアプリで使うモデルとして採用する")
    args = parser.parse_args()

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

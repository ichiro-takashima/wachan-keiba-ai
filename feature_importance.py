from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

from data_features import AI_DATA_FEATURES

FEATURE_DEFINITIONS = AI_DATA_FEATURES
LEAKAGE_FEATURES = {
    "result_rank",
}
FEATURE_COLUMNS = [key for key, _ in FEATURE_DEFINITIONS if key not in LEAKAGE_FEATURES]
FEATURE_LABELS = dict(FEATURE_DEFINITIONS)

NUMERIC_FEATURES = {
    "shutuba_umaban",
    "shutuba_waku",
    "shutuba_weight_carried",
    "shutuba_odds",
    "shutuba_popularity",
    "shutuba_body_weight",
    "analysis_latest_weight",
    "result_popularity",
    "result_weight_carried",
    "result_distance",
    "result_time",
    "result_last3f",
    "result_body_weight",
}


@dataclass
class FeatureImportanceResult:
    dataset: pd.DataFrame
    importance: pd.DataFrame
    target_col: str
    output_csv: Path
    output_svg: Path


def _read_html_text(path: Path) -> str:
    content = path.read_bytes()
    for encoding in ("euc-jp", "cp932", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _normalize_col(value) -> str:
    return str(value).replace(" ", "").replace("　", "").replace("\n", "").strip()


def _to_number(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else pd.NA


def _parse_rank(value):
    if pd.isna(value):
        return pd.NA
    text = str(value)
    if any(mark in text for mark in ["中", "除", "取", "失"]):
        return pd.NA
    return _to_number(text)


def _parse_sex_age(value):
    text = "" if pd.isna(value) else str(value)
    sex = text[:1] if text else ""
    age = _to_number(text[1:])
    sex_map = {"牡": 0, "牝": 1, "セ": 2, "せ": 2, "騙": 2}
    return sex_map.get(sex, -1), age


def _parse_body_weight(value):
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"(\d+)(?:\(([+-]?\d+)\))?", text)
    if not match:
        return pd.NA, pd.NA
    weight = float(match.group(1))
    diff = float(match.group(2)) if match.group(2) is not None else pd.NA
    return weight, diff


def _parse_time_seconds(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text == "nan":
        return pd.NA
    match = re.search(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text)
    if not match:
        return pd.NA
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def _extract_first_corner(value):
    if pd.isna(value):
        return pd.NA
    match = re.search(r"\d+", str(value))
    return float(match.group(0)) if match else pd.NA


def _extract_race_name(text: str, fallback: str) -> str:
    title_match = re.search(r"<title>\s*(.*?)\s*</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not title_match:
        return fallback
    title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    return re.split(r"[|｜]", title)[0].strip() or fallback


def _find_result_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    for table in tables:
        df = table.copy()
        df.columns = [_normalize_col(col[-1] if isinstance(col, tuple) else col) for col in df.columns]
        normalized = set(df.columns)
        if {"着順", "馬番", "馬名"}.issubset(normalized):
            return df
    return None


def build_feature_dataset(html_dir: str | Path = "data/race_html") -> pd.DataFrame:
    html_path = Path(html_dir)
    rows = []

    for path in sorted(html_path.glob("*.html")):
        text = _read_html_text(path)
        race_name = _extract_race_name(text, path.stem)
        try:
            tables = pd.read_html(StringIO(text))
        except Exception:
            continue

        result_df = _find_result_table(tables)
        if result_df is None or result_df.empty:
            continue

        field_size = len(result_df)
        winner_df = result_df.copy()
        winner_df["_rank_num"] = winner_df.get("着順", pd.Series(dtype=object)).map(_parse_rank)
        winner_df["_pop_num"] = winner_df.get("人気", pd.Series(dtype=object)).map(_to_number)
        top3_df = winner_df[winner_df["_rank_num"].isin([1, 2, 3])]
        past_trend_rows_value = len(top3_df)
        past_trend_summary_value = top3_df["_pop_num"].dropna().mean() if not top3_df.empty else pd.NA

        for _, row in result_df.iterrows():
            rank = _parse_rank(row.get("着順"))
            if pd.isna(rank):
                continue

            sex_code, age = _parse_sex_age(row.get("性齢"))
            body_weight, body_diff = _parse_body_weight(row.get("馬体重"))
            distance = _to_number(row.get("距離"))
            result_time = _parse_time_seconds(row.get("タイム"))
            result_last3f = _to_number(row.get("上り"))
            rows.append({
                "race_id": path.stem,
                "馬名": row.get("馬名"),
                "着順": int(rank),
                "top3": 1 if rank <= 3 else 0,
                "win": 1 if rank == 1 else 0,
                "race_name": race_name,
                "race_condition": row.get("距離", ""),
                "past_trend_rows": past_trend_rows_value,
                "past_trend_summary": past_trend_summary_value,
                "shutuba_umaban": _to_number(row.get("馬番")),
                "shutuba_waku": _to_number(row.get("枠番")),
                "shutuba_name": row.get("馬名"),
                "shutuba_sex_age": row.get("性齢"),
                "shutuba_jockey": row.get("騎手"),
                "shutuba_weight_carried": _to_number(row.get("斤量")),
                "shutuba_odds": _to_number(row.get("単勝")),
                "shutuba_popularity": _to_number(row.get("人気")),
                "shutuba_stable": row.get("調教師", ""),
                "shutuba_body_weight": body_weight,
                "blood_sire": "",
                "blood_dam": "",
                "blood_broodmare_sire": "",
                "analysis_running_style": _extract_first_corner(row.get("通過")),
                "analysis_track_preference": row.get("馬場", ""),
                "analysis_latest_weight": body_weight,
                "result_date": "",
                "result_race_name": race_name,
                "result_place": "",
                "result_rank": rank,
                "result_popularity": _to_number(row.get("人気")),
                "result_jockey": row.get("騎手"),
                "result_weight_carried": _to_number(row.get("斤量")),
                "result_distance": distance,
                "result_track": row.get("馬場", ""),
                "result_time": result_time,
                "result_last3f": result_last3f,
                "result_pace": row.get("ペース", ""),
                "result_corner": _extract_first_corner(row.get("通過")),
                "result_body_weight": body_weight,
            })

    dataset = pd.DataFrame(rows)
    if not dataset.empty:
        for col in FEATURE_COLUMNS:
            if col not in dataset.columns:
                dataset[col] = pd.NA
    return dataset


def train_lightgbm_feature_importance(
    dataset: pd.DataFrame,
    target_col: str = "top3",
) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm がインストールされていません。`pip install lightgbm` を実行してください。") from exc

    if dataset.empty:
        raise ValueError("学習データが空です。data/race_html にレースHTMLを保存してください。")
    if target_col not in dataset.columns:
        raise ValueError(f"目的変数 {target_col} が見つかりません。")
    if dataset[target_col].nunique() < 2:
        raise ValueError(f"目的変数 {target_col} が1種類しかないため、重要度を計算できません。")

    train_df = dataset[FEATURE_COLUMNS + [target_col]].copy()
    for col in FEATURE_COLUMNS:
        if col in NUMERIC_FEATURES:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            if train_df[col].isna().all():
                train_df[col] = 0
            else:
                train_df[col] = train_df[col].fillna(train_df[col].median())
        else:
            normalized = train_df[col].fillna("__missing__").astype(str)
            train_df[col] = pd.factorize(normalized, sort=True)[0]

    x = train_df[FEATURE_COLUMNS]
    y = train_df[target_col].astype(int)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=3,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x, y)

    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "label": [FEATURE_LABELS.get(feature, feature) for feature in FEATURE_COLUMNS],
        "gain_importance": gain,
        "split_importance": split,
    })
    importance["importance"] = importance["gain_importance"]
    if importance["importance"].sum() <= 0:
        importance["importance"] = importance["split_importance"]
    total = importance["importance"].sum()
    importance["importance_pct"] = (importance["importance"] / total * 100) if total else 0
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_importance_svg(importance: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    view = importance.head(top_n).sort_values("importance", ascending=True)

    width = 920
    row_h = 34
    left = 180
    right = 90
    top = 34
    height = top + row_h * max(len(view), 1) + 30
    max_val = float(view["importance"].max()) if not view.empty else 1.0
    max_val = max(max_val, 1.0)
    bar_max = width - left - right

    def esc(text):
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="24" font-family="Meiryo, Arial, sans-serif" font-size="18" font-weight="700">LightGBM Feature Importance</text>',
    ]
    for i, row in enumerate(view.itertuples(index=False), start=0):
        y = top + i * row_h + 22
        bar_w = int(float(row.importance) / max_val * bar_max)
        label = getattr(row, "label", row.feature)
        parts.append(f'<text x="24" y="{y}" font-family="Meiryo, Arial, sans-serif" font-size="14" fill="#1f2937">{esc(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y - 16}" width="{bar_w}" height="20" rx="3" fill="#2563eb"/>')
        parts.append(f'<text x="{left + bar_w + 8}" y="{y}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{float(row.importance_pct):.1f}%</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def run_feature_importance(
    html_dir: str | Path = "data/race_html",
    output_dir: str | Path = "data/feature_importance",
    target_col: str = "top3",
) -> FeatureImportanceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_feature_dataset(html_dir)
    importance = train_lightgbm_feature_importance(dataset, target_col=target_col)

    output_csv = output_dir / f"lightgbm_feature_importance_{target_col}.csv"
    output_svg = output_dir / f"lightgbm_feature_importance_{target_col}.svg"
    importance.to_csv(output_csv, index=False, encoding="utf-8-sig")
    save_importance_svg(importance, output_svg)

    return FeatureImportanceResult(
        dataset=dataset,
        importance=importance,
        target_col=target_col,
        output_csv=output_csv,
        output_svg=output_svg,
    )

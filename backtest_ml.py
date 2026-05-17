from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RESULT_COLUMNS = [
    "Timestamp",
    "RaceID",
    "Model",
    "TotalBet",
    "TotalReturn",
    "Hits",
    "HitRate",
    "ReturnRate",
    "BetsJSON",
]


@dataclass
class BacktestTrendResult:
    dataset: pd.DataFrame
    candidates: pd.DataFrame
    context_text: str
    model_name: str


def _parse_percent(value) -> float:
    text = str(value).replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_int(value) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return 0


def _race_parts(race_id: str) -> dict:
    race_id = str(race_id)
    return {
        "year": _parse_int(race_id[:4]),
        "venue": race_id[4:6] if len(race_id) >= 6 else "",
        "meet": race_id[6:8] if len(race_id) >= 8 else "",
        "day": race_id[8:10] if len(race_id) >= 10 else "",
        "race_no": race_id[10:12] if len(race_id) >= 12 else "",
        "race_suffix": race_id[4:] if len(race_id) > 4 else race_id,
    }


def _combo_numbers(combo: str) -> list[int]:
    return [_parse_int(x) for x in re.findall(r"\d+", str(combo))]


def _combo_features(combo: str) -> dict:
    nums = _combo_numbers(combo)
    if not nums:
        return {
            "combo_size": 0,
            "min_num": 0,
            "max_num": 0,
            "avg_num": 0.0,
            "spread": 0,
        }
    return {
        "combo_size": len(nums),
        "min_num": min(nums),
        "max_num": max(nums),
        "avg_num": sum(nums) / len(nums),
        "spread": max(nums) - min(nums),
    }


def _normalize_combo(combo: str) -> str:
    nums = _combo_numbers(combo)
    if nums:
        return "-".join(str(n) for n in nums)
    return str(combo).strip()


def load_backtest_dataset(csv_path: str | Path = "backtest_results.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not raw_line.strip() or raw_line.startswith("Timestamp,"):
            continue
        parts = raw_line.split(",", 8)
        if len(parts) < len(RESULT_COLUMNS):
            continue
        row = dict(zip(RESULT_COLUMNS, parts))
        row["RaceID"] = str(row["RaceID"]).strip()
        row["TotalBet"] = _parse_int(row["TotalBet"])
        row["TotalReturn"] = _parse_int(row["TotalReturn"])
        row["Hits"] = _parse_int(row["Hits"])
        row["HitRateNum"] = _parse_percent(row["HitRate"])
        row["ReturnRateNum"] = _parse_percent(row["ReturnRate"])
        try:
            row["Bets"] = json.loads(row["BetsJSON"])
        except Exception:
            row["Bets"] = []
        rows.append(row)

    expanded = []
    for row in rows:
        race = _race_parts(row["RaceID"])
        bets = row.get("Bets") or []
        for bet_index, bet in enumerate(bets, start=1):
            combo = str(bet.get("combo", "")).strip()
            item = {
                **race,
                "timestamp": row["Timestamp"],
                "race_id": row["RaceID"],
                "source_model": str(row["Model"]),
                "ticket_type": str(bet.get("type", "")).strip(),
                "combo": _normalize_combo(combo),
                "amount": _parse_int(bet.get("amount", 0)),
                "bet_index": bet_index,
                "race_total_bet": row["TotalBet"],
                "race_total_return": row["TotalReturn"],
                "race_hits": row["Hits"],
                "race_hit_rate": row["HitRateNum"],
                "race_return_rate": row["ReturnRateNum"],
                "race_profit": row["TotalReturn"] - row["TotalBet"],
            }
            item.update(_combo_features(combo))
            expanded.append(item)

    return pd.DataFrame(expanded)


def _encode_frame(train_df: pd.DataFrame, pred_df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_df.copy()
    pred_x = pred_df.copy()
    for col in cat_cols:
        values = pd.concat([train_x[col], pred_x[col]], ignore_index=True).fillna("__missing__").astype(str)
        codes, uniques = pd.factorize(values, sort=True)
        train_x[col] = codes[: len(train_x)]
        pred_x[col] = codes[len(train_x) :]
    return train_x, pred_x


def _candidate_rows(dataset: pd.DataFrame, race_id: str, max_horse_number: int | None = None) -> pd.DataFrame:
    race = _race_parts(race_id)
    if dataset.empty:
        return pd.DataFrame()

    grouped = (
        dataset.groupby(["ticket_type", "combo"], as_index=False)
        .agg(
            samples=("race_id", "count"),
            avg_return_rate=("race_return_rate", "mean"),
            hit_races=("race_hits", lambda s: int((s > 0).sum())),
            avg_amount=("amount", "mean"),
        )
        .sort_values(["avg_return_rate", "samples"], ascending=[False, False])
    )

    rows = []
    for row in grouped.head(80).itertuples(index=False):
        nums = _combo_numbers(row.combo)
        if max_horse_number and nums and max(nums) > max_horse_number:
            continue
        item = {
            **race,
            "race_id": race_id,
            "source_model": "MLTrend",
            "ticket_type": row.ticket_type,
            "combo": row.combo,
            "amount": max(100, int(row.avg_amount // 100) * 100) if row.avg_amount else 100,
            "bet_index": 1,
            "samples": int(row.samples),
            "avg_return_rate": float(row.avg_return_rate),
            "hit_races": int(row.hit_races),
        }
        item.update(_combo_features(row.combo))
        rows.append(item)
    return pd.DataFrame(rows)


def train_backtest_trend_model(
    race_id: str,
    csv_path: str | Path = "backtest_results.csv",
    max_horse_number: int | None = None,
) -> BacktestTrendResult:
    full_dataset = load_backtest_dataset(csv_path)
    if full_dataset.empty:
        return BacktestTrendResult(full_dataset, pd.DataFrame(), "", "none")

    race = _race_parts(race_id)
    dataset = full_dataset[full_dataset["race_suffix"] == race["race_suffix"]].copy()
    if dataset.empty:
        context = (
            "【バックテストML傾向】\n"
            f"対象レースID: {race_id}\n"
            f"抽出条件: レースIDの西暦以降が一致（{race['race_suffix']}）\n"
            "該当するバックテスト履歴がないため、ML傾向は使用しません。"
        )
        return BacktestTrendResult(dataset, pd.DataFrame(), context, "none")

    candidates = _candidate_rows(dataset, race_id, max_horse_number=max_horse_number)
    if candidates.empty:
        return BacktestTrendResult(dataset, candidates, _format_context(dataset, candidates, race_id, "統計"), "statistics")

    feature_cols = [
        "year",
        "venue",
        "meet",
        "day",
        "race_no",
        "race_suffix",
        "source_model",
        "ticket_type",
        "amount",
        "bet_index",
        "combo_size",
        "min_num",
        "max_num",
        "avg_num",
        "spread",
    ]
    cat_cols = ["venue", "meet", "day", "race_no", "race_suffix", "source_model", "ticket_type"]
    num_cols = [col for col in feature_cols if col not in cat_cols]

    train_x = dataset[feature_cols].copy()
    pred_x = candidates[feature_cols].copy()
    for col in num_cols:
        train_x[col] = pd.to_numeric(train_x[col], errors="coerce").fillna(0)
        pred_x[col] = pd.to_numeric(pred_x[col], errors="coerce").fillna(0)

    y = pd.to_numeric(dataset["race_return_rate"], errors="coerce").fillna(0.0)
    model_name = "statistics"
    if len(dataset) >= 12 and y.nunique() >= 2:
        try:
            import lightgbm as lgb

            enc_train, enc_pred = _encode_frame(train_x, pred_x, cat_cols)
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=80,
                learning_rate=0.06,
                num_leaves=15,
                min_child_samples=3,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(enc_train, y)
            candidates["ml_score"] = model.predict(enc_pred)
            model_name = "LightGBM"
        except Exception:
            candidates["ml_score"] = candidates["avg_return_rate"]
    else:
        candidates["ml_score"] = candidates["avg_return_rate"]

    candidates["trend_score"] = (
        candidates["ml_score"].fillna(0) * 0.75
        + candidates["avg_return_rate"].fillna(0) * 0.20
        + candidates["samples"].fillna(0).clip(upper=5) * 2.0
    )
    candidates = candidates.sort_values(["trend_score", "samples"], ascending=[False, False]).reset_index(drop=True)
    context = _format_context(dataset, candidates, race_id, model_name)
    return BacktestTrendResult(dataset, candidates, context, model_name)


def _format_context(dataset: pd.DataFrame, candidates: pd.DataFrame, race_id: str, model_name: str) -> str:
    if dataset.empty:
        return ""

    race = _race_parts(race_id)
    source = dataset
    scope = f"レースIDの西暦以降が一致（{race['race_suffix']}）"

    ticket_summary = (
        source.groupby("ticket_type", as_index=False)
        .agg(
            samples=("race_id", "count"),
            avg_return_rate=("race_return_rate", "mean"),
            hit_races=("race_hits", lambda s: int((s > 0).sum())),
        )
        .sort_values(["avg_return_rate", "samples"], ascending=[False, False])
        .head(5)
    )

    lines = [
        "【バックテストML傾向】",
        f"学習方式: {model_name}",
        f"学習データ: {dataset['race_id'].nunique()}レース / {len(dataset)}買い目",
        f"参照範囲: {scope}（{source['race_id'].nunique()}レース / {len(source)}買い目）",
        "注意: これは過去のバックテスト出力から見た買い目傾向です。馬の実力評価ではなく、最終予想の補助材料として扱ってください。",
    ]

    if not ticket_summary.empty:
        lines.append("馬券種別の傾向:")
        for row in ticket_summary.itertuples(index=False):
            lines.append(
                f"- {row.ticket_type}: 平均回収率 {row.avg_return_rate:.1f}% / サンプル {int(row.samples)} / 的中レース {int(row.hit_races)}"
            )

    if not candidates.empty:
        lines.append("ML推奨候補:")
        for row in candidates.head(8).itertuples(index=False):
            lines.append(
                f"- {row.ticket_type} {row.combo}: score {float(row.trend_score):.1f} / 平均回収率 {float(row.avg_return_rate):.1f}% / サンプル {int(row.samples)}"
            )

    return "\n".join(lines)


def build_backtest_ml_context(
    race_id: str,
    csv_path: str | Path = "backtest_results.csv",
    max_horse_number: int | None = None,
) -> str:
    return train_backtest_trend_model(race_id, csv_path, max_horse_number=max_horse_number).context_text

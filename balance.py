"""My収支（実際に反映した予想の購入・払戻）を記録・集計するモジュール。

「結果照合」でチェックを入れた予想や、手動入力した実際の購入結果を
1レコードずつCSVに蓄積し、月別・年別・券種別に収支を集計する。
"""
import json
import os
import uuid

import pandas as pd

BALANCE_FILE = "my_balance.csv"

BALANCE_COLUMNS = [
    "entry_id", "saved_at", "race_date", "race_id", "race_name",
    "source", "ticket_type", "bet_count", "hits",
    "total_bet", "total_return", "profit", "ticket_stats", "memo",
]
_NUMERIC_COLUMNS = ["bet_count", "hits", "total_bet", "total_return", "profit"]


def load_balance_df():
    """My収支の全記録を読み込む。race_dateはdatetime、金額系は数値に変換済み。"""
    if not os.path.exists(BALANCE_FILE):
        return pd.DataFrame(columns=BALANCE_COLUMNS)
    try:
        df = pd.read_csv(BALANCE_FILE, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=BALANCE_COLUMNS)

    for col in BALANCE_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in _NUMERIC_COLUMNS else ""
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[BALANCE_COLUMNS]


def append_entries(entries):
    """entries（dictのリスト）をMy収支CSVに追記する。"""
    if not entries:
        return
    raw_df = (
        pd.read_csv(BALANCE_FILE, encoding="utf-8-sig")
        if os.path.exists(BALANCE_FILE)
        else pd.DataFrame(columns=BALANCE_COLUMNS)
    )

    new_rows = []
    for entry in entries:
        total_bet = int(entry.get("total_bet", 0) or 0)
        total_return = int(entry.get("total_return", 0) or 0)
        row = {col: entry.get(col, "") for col in BALANCE_COLUMNS}
        row["entry_id"] = uuid.uuid4().hex[:12]
        row["saved_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        row["total_bet"] = total_bet
        row["total_return"] = total_return
        row["profit"] = total_return - total_bet
        row["bet_count"] = int(entry.get("bet_count", 0) or 0)
        row["hits"] = int(entry.get("hits", 0) or 0)
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows, columns=BALANCE_COLUMNS)
    out_df = pd.concat([raw_df, new_df], ignore_index=True)
    out_df.to_csv(BALANCE_FILE, index=False, encoding="utf-8-sig")


def delete_entries(entry_ids):
    """指定したentry_idの記録をMy収支CSVから削除する。"""
    if not entry_ids or not os.path.exists(BALANCE_FILE):
        return
    df = pd.read_csv(BALANCE_FILE, encoding="utf-8-sig")
    df = df[~df["entry_id"].isin(entry_ids)]
    df.to_csv(BALANCE_FILE, index=False, encoding="utf-8-sig")


def summarize_by_period(df, freq="M"):
    """月別（freq="M"）または年別（freq="Y"）に投資・払戻・収支・回収率を集計する。"""
    empty = pd.DataFrame(columns=["period", "投資", "払戻", "収支", "回収率", "的中数", "購入点数", "件数"])
    if df is None or df.empty:
        return empty
    work = df.dropna(subset=["race_date"]).copy()
    if work.empty:
        return empty

    work["period"] = work["race_date"].dt.to_period(freq).astype(str)
    grouped = work.groupby("period").agg(
        投資=("total_bet", "sum"),
        払戻=("total_return", "sum"),
        的中数=("hits", "sum"),
        購入点数=("bet_count", "sum"),
        件数=("entry_id", "count"),
    ).reset_index()
    grouped["収支"] = grouped["払戻"] - grouped["投資"]
    grouped["回収率"] = grouped.apply(
        lambda r: round(r["払戻"] / r["投資"] * 100, 1) if r["投資"] > 0 else 0.0, axis=1
    )
    return grouped.sort_values("period")[["period", "投資", "払戻", "収支", "回収率", "的中数", "購入点数", "件数"]]


def summarize_by_ticket_type(df):
    """券種別に投資・払戻・収支・回収率を集計する（全期間合計）。"""
    empty = pd.DataFrame(columns=["券種", "投資", "払戻", "収支", "回収率", "的中数"])
    if df is None or df.empty:
        return empty

    rows = []
    for _, r in df.iterrows():
        parsed = None
        stats_raw = r.get("ticket_stats", "")
        if isinstance(stats_raw, str) and stats_raw.strip():
            try:
                parsed = json.loads(stats_raw)
            except Exception:
                parsed = None

        if parsed:
            for ticket_type, s in parsed.items():
                rows.append({
                    "券種": ticket_type,
                    "投資": s.get("bet", 0),
                    "払戻": s.get("return", 0),
                    "的中数": s.get("hits", 0),
                })
        else:
            rows.append({
                "券種": r.get("ticket_type") or "その他",
                "投資": r.get("total_bet", 0),
                "払戻": r.get("total_return", 0),
                "的中数": r.get("hits", 0),
            })

    if not rows:
        return empty
    detail_df = pd.DataFrame(rows)
    grouped = detail_df.groupby("券種").sum(numeric_only=True).reset_index()
    grouped["収支"] = grouped["払戻"] - grouped["投資"]
    grouped["回収率"] = grouped.apply(
        lambda r: round(r["払戻"] / r["投資"] * 100, 1) if r["投資"] > 0 else 0.0, axis=1
    )
    return grouped.sort_values("収支", ascending=False)

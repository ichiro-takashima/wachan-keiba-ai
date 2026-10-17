import re

import pandas as pd

from data_features import find_matching_column


# 見習い騎手の減量記号などを除去して、出馬表と結果ページの騎手名を突き合わせる
_JOCKEY_MARKS = r"[☆▲△◇★◆\s　]"


def normalize_jockey_name(name):
    return re.sub(_JOCKEY_MARKS, "", str(name or ""))


def _to_int(value):
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _form_label(rides, wins, top3, outperform_avg):
    if rides == 0:
        return "当日初騎乗"
    if wins >= 2 or (rides >= 2 and top3 / rides >= 0.5):
        return "好調"
    if rides >= 3 and top3 == 0:
        return "不調"
    if outperform_avg is not None and outperform_avg >= 2:
        return "やや好調"
    if outperform_avg is not None and outperform_avg <= -3:
        return "やや不調"
    return "普通"


def summarize_jockey_stats(day_results_df):
    """当日結果(1行=1騎乗)から騎手ごとの当日成績を集計し、正規化騎手名をキーにした辞書を返す。"""
    stats = {}
    if day_results_df is None or day_results_df.empty:
        return stats

    for jockey_key, group in day_results_df.groupby(day_results_df["騎手"].map(normalize_jockey_name)):
        if not jockey_key:
            continue
        group = group.sort_values("レース番号")
        finishes = []
        outperforms = []
        history = []
        for _, row in group.iterrows():
            rank = _to_int(row.get("着順"))
            pop = _to_int(row.get("人気"))
            rank_text = f"{rank}着" if rank is not None else str(row.get("着順", "")) or "-"
            pop_text = f"({pop}人気)" if pop is not None else ""
            history.append(f"{row['レース番号']}R:{rank_text}{pop_text}")
            if rank is None:
                continue
            finishes.append(rank)
            if pop is not None:
                outperforms.append(pop - rank)

        rides = len(finishes)
        wins = sum(1 for r in finishes if r == 1)
        seconds = sum(1 for r in finishes if r == 2)
        thirds = sum(1 for r in finishes if r == 3)
        top3 = wins + seconds + thirds
        outperform_avg = sum(outperforms) / len(outperforms) if outperforms else None
        stats[jockey_key] = {
            "当日騎乗": rides,
            "1着": wins,
            "2着": seconds,
            "3着": thirds,
            "勝率": wins / rides if rides else None,
            "複勝率": top3 / rides if rides else None,
            "人気差平均": outperform_avg,
            "調子": _form_label(rides, wins, top3, outperform_avg),
            "当日成績": " ".join(history),
        }
    return stats


def build_jockey_day_form_table(day_results_df, shutuba_df):
    """出馬表の各馬について、騎乗騎手の当日成績を1行ずつまとめた表を返す。"""
    if shutuba_df is None or shutuba_df.empty:
        return pd.DataFrame()
    jockey_col = find_matching_column(shutuba_df, ["騎手"])
    if not jockey_col:
        return pd.DataFrame()
    num_col = find_matching_column(shutuba_df, ["馬番"])
    name_col = find_matching_column(shutuba_df, ["馬名"])

    stats = summarize_jockey_stats(day_results_df)
    rows = []
    for _, row in shutuba_df.iterrows():
        jockey = str(row.get(jockey_col, "")).strip()
        s = stats.get(normalize_jockey_name(jockey))
        if s is None:
            s = {"当日騎乗": 0, "1着": 0, "2着": 0, "3着": 0, "勝率": None, "複勝率": None,
                 "人気差平均": None, "調子": "当日初騎乗", "当日成績": ""}
        rows.append({
            "馬番": row.get(num_col, "") if num_col else "",
            "馬名": row.get(name_col, "") if name_col else "",
            "騎手": jockey,
            "当日騎乗": s["当日騎乗"],
            "着別度数": f"{s['1着']}-{s['2着']}-{s['3着']}-{s['当日騎乗'] - s['1着'] - s['2着'] - s['3着']}",
            "勝率": f"{s['勝率']:.0%}" if s["勝率"] is not None else "-",
            "複勝率": f"{s['複勝率']:.0%}" if s["複勝率"] is not None else "-",
            "人気差平均": round(s["人気差平均"], 1) if s["人気差平均"] is not None else None,
            "調子": s["調子"],
            "当日成績": s["当日成績"],
        })
    return pd.DataFrame(rows)


def jockey_day_form_lookup(form_df):
    """馬番 -> 当日騎手成績の辞書（ルールベース評価用）。"""
    lookup = {}
    if form_df is None or form_df.empty:
        return lookup
    for _, row in form_df.iterrows():
        lookup[str(row.get("馬番", "")).strip()] = row.to_dict()
    return lookup


def score_jockey_day_form(form_row):
    """当日騎手成績から外部要因スコアの加減点と理由を返す。"""
    if not form_row:
        return 0.0, None
    label = form_row.get("調子")
    if label == "好調":
        return 1.0, "騎手当日好調 +1"
    if label == "やや好調":
        return 0.5, "騎手当日人気以上の好走 +0.5"
    if label == "不調":
        return -0.5, "騎手当日不調 -0.5"
    return 0.0, None


def format_jockey_day_form_for_prompt(form_df, races_checked=None):
    if form_df is None or form_df.empty:
        return ""
    if not (form_df["当日騎乗"] > 0).any():
        return ""
    header = "【騎手の当日成績（同日・同競馬場で本レースより前に行われたレース）】"
    note = (
        "着別度数は1着-2着-3着-着外。人気差平均は(人気-着順)の平均で、プラスほど人気以上に好走。"
        "騎手の当日の調子・馬場傾向への対応を評価材料に加えてください。ただし騎乗数が少ない場合は参考程度にしてください。"
    )
    if races_checked:
        note = f"対象: 1R〜{races_checked}R。" + note
    return "\n".join([header, note, form_df.fillna("").to_csv(index=False, sep="|")])

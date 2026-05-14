import re

import pandas as pd

from data_features import find_matching_column


def _to_number(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).replace(",", "").strip()
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _range_text(low, high):
    return f"{low}-{high}点" if low != high else f"{low}点"


def analyze_race_volatility(shutuba_df=None, past_trend_df=None):
    """出馬表と過去傾向から、荒れ具合と券種別の推奨点数を返す。"""
    reasons = []
    metrics = {}
    score = 35

    if shutuba_df is not None and not shutuba_df.empty:
        odds_col = find_matching_column(shutuba_df, ["オッズ", "単勝", "単勝オッズ"])
        pop_col = find_matching_column(shutuba_df, ["人気", "人気順"])
        field_size = len(shutuba_df)
        metrics["頭数"] = field_size

        if field_size >= 16:
            score += 12
            reasons.append("多頭数で紛れが起きやすい")
        elif field_size >= 14:
            score += 7
            reasons.append("頭数がやや多い")
        elif field_size <= 8:
            score -= 8
            reasons.append("少頭数で相手を絞りやすい")

        odds_values = []
        if odds_col:
            for _, row in shutuba_df.iterrows():
                odds = _to_number(row.get(odds_col))
                popularity = _to_number(row.get(pop_col)) if pop_col else None
                if odds and odds > 0:
                    odds_values.append({"odds": odds, "popularity": popularity})

        odds_values.sort(key=lambda x: x["odds"])
        if odds_values:
            favorite = odds_values[0]["odds"]
            metrics["1番人気想定オッズ"] = favorite
            if favorite <= 2.0:
                score -= 18
                reasons.append("抜けた人気馬がいる")
            elif favorite <= 3.5:
                score -= 6
                reasons.append("上位人気に一定の信頼感がある")
            elif favorite <= 6.0:
                score += 10
                reasons.append("1番人気のオッズが高め")
            else:
                score += 20
                reasons.append("明確な軸馬が見えにくい")

            if len(odds_values) >= 2:
                gap = odds_values[1]["odds"] - favorite
                metrics["上位2頭オッズ差"] = gap
                if gap < 1.0:
                    score += 12
                    reasons.append("上位人気が接近している")
                elif gap >= 3.0:
                    score -= 8
                    reasons.append("人気上位に序列がある")

            if len(odds_values) >= 5:
                top5 = [x["odds"] for x in odds_values[:5]]
                spread = max(top5) - min(top5)
                metrics["上位5頭オッズ幅"] = spread
                if spread <= 4.0:
                    score += 12
                    reasons.append("上位5頭のオッズが詰まっている")
                elif spread >= 12.0:
                    score -= 6
                    reasons.append("上位人気と伏兵の差が大きい")
        else:
            score += 8
            reasons.append("オッズ情報が不足しており不確実性が高い")
    else:
        score += 10
        reasons.append("出馬表がないため荒れ具合の判断材料が少ない")

    if past_trend_df is not None and not past_trend_df.empty:
        pop_col = find_matching_column(past_trend_df, ["人気", "人気順"])
        rank_col = find_matching_column(past_trend_df, ["着順", "着"])
        if pop_col and rank_col:
            trend = past_trend_df.copy()
            trend["_rank"] = pd.to_numeric(trend[rank_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            trend["_pop"] = pd.to_numeric(trend[pop_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            win_pop = trend.loc[trend["_rank"] == 1, "_pop"].dropna()
            top3_pop = trend.loc[trend["_rank"].between(1, 3), "_pop"].dropna()
            if not win_pop.empty:
                avg_win_pop = float(win_pop.mean())
                metrics["過去勝ち馬平均人気"] = avg_win_pop
                if avg_win_pop >= 5:
                    score += 12
                    reasons.append("過去の勝ち馬人気が低めで波乱傾向")
                elif avg_win_pop <= 2:
                    score -= 8
                    reasons.append("過去は上位人気が勝ちやすい")
            if not top3_pop.empty and float(top3_pop.mean()) >= 5:
                score += 7
                reasons.append("過去の馬券内に伏兵が混じりやすい")

    score = max(0, min(100, int(round(score))))
    if score <= 30:
        level = "堅い"
        summary = "点数を絞り、軸の信頼度を重視"
        ranges = {
            "単勝・複勝": (1, 2),
            "ワイド": (2, 3),
            "馬連": (2, 4),
            "3連複": (4, 8),
            "3連単": (0, 6),
        }
    elif score <= 60:
        level = "標準"
        summary = "軸を決めつつ相手を中程度に広げる"
        ranges = {
            "単勝・複勝": (1, 2),
            "ワイド": (3, 5),
            "馬連": (4, 6),
            "3連複": (8, 15),
            "3連単": (6, 18),
        }
    elif score <= 80:
        level = "荒れ気味"
        summary = "上位評価と穴を混ぜ、3連系は厚めに構える"
        ranges = {
            "単勝・複勝": (2, 4),
            "ワイド": (5, 8),
            "馬連": (6, 10),
            "3連複": (15, 28),
            "3連単": (18, 36),
        }
    else:
        level = "大荒れ警戒"
        summary = "見送りも含め、買うなら広く薄く"
        ranges = {
            "単勝・複勝": (0, 2),
            "ワイド": (6, 10),
            "馬連": (8, 12),
            "3連複": (25, 45),
            "3連単": (30, 60),
        }

    ticket_rows = [
        {
            "券種": ticket_type,
            "推奨点数": _range_text(*point_range),
            "最低必要予算目安": f"{point_range[0] * 100:,}-{point_range[1] * 100:,}円",
        }
        for ticket_type, point_range in ranges.items()
    ]

    return {
        "score": score,
        "level": level,
        "summary": summary,
        "reasons": reasons[:6],
        "metrics": metrics,
        "ticket_rows": ticket_rows,
    }


def format_volatility_for_prompt(volatility):
    if not volatility:
        return ""
    lines = [
        f"【荒れ具合診断】荒れ指数: {volatility['score']}/100、判定: {volatility['level']}",
        f"方針: {volatility['summary']}",
    ]
    if volatility.get("reasons"):
        lines.append("根拠: " + " / ".join(volatility["reasons"]))
    if volatility.get("ticket_rows"):
        lines.append("券種別の推奨購入点数:")
        for row in volatility["ticket_rows"]:
            lines.append(f"・{row['券種']}: {row['推奨点数']}（{row['最低必要予算目安']}）")
    lines.append("買い目を出す時は、この推奨点数の範囲を優先し、予算不足なら点数を下限側へ寄せてください。")
    return "\n".join(lines)
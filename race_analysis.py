import itertools
import json
import re

import pandas as pd
import streamlit as st

from data_features import find_matching_column
from race_volatility import format_volatility_for_prompt

def analyze_running_style(results_df):
    """通過順位のデータから脚質を判定する"""
    if results_df is None or results_df.empty:
        return "不明"

    corner_col = find_matching_column(results_df, ["通過", "コーナー通過順位", "コーナー"])
    if not corner_col:
        st.warning(f"脚質判定に必要な「通過」列が見つかりませんでした。取得された列名を確認してください: {results_df.columns.tolist()}")
        return "不明"

    corner_pos_series = results_df[corner_col].dropna()
    last_5_races = corner_pos_series.head(5)
    if last_5_races.empty:
        return "不明"

    avg_positions = []
    for pos_str in last_5_races:
        positions = [int(p) for p in re.findall(r"\d+", str(pos_str))]
        if positions:
            avg_positions.append(sum(positions) / len(positions))

    if not avg_positions:
        return "不明"

    avg_pos = sum(avg_positions) / len(avg_positions)
    if avg_pos <= 2.5:
        return "逃げ"
    if avg_pos <= 5.5:
        return "先行"
    if avg_pos <= 10.5:
        return "差し"
    return "追込"


def analyze_track_preference(results_df):
    """戦績から馬場適性を分析する"""
    if results_df is None or results_df.empty:
        return "データなし"

    track_col = find_matching_column(results_df, ["馬場", "馬場状態"])
    rank_col = find_matching_column(results_df, ["着順", "着"])
    if not track_col or not rank_col:
        return "データなし"

    df = results_df.copy()
    df["着順_num"] = df[rank_col].astype(str).str.extract(r"(\d+)")[0]
    df["着順_num"] = pd.to_numeric(df["着順_num"], errors="coerce")

    good_track = df[df[track_col] == "良"]
    good_track_in_money = good_track[good_track["着順_num"] <= 3].shape[0]
    heavy_track = df[df[track_col].isin(["稍", "重", "不"])]
    heavy_track_in_money = heavy_track[heavy_track["着順_num"] <= 3].shape[0]

    if good_track.shape[0] == 0 and heavy_track.shape[0] == 0:
        return "経験なし"
    if good_track_in_money > heavy_track_in_money:
        return f"良馬場巧者({good_track_in_money}回)"
    if heavy_track_in_money > good_track_in_money:
        return f"道悪巧者({heavy_track_in_money}回)"
    if good_track_in_money > 0:
        return "馬場不問"
    return "傾向なし"


def calculate_deviations(scores):
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std_dev = variance ** 0.5
    if std_dev == 0:
        return [50.0] * len(scores)
    return [50 + (x - mean) / std_dev * 10 for x in scores]


def calculate_weighted_mean(values):
    vals = list(values)
    if not vals:
        return 0
    weights = [max(0.1, 1.0 - 0.2 * i) for i in range(len(vals))]
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


def _to_number(value):
    if value is None:
        return None
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _numeric_series(df, candidates):
    col = find_matching_column(df, candidates)
    if not col:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce").dropna()


def _top3_rate(df):
    if df is None or df.empty or "rank_num" not in df:
        return None
    valid = df[df["rank_num"].notna()]
    if valid.empty:
        return None
    return len(valid[valid["rank_num"] <= 3]) / len(valid)


def _score_rule_factors(results, shutuba_meta, current_dist, current_place, current_track_type):
    score = 0.0
    details = []
    ability = 0.0
    suitability = 0.0
    value = 0.0
    external = 0.0

    rank_series = _numeric_series(results, ["着順", "着"]) if results is not None and not results.empty else pd.Series(dtype="float64")
    if not rank_series.empty:
        latest_rank = rank_series.iloc[0]
        last5 = rank_series.head(5)
        top3_count = int((last5 <= 3).sum())
        if latest_rank == 1:
            ability += 3
            details.append("前走1着 +3")
        elif latest_rank <= 3:
            ability += 2
            details.append("前走3着以内 +2")
        if top3_count >= 3:
            ability += 2
            details.append("近5走で3着以内3回以上 +2")
        elif top3_count >= 2:
            ability += 1
            details.append("近5走で3着以内2回 +1")

    last3f = _numeric_series(results, ["上り", "上がり", "上り3F", "上がり3F"]) if results is not None and not results.empty else pd.Series(dtype="float64")
    if not last3f.empty:
        avg_last3f = calculate_weighted_mean(last3f.head(5))
        if avg_last3f <= 34.5:
            ability += 2
            details.append("近走上がり優秀 +2")
        elif avg_last3f <= 35.5:
            ability += 1
            details.append("近走上がり良好 +1")

    rating = _numeric_series(results, ["レーティング", "rating", "指数"]) if results is not None and not results.empty else pd.Series(dtype="float64")
    if not rating.empty:
        avg_rating = calculate_weighted_mean(rating.head(5))
        if avg_rating >= 100:
            ability += 2
            details.append("レーティング100以上 +2")
        elif avg_rating >= 90:
            ability += 1
            details.append("レーティング90以上 +1")

    if results is not None and not results.empty and not rank_series.empty:
        df_work = results.copy()
        df_work["rank_num"] = rank_series.reindex(results.index)
        if current_dist:
            dist_col = find_matching_column(df_work, ["距離"])
            if dist_col:
                df_work["dist_num"] = pd.to_numeric(df_work[dist_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
                dist_df = df_work[df_work["dist_num"].notna() & (abs(df_work["dist_num"] - current_dist) <= 200)]
                rate = _top3_rate(dist_df)
                if rate is not None and rate >= 0.4:
                    suitability += 2
                    details.append("距離適性あり +2")
        if current_track_type:
            track_col = find_matching_column(df_work, ["馬場", "馬場状態"])
            if track_col:
                if current_track_type == "良":
                    track_df = df_work[df_work[track_col].astype(str).str.contains("良", na=False)]
                else:
                    track_df = df_work[df_work[track_col].astype(str).str.contains("稍|重|不", na=False)]
                rate = _top3_rate(track_df)
                if rate is not None and rate >= 0.4:
                    suitability += 1.5
                    details.append("馬場適性あり +1.5")
        if current_place:
            place_col = find_matching_column(df_work, ["開催"])
            if place_col:
                place_df = df_work[df_work[place_col].astype(str).str.contains(current_place, na=False)]
                rate = _top3_rate(place_df)
                if rate is not None and rate >= 0.35:
                    suitability += 1.5
                    details.append("コース適性あり +1.5")

    odds = _to_number(shutuba_meta.get("odds"))
    popularity = _to_number(shutuba_meta.get("popularity"))
    if odds is not None:
        if odds >= 5 and ability + suitability >= 4:
            value += 2
            details.append("能力・適性に対して単勝オッズ妙味 +2")
        elif odds >= 10 and ability + suitability >= 3:
            value += 1
            details.append("中穴妙味 +1")
    if popularity is not None and popularity >= 6 and ability + suitability >= 4:
        value += 1.5
        details.append("人気以上に評価可能 +1.5")

    current_jockey = str(shutuba_meta.get("jockey", "")).strip()
    if results is not None and not results.empty and current_jockey:
        jockey_col = find_matching_column(results, ["騎手"])
        if jockey_col:
            latest_jockey = str(results.iloc[0].get(jockey_col, "")).strip()
            if latest_jockey and latest_jockey != current_jockey:
                external -= 0.5
                details.append("騎手乗り替わり -0.5")
    frame = _to_number(shutuba_meta.get("frame"))
    if frame is not None and frame in {1, 2, 7, 8}:
        external += 0.5
        details.append("枠順補正 +0.5")
    carried_weight = _to_number(shutuba_meta.get("weight_carried"))
    if carried_weight is not None:
        if carried_weight >= 58:
            external -= 1
            details.append("斤量58kg以上 -1")
        elif carried_weight <= 53:
            external += 1
            details.append("斤量53kg以下 +1")

    score = ability + suitability + value + external
    return {
        "rule_score": score,
        "ability_score": ability,
        "suitability_score": suitability,
        "value_score": value,
        "external_score": external,
        "rule_details": details[:5],
        "odds": odds,
        "popularity": popularity,
    }


def _parse_axis_horses(axis_text):
    return [part for part in re.split(r"[\s,、・/]+", str(axis_text or "").strip()) if part]


def _unique_ordered(values):
    seen = set()
    ordered = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _candidate_custom_combos(ticket_type, ranked_umabans, axis_text, betting_method="通常"):
    ranked_umabans = _unique_ordered(ranked_umabans)
    axis_horses = _unique_ordered(_parse_axis_horses(axis_text))
    if not axis_horses and betting_method == "ながし" and ranked_umabans:
        axis_horses = [ranked_umabans[0]]

    pool = _unique_ordered(axis_horses + ranked_umabans)
    if ticket_type in ["単勝", "複勝"]:
        return axis_horses[:1] or ranked_umabans[:1] or ["1"]

    if ticket_type in ["馬連", "ワイド", "枠連"]:
        combos = []
        if axis_horses:
            for axis in axis_horses:
                for target in ranked_umabans:
                    if target != axis:
                        combos.append("-".join([axis, target]))
        else:
            combos = ["-".join(pair) for pair in itertools.combinations(pool, 2)]
        return _unique_ordered(combos)

    if ticket_type == "馬単":
        combos = []
        if axis_horses:
            for axis in axis_horses:
                for target in ranked_umabans:
                    if target != axis:
                        combos.append(f"{axis}-{target}")
        else:
            combos = ["-".join(pair) for pair in itertools.permutations(pool, 2)]
        return _unique_ordered(combos)

    if ticket_type == "3連複":
        combos = []
        if axis_horses:
            fixed = axis_horses[:2]
            targets = [u for u in ranked_umabans if u not in fixed]
            need = 3 - len(fixed)
            for extra in itertools.combinations(targets, need):
                combos.append("-".join(fixed + list(extra)))
        else:
            combos = ["-".join(combo) for combo in itertools.combinations(pool, 3)]
        return _unique_ordered(combos)

    if ticket_type == "3連単":
        combos = []
        if axis_horses:
            fixed = axis_horses[:2]
            targets = [u for u in ranked_umabans if u not in fixed]
            need = 3 - len(fixed)
            for extra in itertools.permutations(targets, need):
                combos.append("-".join(fixed + list(extra)))
        else:
            combos = ["-".join(combo) for combo in itertools.permutations(pool, 3)]
        return _unique_ordered(combos)

    return axis_horses[:1] or ranked_umabans[:1] or ["1"]


def run_rule_based_analysis(all_horse_data, shutuba_df, race_info=None, race_id=None, include_blood=True):
    horses = []
    data_weight = 1.0
    ten_weight = 1.0
    blood_weight = 1.0 if include_blood else 0.0
    current_dist = None
    current_track_type = None

    if race_info and "data" in race_info:
        race_data = race_info["data"]
        dist_match = re.search(r"(\d+)m", race_data)
        if dist_match:
            current_dist = int(dist_match.group(1))
            if current_dist <= 1400:
                ten_weight += 0.5
            elif current_dist >= 2400:
                data_weight += 0.5
        if any(w in race_data for w in ["稍重", "重", "不良", "稍"]):
            current_track_type = "道悪"
            if include_blood:
                blood_weight += 0.5
        elif "良" in race_data:
            current_track_type = "良"

    umaban_col = find_matching_column(shutuba_df, ["馬番"]) if shutuba_df is not None else None
    name_col = find_matching_column(shutuba_df, ["馬名"]) if shutuba_df is not None else None
    frame_col = find_matching_column(shutuba_df, ["枠", "枠番"]) if shutuba_df is not None else None
    jockey_col = find_matching_column(shutuba_df, ["騎手"]) if shutuba_df is not None else None
    carried_weight_col = find_matching_column(shutuba_df, ["斤量"]) if shutuba_df is not None else None
    odds_col = find_matching_column(shutuba_df, ["オッズ", "単勝"]) if shutuba_df is not None else None
    popularity_col = find_matching_column(shutuba_df, ["人気"]) if shutuba_df is not None else None

    current_place = None
    if race_id and len(str(race_id)) >= 6:
        place_code = str(race_id)[4:6]
        place_dict = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
        }
        current_place = place_dict.get(place_code)

    for horse in all_horse_data:
        hid = horse["id"]
        ped = horse.get("pedigree", {})
        results = horse.get("results", pd.DataFrame())
        name = ped.get("name", "不明")

        umaban = ""
        shutuba_meta = {}
        if shutuba_df is not None and not shutuba_df.empty and name_col and umaban_col:
            match_row = shutuba_df[shutuba_df[name_col].astype(str).str.replace(r"\s+", "", regex=True) == str(name).replace(" ", "")]
            if not match_row.empty:
                matched = match_row.iloc[0]
                umaban = str(matched[umaban_col])
                shutuba_meta = {
                    "frame": matched.get(frame_col, "") if frame_col else "",
                    "jockey": matched.get(jockey_col, "") if jockey_col else "",
                    "weight_carried": matched.get(carried_weight_col, "") if carried_weight_col else "",
                    "odds": matched.get(odds_col, "") if odds_col else "",
                    "popularity": matched.get(popularity_col, "") if popularity_col else "",
                }

        data_score = 0
        if not results.empty:
            rank_col = find_matching_column(results, ["着順", "着"])
            if rank_col:
                ranks = pd.to_numeric(results[rank_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").dropna()
                if not ranks.empty:
                    data_score += (18 - calculate_weighted_mean(ranks)) * 5

            up_col = find_matching_column(results, ["上り", "上がり", "上り3F", "上がり3F"])
            if up_col:
                up_times = pd.to_numeric(results[up_col].astype(str).str.extract(r"(\d+\.\d+)")[0], errors="coerce").dropna()
                if not up_times.empty:
                    data_score += (40 - calculate_weighted_mean(up_times)) * 2

        ten_score = 0
        corner_col = find_matching_column(results, ["通過", "コーナー通過順位", "コーナー"])
        if corner_col:
            first_corners = []
            for pos_str in results[corner_col].dropna():
                positions = [int(p) for p in re.findall(r"\d+", str(pos_str))]
                if positions:
                    first_corners.append(positions[0])
            if first_corners:
                ten_score = (18 - calculate_weighted_mean(first_corners)) * 3

        blood_score = 0
        if include_blood and not results.empty:
            rank_col = find_matching_column(results, ["着順", "着"])
            if rank_col:
                df_work = results.copy()
                df_work["rank_num"] = pd.to_numeric(df_work[rank_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

                if current_dist:
                    dist_col = find_matching_column(df_work, ["距離"])
                    if dist_col:
                        df_work["dist_num"] = pd.to_numeric(df_work[dist_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
                        dist_df = df_work[df_work["dist_num"].notna() & (abs(df_work["dist_num"] - current_dist) <= 200)]
                        if len(dist_df) > 0:
                            hits = len(dist_df[dist_df["rank_num"] <= 3])
                            blood_score += (hits / len(dist_df)) * 10

                if current_track_type:
                    track_col = find_matching_column(df_work, ["馬場", "馬場状態"])
                    if track_col:
                        if current_track_type == "良":
                            track_df = df_work[df_work[track_col].astype(str).str.contains("良", na=False)]
                        else:
                            track_df = df_work[df_work[track_col].astype(str).str.contains("稍|重|不", na=False)]
                        if len(track_df) > 0:
                            hits = len(track_df[track_df["rank_num"] <= 3])
                            blood_score += (hits / len(track_df)) * 10

                if current_place:
                    place_col = find_matching_column(df_work, ["開催"])
                    if place_col:
                        place_df = df_work[df_work[place_col].astype(str).str.contains(current_place, na=False)]
                        if len(place_df) > 0:
                            hits = len(place_df[place_df["rank_num"] <= 3])
                            blood_score += (hits / len(place_df)) * 10

        rule_factors = _score_rule_factors(results, shutuba_meta, current_dist, current_place, current_track_type)

        horses.append({
            "id": hid,
            "name": name,
            "umaban": umaban,
            "data_score": data_score,
            "ten_score": ten_score,
            "blood_score": blood_score,
            "has_data": not results.empty,
            **rule_factors,
        })

    data_devs = calculate_deviations([h["data_score"] for h in horses])
    ten_devs = calculate_deviations([h["ten_score"] for h in horses])
    blood_devs = calculate_deviations([h["blood_score"] for h in horses])
    rule_devs = calculate_deviations([h["rule_score"] for h in horses])

    rule_weight = 0.8
    total_weight = data_weight + ten_weight + blood_weight + rule_weight
    for i, h in enumerate(horses):
        h["data_dev"] = data_devs[i]
        h["ten_dev"] = ten_devs[i]
        h["blood_dev"] = blood_devs[i]
        h["rule_dev"] = rule_devs[i]
        h["total_dev"] = (
            data_devs[i] * data_weight
            + ten_devs[i] * ten_weight
            + blood_devs[i] * blood_weight
            + rule_devs[i] * rule_weight
        ) / total_weight

    strengths = [max(0.01, h["total_dev"] - 35) for h in horses]
    strength_total = sum(strengths) or 1
    for h, strength in zip(horses, strengths):
        odds = h.get("odds")
        my_probability = strength / strength_total
        market_probability = (1 / odds) if odds and odds > 0 else None
        h["my_probability"] = my_probability
        h["market_probability"] = market_probability
        h["expected_value"] = (my_probability * odds) if odds and odds > 0 else None
        h["is_value_pick"] = bool(h["expected_value"] and h["expected_value"] >= 1.05)

    def format_ranking(sorted_horses, val_key):
        lines = []
        marks = ["◎", "○", "▲", "△", "☆"]
        for i, h in enumerate(sorted_horses[:5]):
            mark = marks[i] if i < len(marks) else ""
            lines.append(f"{mark} {h['name']} (馬番:{h['umaban']}) - 偏差値: {h[val_key]:.1f}")
        return "\n".join(lines)

    analysis_result = {
        "📊 指数・データ重視": f"【過去着順・上りタイム偏差値による評価】\n{format_ranking(sorted(horses, key=lambda x: x['data_dev'], reverse=True), 'data_dev')}",
        "🏇 展開重視": f"【テンの速さ・道中位置取り偏差値による評価】\n{format_ranking(sorted(horses, key=lambda x: x['ten_dev'], reverse=True), 'ten_dev')}",
        "💰 期待値・ルール評価": f"【能力・適性・期待値・外部要因の加点評価】\n{format_ranking(sorted(horses, key=lambda x: x['rule_dev'], reverse=True), 'rule_dev')}",
        "_raw_horses": horses,
        "_weight_info": f"※適用された評価ウェイト: データ {data_weight:.1f} / 展開 {ten_weight:.1f} / 血統・適性 {blood_weight:.1f} / 期待値ルール {rule_weight:.1f}",
    }
    if include_blood:
        analysis_result = {
            "🩸 血統・適性重視": f"【独自適性スコアによる評価】\n{format_ranking(sorted(horses, key=lambda x: x['blood_dev'], reverse=True), 'blood_dev')}",
            **analysis_result,
        }
    return analysis_result


def generate_rule_based_summary(analysis_res, effective_budget, custom_requests, bet_plan_mode):
    horses = analysis_res.get("_raw_horses", [])
    weight_info = analysis_res.get("_weight_info", "")
    volatility = analysis_res.get("_volatility")
    if not horses:
        return "データがありません。"

    sorted_horses = sorted(horses, key=lambda x: x["total_dev"], reverse=True)
    summary_text = "【総合評価（偏差値平均）】\n"
    if weight_info:
        summary_text += f"{weight_info}\n\n"
    if volatility:
        summary_text += format_volatility_for_prompt(volatility) + "\n\n"
    marks = ["◎", "○", "▲", "△", "注", "穴", "大穴"]
    top_horses = sorted_horses[:7]
    for i, h in enumerate(top_horses):
        mark = marks[i] if i < len(marks) else ""
        ev_text = f" / EV:{h['expected_value']:.2f}" if h.get("expected_value") is not None else ""
        odds_text = f" / 単勝:{h['odds']:.1f}" if h.get("odds") is not None else ""
        summary_text += f"{mark} {h['name']} (馬番:{h['umaban']}) - 総合偏差値: {h['total_dev']:.1f}{odds_text}{ev_text}\n"
        if h.get("rule_details"):
            summary_text += "  ルール根拠: " + " / ".join(h["rule_details"]) + "\n"

    no_data_count = sum(1 for h in horses if not h.get("has_data", True))
    is_data_shortage = no_data_count >= len(horses) / 2
    is_top_close = len(sorted_horses) >= 2 and (sorted_horses[0]["total_dev"] - sorted_horses[1]["total_dev"]) < 1.0
    is_crowded = len(sorted_horses) >= 5 and (sorted_horses[0]["total_dev"] - sorted_horses[4]["total_dev"]) < 2.5

    if is_data_shortage or is_top_close or is_crowded:
        reasons = []
        if is_data_shortage:
            reasons.append("データ欠損（過去戦績なし等の馬）が半数以上いるため")
        if is_top_close:
            reasons.append("1位と2位の総合偏差値差が1.0未満で上位拮抗のため")
        if is_crowded:
            reasons.append("1位と5位の総合偏差値差が2.5未満でスコアが極端に密集しているため")

        summary_text += "\n🚨 **【見送り推奨】** 🚨\n以下の理由により、本レースは**購入見送り（ケン）**を推奨します。\n"
        for reason in reasons:
            summary_text += f"・{reason}\n"
        summary_text += "\n```json\n{\"bets\": []}\n```\n"
        return summary_text

    top_odds = sorted_horses[0].get("odds") if sorted_horses else None
    top_popularity = sorted_horses[0].get("popularity") if sorted_horses else None
    if top_popularity == 1 and top_odds is not None and top_odds <= 1.5:
        haran = "堅い"
        race_type = "堅実レース"
    elif len(sorted_horses) >= 5:
        diff = sorted_horses[0]["total_dev"] - sorted_horses[4]["total_dev"]
        haran = "荒れる" if diff < 5 else "堅い" if diff > 15 else "標準"
        race_type = "波乱レース" if haran == "荒れる" else "堅実レース" if haran == "堅い" else "標準レース"
    else:
        haran = "標準"
        race_type = "標準レース"

    value_picks = [h for h in top_horses if h.get("is_value_pick") and h.get("umaban")]
    summary_text += f"\n【レース分類】: {race_type}\n"
    summary_text += f"【波乱度判定】: {haran}\n"
    if value_picks:
        summary_text += "【期待値候補】: " + "、".join(
            f"{h['name']}({h['umaban']}, EV {h['expected_value']:.2f})" for h in value_picks[:3]
        ) + "\n"
    bets = []
    top_umabans = [h["umaban"] for h in top_horses if h["umaban"]]
    ranked_umabans = top_umabans[:]
    while len(ranked_umabans) < 7:
        ranked_umabans.append(str(len(ranked_umabans) + 1))

    if bet_plan_mode != "カスタム":
        top_u = ranked_umabans[:]
        top1, top2, top3, top4, top5, top6, top7 = top_u[:7]

        value_axis = value_picks[0] if value_picks else None
        if value_axis and value_axis.get("odds") and value_axis["odds"] >= 5 and value_axis.get("rule_score", 0) >= 4:
            win_amount = max(100, int(effective_budget * 0.6) // 100 * 100)
            place_amount = max(100, (effective_budget - win_amount) // 100 * 100)
            bets.append({"type": "単勝", "combo": value_axis["umaban"], "amount": win_amount})
            if effective_budget - win_amount >= 100:
                bets.append({"type": "複勝", "combo": value_axis["umaban"], "amount": place_amount})
            summary_text += "【買い方判断】: 軸候補の期待値が高いため、単勝・複勝中心で勝負します。\n"
        elif haran == "堅い":
            amt = max(100, effective_budget // 8 // 100 * 100)
            for x in [top3, top4, top5, top6, top7]:
                bets.append({"type": "3連複", "combo": f"{top1}-{top2}-{x}", "amount": amt})
            bets.append({"type": "3連単", "combo": f"{top1}-{top2}-{top3}", "amount": amt})
            bets.append({"type": "3連単", "combo": f"{top1}-{top2}-{top4}", "amount": amt})
            bets.append({"type": "3連単", "combo": f"{top1}-{top2}-{top5}", "amount": amt})
            summary_text += "【買い方判断】: 上位の信頼度が高いため、1〜2頭軸の3連系で組み立てます。\n"
        elif haran == "標準":
            amt = max(100, effective_budget // 15 // 100 * 100)
            for a, b in itertools.combinations([top2, top3, top4, top5, top6, top7], 2):
                bets.append({"type": "3連複", "combo": f"{top1}-{a}-{b}", "amount": amt})
            summary_text += "【買い方判断】: 軸は置きつつ相手を広げる三連複流しで組み立てます。\n"
        else:
            partners = [u for u in [top1, top2, top3, top4, top5, top6] if u != top1]
            amt = max(100, effective_budget // max(1, len(partners) * 2) // 100 * 100)
            for target in partners:
                bets.append({"type": "ワイド", "combo": f"{top1}-{target}", "amount": amt})
                bets.append({"type": "馬連", "combo": f"{top1}-{target}", "amount": amt})
            summary_text += "【買い方判断】: 上位拮抗の波乱想定のため、ワイド・馬連で中穴まで拾います。\n"
    else:
        for req in custom_requests:
            t_type = req.get("ticket_type", "単勝")
            budget = int(req.get("budget", 100) or 100)
            combo_count = int(req.get("combo_count", 1) or 1)
            axis = req.get("axis_horse", "").strip()

            if t_type in ["単勝", "複勝"]:
                target_count = 1
            else:
                target_count = max(1, combo_count)
                affordable_count = max(1, budget // 100)
                if target_count > affordable_count:
                    summary_text += f"【注意】: {t_type}は予算{budget}円では{combo_count}点を100円単位で買えないため、{affordable_count}点に調整します。\n"
                    target_count = affordable_count

            candidates = _candidate_custom_combos(t_type, ranked_umabans, axis, req.get("betting_method", "通常"))
            selected_combos = candidates[:target_count]
            if not selected_combos:
                continue
            if len(selected_combos) < target_count:
                summary_text += f"【注意】: {t_type}は候補が{len(selected_combos)}点分しか作れなかったため、その点数で提案します。\n"

            amount = budget if t_type in ["単勝", "複勝"] else (budget // len(selected_combos) // 100 * 100)
            amount = max(100, amount)
            if amount * len(selected_combos) > budget and t_type not in ["単勝", "複勝"]:
                amount = 100

            for combo in selected_combos:
                bets.append({"type": t_type, "combo": combo, "amount": amount})

    summary_text += "\n【提案買い目】\n"
    for bet in bets:
        summary_text += f"・{bet['type']} {bet['combo']} : {bet['amount']}円\n"

    summary_text += f"\n```json\n{json.dumps({'bets': bets}, ensure_ascii=False)}\n```"
    return summary_text


def build_race_comparison_table(all_horse_data, shutuba_df=None, max_races=8):
    """出走馬同士で共通して走ったレースを横比較できる対戦表を作る。"""
    if not all_horse_data:
        return pd.DataFrame()

    horse_meta = []
    umaban_col = find_matching_column(shutuba_df, ["馬番"]) if shutuba_df is not None else None
    name_col = find_matching_column(shutuba_df, ["馬名"]) if shutuba_df is not None else None
    waku_col = find_matching_column(shutuba_df, ["枠"]) if shutuba_df is not None else None
    shutuba_lookup = {}
    if shutuba_df is not None and not shutuba_df.empty and name_col:
        for _, row in shutuba_df.iterrows():
            shutuba_lookup[str(row.get(name_col, "")).strip()] = {
                "馬番": row.get(umaban_col, "") if umaban_col else "",
                "枠": row.get(waku_col, "") if waku_col else "",
            }

    race_map = {}
    for horse in all_horse_data:
        horse_name = str(horse.get("pedigree", {}).get("name", "")).strip()
        horse_results = horse.get("results")
        if horse_results is None or horse_results.empty:
            horse_meta.append({"馬名": horse_name, "馬番": "", "枠": ""})
            continue

        date_col = find_matching_column(horse_results, ["日付"])
        place_col = find_matching_column(horse_results, ["開催"])
        r_col = find_matching_column(horse_results, ["R"])
        race_name_col = find_matching_column(horse_results, ["レース名", "レース"])
        distance_col = find_matching_column(horse_results, ["距離"])
        rank_col = find_matching_column(horse_results, ["着順", "着"])
        time_col = find_matching_column(horse_results, ["タイム"])
        if not all([date_col, race_name_col, rank_col]):
            horse_meta.append({"馬名": horse_name, "馬番": "", "枠": ""})
            continue

        for _, row in horse_results.head(20).iterrows():
            race_name = str(row.get(race_name_col, "")).strip()
            if not race_name:
                continue
            date_raw = str(row.get(date_col, "")).strip()
            date_ts = pd.to_datetime(date_raw, errors="coerce")
            date_key = date_ts.strftime("%Y/%m/%d") if pd.notna(date_ts) else date_raw
            place = str(row.get(place_col, "")).strip() if place_col else ""
            distance = str(row.get(distance_col, "")).strip() if distance_col else ""
            round_no = str(row.get(r_col, "")).strip() if r_col else ""
            race_key = " | ".join([date_key, place, round_no, distance, race_name])

            rank_txt = str(row.get(rank_col, "")).strip()
            rank_num = pd.to_numeric(pd.Series([rank_txt]).astype(str).str.extract(r"(\d+)")[0], errors="coerce").iloc[0]
            time_txt = str(row.get(time_col, "")).strip() if time_col else ""
            parts = [f"{rank_txt}着" if rank_txt else "", time_txt]
            cell_txt = " ".join([p for p in parts if p and p != "nan"]).strip()

            label_lines = [date_key]
            if place or distance:
                label_lines.append(" ".join([p for p in [place, distance] if p]))
            label_lines.append(race_name)
            race_label = "\n".join([x for x in label_lines if x])

            if race_key not in race_map:
                race_map[race_key] = {
                    "label": race_label,
                    "date": date_ts if pd.notna(date_ts) else pd.Timestamp.min,
                    "participants": set(),
                    "cells": {},
                    "ranks": {},
                }
            race_map[race_key]["participants"].add(horse_name)
            race_map[race_key]["cells"][horse_name] = cell_txt if cell_txt else "—"
            race_map[race_key]["ranks"][horse_name] = rank_num

        horse_entry = {"馬名": horse_name, "馬番": "", "枠": ""}
        if horse_name in shutuba_lookup:
            horse_entry.update(shutuba_lookup[horse_name])
        horse_meta.append(horse_entry)

    if not race_map:
        return pd.DataFrame()

    candidate_races = [(k, v) for k, v in race_map.items() if len(v["participants"]) >= 2]
    if not candidate_races:
        return pd.DataFrame()

    candidate_races.sort(key=lambda x: (x[1]["date"], len(x[1]["participants"])), reverse=True)
    selected_races = candidate_races[:max_races]

    base_df = pd.DataFrame(horse_meta).drop_duplicates(subset=["馬名"])
    base_df["馬番_num"] = pd.to_numeric(base_df["馬番"], errors="coerce")
    base_df = base_df.sort_values(by=["馬番_num", "馬名"], na_position="last").drop(columns=["馬番_num"])

    for _, race_meta in selected_races:
        col_name = race_meta["label"]
        base_df[col_name] = base_df["馬名"].map(race_meta["cells"]).fillna("")

    return base_df


def style_race_comparison(df):
    """対戦表の着順に応じてセル色を付ける。"""
    if df.empty:
        return df

    race_cols = [c for c in df.columns if c not in ["枠", "馬番", "馬名"]]

    def color_cell(value):
        rank_match = re.search(r"(\d+)着", str(value))
        if not rank_match:
            return ""
        rank = int(rank_match.group(1))
        if rank == 1:
            return "background-color: #fff4a3;"
        if rank == 2:
            return "background-color: #d6efff;"
        if rank == 3:
            return "background-color: #ffd9d9;"
        return ""

    styler = df.style
    if race_cols:
        styler = styler.applymap(color_cell, subset=race_cols)
    return styler

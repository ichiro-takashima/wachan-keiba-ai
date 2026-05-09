import itertools
import json
import re

import pandas as pd
import streamlit as st

from data_features import find_matching_column


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


def run_rule_based_analysis(all_horse_data, shutuba_df, race_info=None, race_id=None):
    horses = []
    data_weight = 1.0
    ten_weight = 1.0
    blood_weight = 1.0
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
            blood_weight += 0.5
        elif "良" in race_data:
            current_track_type = "良"

    umaban_col = find_matching_column(shutuba_df, ["馬番"]) if shutuba_df is not None else None
    name_col = find_matching_column(shutuba_df, ["馬名"]) if shutuba_df is not None else None

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
        if shutuba_df is not None and not shutuba_df.empty and name_col and umaban_col:
            match_row = shutuba_df[shutuba_df[name_col].astype(str).str.replace(r"\s+", "", regex=True) == str(name).replace(" ", "")]
            if not match_row.empty:
                umaban = str(match_row.iloc[0][umaban_col])

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
        if not results.empty:
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

        horses.append({
            "id": hid,
            "name": name,
            "umaban": umaban,
            "data_score": data_score,
            "ten_score": ten_score,
            "blood_score": blood_score,
            "has_data": not results.empty,
        })

    data_devs = calculate_deviations([h["data_score"] for h in horses])
    ten_devs = calculate_deviations([h["ten_score"] for h in horses])
    blood_devs = calculate_deviations([h["blood_score"] for h in horses])

    total_weight = data_weight + ten_weight + blood_weight
    for i, h in enumerate(horses):
        h["data_dev"] = data_devs[i]
        h["ten_dev"] = ten_devs[i]
        h["blood_dev"] = blood_devs[i]
        h["total_dev"] = (data_devs[i] * data_weight + ten_devs[i] * ten_weight + blood_devs[i] * blood_weight) / total_weight

    def format_ranking(sorted_horses, val_key):
        lines = []
        marks = ["◎", "○", "▲", "△", "☆"]
        for i, h in enumerate(sorted_horses[:5]):
            mark = marks[i] if i < len(marks) else ""
            lines.append(f"{mark} {h['name']} (馬番:{h['umaban']}) - 偏差値: {h[val_key]:.1f}")
        return "\n".join(lines)

    return {
        "🩸 血統・適性重視": f"【独自適性スコアによる評価】\n{format_ranking(sorted(horses, key=lambda x: x['blood_dev'], reverse=True), 'blood_dev')}",
        "📊 指数・データ重視": f"【過去着順・上りタイム偏差値による評価】\n{format_ranking(sorted(horses, key=lambda x: x['data_dev'], reverse=True), 'data_dev')}",
        "🏇 展開重視": f"【テンの速さ・道中位置取り偏差値による評価】\n{format_ranking(sorted(horses, key=lambda x: x['ten_dev'], reverse=True), 'ten_dev')}",
        "_raw_horses": horses,
        "_weight_info": f"※適用された評価ウェイト: データ {data_weight:.1f} / 展開 {ten_weight:.1f} / 血統・適性 {blood_weight:.1f}",
    }


def generate_rule_based_summary(analysis_res, effective_budget, custom_requests, bet_plan_mode):
    horses = analysis_res.get("_raw_horses", [])
    weight_info = analysis_res.get("_weight_info", "")
    if not horses:
        return "データがありません。"

    sorted_horses = sorted(horses, key=lambda x: x["total_dev"], reverse=True)
    summary_text = "【総合評価（偏差値平均）】\n"
    if weight_info:
        summary_text += f"{weight_info}\n\n"
    marks = ["◎", "○", "▲", "△", "注", "穴", "大穴"]
    top_horses = sorted_horses[:7]
    for i, h in enumerate(top_horses):
        mark = marks[i] if i < len(marks) else ""
        summary_text += f"{mark} {h['name']} (馬番:{h['umaban']}) - 総合偏差値: {h['total_dev']:.1f}\n"

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

    if len(sorted_horses) >= 5:
        diff = sorted_horses[0]["total_dev"] - sorted_horses[4]["total_dev"]
        haran = "荒れる" if diff < 5 else "堅い" if diff > 15 else "標準"
    else:
        haran = "標準"

    summary_text += f"\n【波乱度判定】: {haran}\n"
    bets = []
    top_umabans = [h["umaban"] for h in top_horses if h["umaban"]]

    if bet_plan_mode != "カスタム":
        top_u = [h["umaban"] for h in top_horses if h["umaban"]]
        while len(top_u) < 7:
            top_u.append(str(len(top_u) + 1))
        top1, top2, top3, top4, top5, top6, top7 = top_u[:7]

        if haran == "堅い":
            amt = max(100, effective_budget // 8 // 100 * 100)
            for x in [top3, top4, top5, top6, top7]:
                bets.append({"type": "3連複", "combo": f"{top1}-{top2}-{x}", "amount": amt})
            bets.append({"type": "3連単", "combo": f"{top1}-{top2}-{top3}", "amount": amt})
            bets.append({"type": "3連単", "combo": f"{top1}-{top2}-{top4}", "amount": amt})
            bets.append({"type": "3連単", "combo": f"{top1}-{top2}-{top5}", "amount": amt})
        elif haran == "標準":
            amt = max(100, effective_budget // 15 // 100 * 100)
            for a, b in itertools.combinations([top2, top3, top4, top5, top6, top7], 2):
                bets.append({"type": "3連複", "combo": f"{top1}-{a}-{b}", "amount": amt})
        else:
            amt = max(100, effective_budget // 20 // 100 * 100)
            for a, b, c in itertools.combinations([top1, top2, top3, top4, top5, top6], 3):
                bets.append({"type": "3連複", "combo": f"{a}-{b}-{c}", "amount": amt})
    else:
        for req in custom_requests:
            t_type = req.get("ticket_type", "単勝")
            budget = req.get("budget", 100)
            axis = req.get("axis_horse", "").strip()

            if not axis and top_umabans:
                axis = top_umabans[0]
            elif not axis:
                axis = "1"

            if t_type in ["単勝", "複勝"]:
                bets.append({"type": t_type, "combo": axis, "amount": budget})
            elif t_type in ["馬連", "ワイド", "馬単", "枠連"]:
                target = top_umabans[1] if len(top_umabans) > 1 and top_umabans[1] != axis else (top_umabans[0] if top_umabans else "2")
                bets.append({"type": t_type, "combo": f"{axis}-{target}", "amount": budget})
            elif t_type in ["3連複", "3連単"]:
                target1 = top_umabans[1] if len(top_umabans) > 1 and top_umabans[1] != axis else "2"
                target2 = top_umabans[2] if len(top_umabans) > 2 and top_umabans[2] != axis else "3"
                bets.append({"type": t_type, "combo": f"{axis}-{target1}-{target2}", "amount": budget})
            else:
                bets.append({"type": t_type, "combo": axis, "amount": budget})

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

import streamlit as st
import pandas as pd
import time
import os
import re
import json
import csv
import scraper
import bet_plans
import balance
from datetime import datetime
from evaluator import Evaluator
from feature_importance import run_feature_importance
from ai_clients import ask_chatgpt as call_chatgpt, ask_gemini as call_gemini
from backtest_ml import train_backtest_trend_model
from data_features import (
    feature_enabled,
    find_matching_column as _find_matching_column,
    get_selected_result_columns,
    get_selected_shutuba_df,
    render_ai_data_feature_selector,
)
from race_analysis import (
    analyze_running_style,
    analyze_track_preference,
    build_race_comparison_table as _build_race_comparison_table,
    generate_rule_based_summary,
    run_rule_based_analysis,
    style_race_comparison as _style_race_comparison,
)
from race_volatility import analyze_race_volatility, format_volatility_for_prompt
from jockey_form import build_jockey_day_form_table, format_jockey_day_form_for_prompt
from tickets import (
    calculate_return,
    check_gamiri,
    extract_bets_from_text,
    parse_ids_input as _parse_ids_input,
)
from ui_helpers import load_css, memo_ui

load_css("style.css")

# --- 設定 ---
# コマンドプロンプトで setx した値を取得
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
if API_KEY_GEMINI is None:
    st.sidebar.error("⚠️ APIキーが読み込めていません。再度 setx コマンドを実行し、PCを再起動してください。")
API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")
if API_KEY_OPENAI is None:
    st.sidebar.error("⚠️ APIキーが読み込めていません。再度 setx コマンドを実行し、PCを再起動してください。")

ai_choice = st.radio("使用するAIを選択", ["Gemini", "ChatGPT", "両方で比較", "ルールベース (AI不使用)"], horizontal=True)
# --- 分析用ヘルパー関数 ---
def ask_gemini(prompt):
    return call_gemini(prompt, API_KEY_GEMINI)

def ask_chatgpt(prompt):
    return call_chatgpt(prompt, API_KEY_OPENAI)


AI_PERSONA_PROMPT = """# 目的
あなたは、徹底したデータ主義と歴史的検証に基づいて競馬を分析する、現代的なデータ分析型の競馬予想AIです。感情論を一切排除し、長期的な視点でリスクをコントロールしながら、根拠のある買い目を導き出すプロフェッショナルとして振る舞ってください。

# ペルソナ設定
- 思考スタンス: 徹底したデータ主義、歴史的検証の重視。感情論や世間の人気（オッズ）に流されず、統計と傾向を元に冷静に判断する。安易な人気馬の過信を戒める「逆張り」の視点を持つ。
- 予想スタイル: 「条件付き」の柔軟な予想。自身の予想を絶対視せず、雨天や馬場状態の変化（例: 良馬場想定 / 重馬場想定）といった外部要因を織り込んだ複数のシナリオを提示する。
- 分析アプローチ: 負け戦や過去の失敗から学ぶ実戦的知見を重視する。レース結果や評価を、騎手の仕掛け、コース適性、ローテーション、馬の癖など複数の変数に分解して言語化する。
- ユーザーとの関係性: 共に勝ち筋を探る共有型のコンサルタント。一発逆転を狙うギャンブラーではなく、長期的な視点で資産（馬券）を守りながら勝負する。

# 出力・対話のガイドライン
1. 結論には必ず根拠となるデータ、傾向、過去結果、統計的示唆を添える。
2. 「良馬場想定」「重馬場想定」「スローペースなら」など、前提条件と条件分岐を明記する。
3. 馬の評価は、コース適性、ローテーション、騎手の仕掛け、馬の癖、展開適性などの変数でロジカルに説明する。
4. 冷静、客観的、協力的に話す。過度な煽りや「絶対に来る」「大鉄板」などの断定表現は禁止する。
"""

OUTPUT_FORMAT_INSTRUCTIONS = [
    "トークンの無駄遣いを避けるため、前置きや長い分析文は書かず、以下のフォーマットに厳密に従って出力してください。",
    "券種・買い目プランが複数ある場合は、このブロックをプランごとに繰り返してください。",
    "",
    "【本日の勝負レース：（券種・方式）】",
    "",
    "■ 印・買い目構成",
    "◎ （馬番）. （馬名）",
    "○ （馬番）. （馬名）",
    "▲ （馬番）. （馬名）",
    "△ （馬番）. （馬名）　※△は複数可",
    "- - - - - - - - - - - - - - - - - - - - ",
    "（三連単・馬単など着順を指定する券種の場合のみ、以下のように着順別の馬番を列挙）",
    "【1着】 （馬番）",
    "【2着】 （馬番, 馬番, ...）",
    "【3着】 （馬番, 馬番, ...）",
    "（三連複・馬連・ワイドなど着順を問わない券種の場合は、【1着】〜【3着】の代わりに組み合わせを列挙）",
    "",
    "■ 投資・購入情報",
    "* 式別：（券種・方式）",
    "* 買い目点数：（点数）点",
    "* 予算：（合計金額）円（1点（単価）円）",
]


def collect_single_race_data(target_race_id, cutoff_date):
    """単一レース予想に必要なデータをまとめて取得する。"""
    h_ids = scraper.get_horse_ids_from_race(target_race_id)
    if not h_ids:
        return None

    temp_horse_data = []
    temp_pedigree_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, hid in enumerate(h_ids):
        status_text.write(f"🔍 {i+1}/{len(h_ids)}頭目 (ID: {hid}) を取得中...")

        ped_data = scraper.scrape_horse_ped(hid)
        temp_pedigree_list.append(ped_data)

        results_df = scraper.scrape_race_results_dedicated(hid)

        if not results_df.empty:
            results_df['日付'] = pd.to_datetime(results_df['日付'], errors='coerce')
            results_df = results_df[results_df['日付'] < pd.Timestamp(cutoff_date)]
            results_df['日付'] = results_df['日付'].dt.strftime('%Y/%m/%d')

        temp_horse_data.append({"id": hid, "pedigree": ped_data, "results": results_df})
        progress_bar.progress((i + 1) / len(h_ids))

    status_text.write("📊 出馬表（レース情報）を取得中...")
    shutuba_df = scraper.scrape_shutuba_table(target_race_id)
    race_info = scraper.scrape_race_info(target_race_id)
    past_trend_df = scraper.scrape_past_top3_trend(target_race_id, years=5)

    status_text.write("🏇 騎手の当日成績を取得中...")
    jockey_day_results_df = scraper.scrape_jockey_day_results(
        target_race_id,
        progress_callback=lambda no, total, rid: status_text.write(f"🏇 騎手の当日成績を取得中... {no}/{total}R ({rid})"),
    )
    jockey_day_form_df = build_jockey_day_form_table(jockey_day_results_df, shutuba_df)

    status_text.empty()
    progress_bar.empty()

    return {
        "all_horse_data": temp_horse_data,
        "pedigree_list": temp_pedigree_list,
        "shutuba_table": shutuba_df,
        "race_info": race_info,
        "past_trend_df": past_trend_df,
        "race_volatility": analyze_race_volatility(shutuba_df, past_trend_df),
        "jockey_day_form": jockey_day_form_df,
        "jockey_day_races": jockey_day_results_df["レース番号"].nunique() if not jockey_day_results_df.empty else 0,
    }


def save_single_race_data(data):
    st.session_state.all_horse_data = data["all_horse_data"]
    st.session_state.pedigree_list = data["pedigree_list"]
    st.session_state.shutuba_table = data["shutuba_table"]
    st.session_state.race_info = data["race_info"]
    st.session_state.past_trend_df = data["past_trend_df"]
    st.session_state.race_volatility = data["race_volatility"]
    st.session_state.jockey_day_form = data["jockey_day_form"]
    st.session_state.jockey_day_races = data["jockey_day_races"]
    st.session_state.chat_messages = []
    st.session_state.analysis_results = {}
    st.session_state.full_ai_reports = {}


def _df_to_prompt_table(df, max_rows=10, max_cols=12):
    if df is None or df.empty:
        return "データなし"
    compact_df = df.head(max_rows).copy()
    if len(compact_df.columns) > max_cols:
        compact_df = compact_df.iloc[:, :max_cols]
    return compact_df.fillna("").to_csv(index=False, sep="|")


def _horse_number_lookup(shutuba_df):
    lookup = {}
    if shutuba_df is None or shutuba_df.empty:
        return lookup
    name_col = _find_matching_column(shutuba_df, ["馬名"])
    num_col = _find_matching_column(shutuba_df, ["馬番"])
    if not name_col or not num_col:
        return lookup
    for _, row in shutuba_df.iterrows():
        name = str(row.get(name_col, "")).replace(" ", "").strip()
        if name:
            lookup[name] = row.get(num_col, "")
    return lookup


def render_bet_plan_selector():
    """netkeibaの馬券購入画面を参考にした、券種・方式・目標点数・配分を指定するUI。

    馬番は指定せず、AIがレース分析に基づいて実際の馬番を決定する。
    """
    st.caption(
        "netkeibaの馬券購入画面と同じ考え方（通常・ボックス・流し・フォーメーション）で、"
        "券種・方式・目標点数（例: 三連単流し10点）・配分金額を指定します。"
        "実際にどの馬番を対象にするかはAIがレース分析に基づいて提案します。"
    )

    ticket_types = list(bet_plans.TICKET_ARITY.keys())
    bp_ticket_type = st.selectbox("券種", ticket_types, key="bp_ticket_type")
    arity = bet_plans.TICKET_ARITY[bp_ticket_type]
    ordered = bp_ticket_type in bet_plans.ORDERED_TICKETS
    methods = bet_plans.METHODS_BY_TICKET[bp_ticket_type]
    bp_method = st.radio("方式", methods, horizontal=True, key="bp_method")

    block = {"ticket_type": bp_ticket_type, "method": bp_method}

    if bp_method == "流し":
        if arity == 3:
            axis_count_label = st.radio("軸の頭数", ["1頭流し", "2頭流し"], horizontal=True, key="bp_nagashi_axis_count3")
            axis_count = 1 if axis_count_label == "1頭流し" else 2
            block["axis_count"] = axis_count
            if ordered and axis_count == 1:
                block["axis_position"] = st.radio("軸の着順", ["1着", "2着", "3着"], horizontal=True, key="bp_nagashi_pos3")
        elif ordered:
            block["axis_position"] = st.radio("軸の着順", ["1着", "2着"], horizontal=True, key="bp_nagashi_pos2")

    points_label = "何頭に賭けるか（点数）" if arity == 1 else "目標点数"
    bp_points = st.number_input(points_label, min_value=1, max_value=200, value=5, step=1, key="bp_points")
    block["points"] = int(bp_points)

    st.markdown("**配分金額**")
    amt_col1, amt_col2 = st.columns(2)
    with amt_col1:
        amount_mode_label = st.radio(
            "金額の指定方法", ["1点あたり金額", "この買い目全体の予算（均等割り）"], key="bp_amount_mode"
        )
    amount_mode = "per_point" if amount_mode_label == "1点あたり金額" else "total"
    block["amount_mode"] = amount_mode
    with amt_col2:
        block["amount_value"] = st.number_input(
            "金額（円）",
            min_value=100,
            step=100,
            value=100 if amount_mode == "per_point" else 1000,
            key="bp_amount_value",
        )

    st.caption(f"この設定の見積もり: {bet_plans.describe_block(block)}")

    if st.button("この買い方をプランに追加", key="bp_add_block", use_container_width=True):
        st.session_state.bet_plan_blocks.append(block)
        st.success("買い方プランに追加しました。")
        st.rerun()

    if st.session_state.bet_plan_blocks:
        st.markdown("#### 現在の買い方プラン")
        grand_total = 0
        remove_idx = None
        for idx, b in enumerate(st.session_state.bet_plan_blocks):
            block_total = bet_plans.block_total_amount(b)
            grand_total += block_total
            bc1, bc2 = st.columns([5, 1])
            with bc1:
                st.write(bet_plans.describe_block(b))
            with bc2:
                if st.button("削除", key=f"bp_remove_{idx}"):
                    remove_idx = idx
        if remove_idx is not None:
            st.session_state.bet_plan_blocks.pop(remove_idx)
            st.rerun()
        st.info(f"買い方プラン合計（見積もり）: {grand_total}円")
    else:
        st.caption("まだ買い方プランが追加されていません。追加すると、AIはこのプランの点数・配分を目安に実際の馬番を提案します。")


def build_full_ai_betting_prompt(race_id, max_budget, plan_blocks=None, use_full_budget=False):
    race_info = st.session_state.race_info or {"name": "不明", "data": "不明"}
    shutuba_df = st.session_state.shutuba_table
    horse_number_by_name = _horse_number_lookup(shutuba_df)
    plan_blocks = plan_blocks or []
    plan_total = sum(bet_plans.block_total_amount(b) for b in plan_blocks)

    if plan_blocks:
        budget_policy = (
            f"ユーザーは下記の【ユーザーが指定した買い方プラン】（見積合計{plan_total}円）の券種・方式・目標点数・配分金額をすでに決めています。"
            "この券種・方式・点数・金額の枠組みは変更せず、その枠組みに当てはめる実際の馬番をあなたの分析に基づいて決定してください。"
        )
    else:
        budget_policy = (
            f"最大予算{max_budget}円を必ず全額使い切ってください。集計用JSONのbetsのamount合計も必ず{max_budget}円にしてください。"
            if use_full_budget
            else "最大予算は上限です。期待値が低い場合は0円または一部だけ賭ける判断も許可します。"
        )

    sections = [
        AI_PERSONA_PROMPT,
        "以下のスクレイピング済みデータを複合的に評価し、レース分析を行ってください（分析はあなたの内部で行い、出力には含めないでください）。",
        budget_policy,
        (
            "買い方、券種、点数、配分は完全にあなたの判断に委ねます。"
            if not plan_blocks
            else "券種・方式（通常/ボックス/流し/フォーメーション）・目標点数・配分金額はユーザーが指定済みのため変更しないでください。"
            "そのプランごとに、対象とする馬番をあなたの分析に基づいて選び、具体的な買い目（組み合わせ）として提案してください。"
            "点数はプランの目標点数にできるだけ合わせ、やむを得ず変える場合も理由を明記してください。"
        ),
        "金額は必ず100円単位にしてください（馬券は10円単位の調整ができないため）。50円、150円、250円など100の倍数ではない金額は絶対に出さないでください。",
        "馬は必ず馬番で指定してください。",
        "",
        "【出力要件】",
        *OUTPUT_FORMAT_INSTRUCTIONS,
        (
            "券種・組み合わせ・金額は完全にあなたの判断で決めてください。"
            if not plan_blocks
            else "各ブロックの式別・点数・予算は、指定された買い方プランの内容に合わせてください。"
        ),
        "最後に必ず集計用JSONをこの形式で出力: ```json\n{\"bets\":[{\"type\":\"ワイド\",\"combo\":\"1-2\",\"amount\":100}]}\n```",
        "",
        f"【対象レース】ID: {race_id}",
        f"レース名: {race_info.get('name', '不明')}",
        f"条件: {race_info.get('data', '不明')}",
        f"最大予算: {max_budget}円",
        f"予算方針: {'最大予算を全額使用' if use_full_budget else '最大予算内で任意使用'}",
    ]
    if plan_blocks:
        plan_lines = ["【ユーザーが指定した買い方プラン】"]
        for b in plan_blocks:
            plan_lines.append(f"- {bet_plans.describe_block(b)}")
        plan_lines.append(f"見積合計: {plan_total}円")
        sections.extend([""] + plan_lines)

    volatility_prompt = format_volatility_for_prompt(st.session_state.race_volatility)
    if volatility_prompt:
        sections.extend(["", volatility_prompt])

    sections.extend(["", "【出馬表】", _df_to_prompt_table(shutuba_df, max_rows=24, max_cols=14)])

    if st.session_state.past_trend_df is not None and not st.session_state.past_trend_df.empty:
        sections.extend(["", "【同レース過去5年の1〜3着傾向】", _df_to_prompt_table(st.session_state.past_trend_df, max_rows=15, max_cols=10)])

    jockey_prompt = format_jockey_day_form_for_prompt(st.session_state.jockey_day_form, st.session_state.jockey_day_races)
    if jockey_prompt:
        sections.extend(["", jockey_prompt])

    sections.append("")
    sections.append("【各馬データ: 血統 + 過去10戦】")
    for horse in st.session_state.all_horse_data or []:
        ped = horse.get("pedigree", {})
        name = ped.get("name", "不明")
        number = horse_number_by_name.get(str(name).replace(" ", "").strip(), "")
        results_df = horse.get("results", pd.DataFrame())
        sections.extend([
            "",
            f"■ {number}番 {name} (horse_id: {horse.get('id', '')})",
            f"血統: 父={ped.get('sire', '不明')} / 母={ped.get('dam', '不明')} / 母父={ped.get('broodmare_sire', '不明')}",
            f"簡易分析: 脚質={analyze_running_style(results_df)} / 馬場適性={analyze_track_preference(results_df)}",
            "過去10戦:",
            _df_to_prompt_table(results_df, max_rows=10, max_cols=14),
        ])

    return "\n".join(sections)


def run_full_ai_betting_analysis(race_id, max_budget, plan_blocks, use_full_budget=False):
    prompt = build_full_ai_betting_prompt(race_id, max_budget, plan_blocks, use_full_budget)
    targets = []
    if ai_choice in ["Gemini", "両方で比較"]:
        targets.append(("Gemini", ask_gemini))
    if ai_choice in ["ChatGPT", "両方で比較"]:
        targets.append(("ChatGPT", ask_chatgpt))

    reports = {}
    for ai_name, ask_func in targets:
        with st.spinner(f"{ai_name} が血統・過去10戦・出馬表から分析中..."):
            response = ask_func(prompt)
        reports[ai_name] = response

        log_file = "prediction_log.csv"
        log_exists = os.path.exists(log_file)
        with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(["Timestamp", "RaceID", "Model", "Budget", "Prompt", "Response"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), race_id, ai_name, max_budget, prompt, response])

    return reports


# --- Streamlit UI ---
st.sidebar.title("🏇 メニュー")
app_mode = st.sidebar.radio("モード選択", ["単一レース予想", "バックテスト", "予想履歴", "My収支"], index=0)

with st.sidebar.expander("🗂️ HTML保存 / 一括テーブル生成", expanded=False):
    st.caption("保存済みHTMLから一括でテーブルを作るための管理パネルです。")
    race_html_dir = st.text_input("レースHTML保存先", value="data/html/race")
    race_ids_text = st.text_area(
        "保存したいレースID（改行 or カンマ区切り）",
        value="",
        height=80,
        help="例: 202601010101, 202601010102"
    )
    skip_existing_race = st.checkbox("既存レースHTMLはスキップ", value=True)
    force_race_refresh = st.checkbox("レースHTMLを強制再取得", value=False)
    if st.button("HTML保存（レース）", use_container_width=True):
        race_ids = _parse_ids_input(race_ids_text)
        if not race_ids:
            st.warning("レースIDを入力してください。")
        else:
            ok_count = 0
            fail_ids = []
            prog = st.progress(0)
            status = st.empty()
            for idx, rid in enumerate(race_ids, start=1):
                status.write(f"保存中: {rid} ({idx}/{len(race_ids)})")
                try:
                    scraper.scrape_html_race(
                        rid,
                        save_dir=race_html_dir,
                        skip_existing=skip_existing_race,
                        force=force_race_refresh
                    )
                    ok_count += 1
                except Exception:
                    fail_ids.append(rid)
                prog.progress(idx / len(race_ids))
            status.empty()
            if fail_ids:
                st.warning(f"完了: 成功 {ok_count} / 失敗 {len(fail_ids)}（{', '.join(fail_ids[:5])}）")
            else:
                st.success(f"完了: {ok_count}件のレースHTMLを保存しました。")

    race_table_html_dir = st.text_input("レースHTML読込元", value="data/html/race")
    race_table_output = st.text_input("レース結果テーブル出力先", value="data/processed/race_results.tsv")
    if st.button("一括テーブル生成（レース結果）", use_container_width=True):
        try:
            results_df = scraper.create_results(
                html_dir=race_table_html_dir,
                output_path=race_table_output,
                sep="\t",
                show_progress=False
            )
            st.success(f"生成完了: {len(results_df)}行 / {len(results_df.columns)}列")
            st.dataframe(results_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"レース結果テーブル生成に失敗しました: {e}")

    horse_html_dir = st.text_input("馬HTML保存先", value="data/html/horse")
    horse_ids_text = st.text_area(
        "保存したい馬ID（改行 or カンマ区切り）",
        value="",
        height=80,
        help="例: 2010101010, 2010101011"
    )
    skip_existing_horse = st.checkbox("既存馬HTMLはスキップ", value=True)
    force_horse_refresh = st.checkbox("馬HTMLを強制再取得", value=False)
    if st.button("HTML保存（馬の過去成績）", use_container_width=True):
        horse_ids = _parse_ids_input(horse_ids_text)
        if not horse_ids:
            st.warning("馬IDを入力してください。")
        else:
            ok_count = 0
            fail_ids = []
            prog = st.progress(0)
            status = st.empty()
            for idx, hid in enumerate(horse_ids, start=1):
                status.write(f"保存中: {hid} ({idx}/{len(horse_ids)})")
                try:
                    scraper.scrape_html_horse(
                        [hid],
                        save_dir=horse_html_dir,
                        skip_existing=skip_existing_horse,
                        force=force_horse_refresh,
                        show_progress=False
                    )
                    ok_count += 1
                except Exception:
                    fail_ids.append(hid)
                prog.progress(idx / len(horse_ids))
            status.empty()
            if fail_ids:
                st.warning(f"完了: 成功 {ok_count} / 失敗 {len(fail_ids)}（{', '.join(fail_ids[:5])}）")
            else:
                st.success(f"完了: {ok_count}件の馬HTMLを保存しました。")

    horse_table_html_dir = st.text_input("馬HTML読込元", value="data/html/horse")
    horse_table_output = st.text_input("馬の過去成績テーブル出力先", value="data/processed/horse_results.tsv")
    if st.button("一括テーブル生成（馬の過去成績）", use_container_width=True):
        try:
            horse_results_df = scraper.create_horse_results(
                horse_html_dir=horse_table_html_dir,
                output_path=horse_table_output,
                sep="\t",
                show_progress=False
            )
            st.success(f"生成完了: {len(horse_results_df)}行 / {len(horse_results_df.columns)}列")
            st.dataframe(horse_results_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"馬の過去成績テーブル生成に失敗しました: {e}")

if app_mode == "バックテスト":
    @st.cache_data(ttl=3600, show_spinner=False)
    def scrape_payouts(race_id): return scraper.scrape_payouts(race_id)

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_race_date(race_id): return scraper.get_race_date(race_id)

    st.title("🔄 予想バックテスト機能")
    st.markdown("過去のレースIDを入力し、AI・ルールベースの予想精度（的中率・回収率）を検証します。結果はCSVに保存されます。")
    bt_models = st.multiselect("検証するモデル", ["Gemini", "ChatGPT", "ルールベース (AI不使用)"], default=["Gemini"])
    
    if "bt_race_ids_input" not in st.session_state:
        st.session_state.bt_race_ids_input = "202405020811\n202305020811"

    with st.expander("💡 過去の同レースIDを一括生成する", expanded=False):
        st.markdown("基準となるレースID（最新のIDなど）から、年数だけを遡ったレースIDを自動生成します。\n※年によって開催回や日数がずれるレースは正しくデータが取得できない場合があります（存在しないIDは自動スキップされます）。")
        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        with col_gen1:
            base_race_id = st.text_input("基準レースID (12桁)", value="202405020811")
        with col_gen2:
            years_to_go_back = st.number_input("遡る年数（基準年含む）", min_value=1, max_value=30, value=10)
        with col_gen3:
            st.write("") # 位置合わせ用
            if st.button("IDを生成", use_container_width=True):
                if len(base_race_id) == 12 and base_race_id.isdigit():
                    base_year = int(base_race_id[:4])
                    rest_id = base_race_id[4:]
                    generated_ids = [f"{base_year - i}{rest_id}" for i in range(years_to_go_back)]
                    st.session_state.bt_race_ids_input = "\n".join(generated_ids)
                    st.rerun()
                else:
                    st.error("12桁の数字を入力してください。")

    with st.expander("🏟️ その日の同会場の全レースIDを一括生成する", expanded=False):
        st.markdown("基準となるレースID（例: 202610020601）から、同じ開催日・同じ会場の全レースIDを自動生成します（1Rから指定Rまで）。\n※実際の開催レース数より多く指定した場合、存在しないIDは自動スキップされます。")
        col_gen4, col_gen5, col_gen6 = st.columns([2, 1, 1])
        with col_gen4:
            base_race_id_day = st.text_input("基準レースID (12桁)", value="202610020601", key="base_race_id_day")
        with col_gen5:
            max_race_no = st.number_input("最大レース数", min_value=1, max_value=12, value=12, key="max_race_no_day")
        with col_gen6:
            st.write("") # 位置合わせ用
            if st.button("IDを生成", use_container_width=True, key="gen_day_race_ids_button"):
                if len(base_race_id_day) == 12 and base_race_id_day.isdigit():
                    venue_day_prefix = base_race_id_day[:10]
                    generated_ids = [f"{venue_day_prefix}{race_no:02d}" for race_no in range(1, int(max_race_no) + 1)]
                    st.session_state.bt_race_ids_input = "\n".join(generated_ids)
                    st.rerun()
                else:
                    st.error("12桁の数字を入力してください。")

    bt_race_ids_str = st.text_area("レースIDリスト（改行区切り）", key="bt_race_ids_input")
    bt_max_horses = st.number_input("最大送信頭数（トークン節約用）", min_value=5, max_value=18, value=12, help="馬番の大きい外枠の馬から除外されます。全頭送るとトークンを消費します。")

    bt_selected_features = render_ai_data_feature_selector(
        "bt",
        "バックテストでも同様にAIへ渡すデータを細かく制御します。"
    )
    bt_use_shutuba_table = feature_enabled(bt_selected_features, "shutuba_umaban", "shutuba_waku", "shutuba_name", "shutuba_sex_age", "shutuba_jockey", "shutuba_weight_carried", "shutuba_odds", "shutuba_popularity", "shutuba_stable", "shutuba_body_weight")
    bt_use_blood = feature_enabled(bt_selected_features, "blood_sire", "blood_dam", "blood_broodmare_sire")
    bt_use_analysis = feature_enabled(bt_selected_features, "analysis_running_style", "analysis_track_preference", "analysis_latest_weight")
    bt_use_past_results = any(key.startswith("result_") for key in bt_selected_features)

    col_bt_op1, col_bt_op2 = st.columns([4, 1])
    with col_bt_op1:
        st.markdown("**✍️ あなたの予想・注目馬（AIへの指示や意見があれば入力してください）**")

    bt_user_opinion = st.text_area(
        "あなたの予想・注目馬",
        placeholder="例: 1番の馬の逃げ残りに期待。雨が降っているので外枠有利。",
        key="bt_user_opinion",
        label_visibility="collapsed"
    )

    st.markdown("#### 予算設定 (バックテスト用)")
    bt_effective_budget = st.number_input(
        "1レースあたりの最大予算 (円)",
        min_value=0,
        step=100,
        value=1000,
        key="bt_overall_budget",
    )
    st.caption("買い方、券種、点数、配分はAIに完全に委ねます。")

    bt_use_backtest_ml = st.checkbox(
        "バックテスト結果からML傾向を学習して予想に反映する",
        value=True,
        key="bt_use_backtest_ml",
        help="backtest_results.csv の買い目履歴を使い、同一レースID系統で回収率が高かった馬券種・馬番構成をプロンプトへ追加します。"
    )
    bt_use_jockey_day_form = st.checkbox(
        "騎手の当日成績を予想に反映する",
        value=False,
        key="bt_use_jockey_day_form",
        help="同日・同競馬場で対象レースより前のレース結果を取得し、騎手の当日成績をプロンプトとルールベース評価に追加します（1レースあたり最大11ページ取得するため時間がかかります）。"
    )

    with st.expander("バックテストML傾向プレビュー", expanded=False):
        preview_race_id = st.text_input("傾向を見るレースID", value=base_race_id if "base_race_id" in locals() else "202405020811", key="bt_ml_preview_race_id")
        if st.button("ML傾向を表示", key="show_backtest_ml_preview"):
            try:
                ml_result = train_backtest_trend_model(preview_race_id, max_horse_number=bt_max_horses)
                if not ml_result.context_text:
                    st.info("backtest_results.csv に学習できる履歴がまだありません。")
                else:
                    st.text(ml_result.context_text)
                    if not ml_result.candidates.empty:
                        st.dataframe(
                            ml_result.candidates[["ticket_type", "combo", "trend_score", "avg_return_rate", "samples", "hit_races"]].head(20),
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"バックテストML傾向の作成に失敗しました: {e}")

    with st.expander("📊 EvaluatorクラスでCSV検証（AI vs 人気順）", expanded=False):
        st.caption("prediction CSV と payout CSV をアップロードすると、evaluator.py のロジックで客観的指標を算出します。")
        ev_col1, ev_col2, ev_col3 = st.columns(3)
        with ev_col1:
            ev_top_n = st.number_input("上位N頭 (予算指定時は無視)", min_value=2, max_value=18, value=3, step=1, key="ev_top_n")
        with ev_col2:
            ev_stake = st.number_input("1点購入額(円)", min_value=100, max_value=10000, value=100, step=100, key="ev_stake")
        with ev_col3:
            ev_budget = st.number_input("1レース予算(円・0で無効)", min_value=0, max_value=100000, value=0, step=100, key="ev_budget")

        ev_mode = st.radio("シミュレーション戦略モード", [
            "デフォルト (馬連・ワイド・三連複)", 
            "的中率重視 (ワイド中心)", 
            "回収率重視 (三連複中心)", 
            "両モード比較 (的中率重視 vs 回収率重視)"
        ], horizontal=True)

        pred_file = st.file_uploader("prediction CSV", type=["csv"], key="ev_pred_file")
        payout_file = st.file_uploader("payout CSV", type=["csv"], key="ev_payout_file")
        st.markdown("**期待する列名の例**")
        st.code("prediction: race_id, horse_number, pred_score, popularity\npayout: race_id, 券種, 組番, 払戻金", language="text")

        if st.button("Evaluatorで比較実行", key="run_evaluator_compare"):
            if pred_file is None or payout_file is None:
                st.warning("prediction CSV と payout CSV の両方をアップロードしてください。")
            else:
                try:
                    pred_df = pd.read_csv(pred_file)
                    payout_df = pd.read_csv(payout_file)

                    race_col = _find_matching_column(pred_df, ["race_id", "レースID", "raceId"])
                    horse_col = _find_matching_column(pred_df, ["horse_number", "馬番"])
                    score_col = _find_matching_column(pred_df, ["pred_score", "score", "予測スコア"])
                    pop_col = _find_matching_column(pred_df, ["popularity", "人気", "人気順"])

                    payout_race_col = _find_matching_column(payout_df, ["race_id", "レースID", "raceId"])
                    payout_type_col = _find_matching_column(payout_df, ["券種", "type"])
                    payout_combo_col = _find_matching_column(payout_df, ["組番", "combo"])
                    payout_amt_col = _find_matching_column(payout_df, ["払戻金", "pay", "payout"])

                    missing = []
                    if race_col is None: missing.append("prediction.race_id")
                    if horse_col is None: missing.append("prediction.horse_number")
                    if score_col is None: missing.append("prediction.pred_score")
                    if pop_col is None: missing.append("prediction.popularity")
                    if payout_race_col is None: missing.append("payout.race_id")
                    if payout_type_col is None: missing.append("payout.券種")
                    if payout_combo_col is None: missing.append("payout.組番")
                    if payout_amt_col is None: missing.append("payout.払戻金")
                    if missing:
                        st.error("必要列が不足しています: " + ", ".join(missing))
                        st.stop()

                    if ev_mode == "デフォルト (馬連・ワイド・三連複)":
                        bet_types_list = [("馬連", "ワイド", "三連複")]
                        strategy_prefixes = [""]
                    elif ev_mode == "的中率重視 (ワイド中心)":
                        bet_types_list = [("ワイド",)]
                        strategy_prefixes = ["的中重視_"]
                    elif ev_mode == "回収率重視 (三連複中心)":
                        bet_types_list = [("三連複",)]
                        strategy_prefixes = ["回収重視_"]
                    else:
                        bet_types_list = [("ワイド",), ("三連複",)]
                        strategy_prefixes = ["的中重視_", "回収重視_"]

                    summary_rows = []
                    detail_dfs = []
                    
                    for bet_types, prefix in zip(bet_types_list, strategy_prefixes):
                        ev = Evaluator(
                            top_n=int(ev_top_n) if ev_budget == 0 else None,
                            stake_per_ticket=int(ev_stake),
                            budget=int(ev_budget) if ev_budget > 0 else None,
                            bet_types=bet_types
                        )
                        sum_df, det_df = ev.compare_model_vs_popularity(
                            prediction_df=pred_df,
                            payout_df=payout_df,
                            score_col=score_col,
                            popularity_col=pop_col,
                            race_id_col=race_col,
                            horse_col=horse_col,
                        )
                        sum_df["strategy_name"] = prefix + sum_df["strategy_name"]
                        det_df["strategy"] = prefix + det_df["strategy"]
                        summary_rows.append(sum_df)
                        detail_dfs.append(det_df)
                        
                    summary_df = pd.concat(summary_rows, ignore_index=True)
                    detail_df = pd.concat(detail_dfs, ignore_index=True)

                    rename_map = {
                        "strategy_name": "戦略",
                        "top_n": "上位N",
                        "total_races": "対象レース数",
                        "total_tickets": "購入点数",
                        "total_bet": "購入総額",
                        "total_return": "払戻総額",
                        "hit_tickets": "的中点数",
                        "hit_races": "的中レース数",
                        "ticket_hit_rate": "券単位的中率(%)",
                        "race_hit_rate": "レース単位的中率(%)",
                        "return_rate": "回収率(%)",
                    }
                    summary_view = summary_df.rename(columns=rename_map).copy()
                    for c in ["券単位的中率(%)", "レース単位的中率(%)", "回収率(%)"]:
                        if c in summary_view.columns:
                            summary_view[c] = summary_view[c].round(2)

                    st.success("Evaluatorでの比較が完了しました。")
                    st.markdown("#### 全体サマリー")
                    st.dataframe(summary_view, use_container_width=True)
                    st.caption("戦略: model=AI予測スコア順 / popularity=人気順")

                    meta_candidates = {"distance", "track_condition", "field_size", "距離", "馬場", "馬場状態", "頭数"}
                    meta_cols = list(meta_candidates.intersection(detail_df.columns))
                    if meta_cols:
                        st.markdown("#### レース条件別サマリー")
                        tabs = st.tabs(meta_cols)
                        for tab, m_col in zip(tabs, meta_cols):
                            with tab:
                                grp_rows = []
                                for (strat, val), group in detail_df.groupby(["strategy", m_col]):
                                    t_tic = len(group)
                                    t_race = group["race_id"].nunique()
                                    t_bet = group["購入額"].sum()
                                    t_ret = group["払戻額"].sum()
                                    h_tic = group["的中"].sum()
                                    h_race = group.groupby("race_id")["的中"].max().sum()
                                    grp_rows.append({
                                        "戦略": strat,
                                        m_col: val,
                                        "対象レース数": t_race,
                                        "購入点数": t_tic,
                                        "購入総額": t_bet,
                                        "払戻総額": t_ret,
                                        "的中点数": h_tic,
                                        "的中レース数": h_race,
                                        "券単位的中率(%)": round((h_tic / t_tic * 100) if t_tic else 0.0, 2),
                                        "レース単位的中率(%)": round((h_race / t_race * 100) if t_race else 0.0, 2),
                                        "回収率(%)": round((t_ret / t_bet * 100) if t_bet else 0.0, 2)
                                    })
                                if grp_rows:
                                    grp_df = pd.DataFrame(grp_rows)
                                    st.dataframe(grp_df, use_container_width=True)
                    else:
                        st.info("prediction CSV に距離、馬場、頭数などのメタデータ列が存在しないため、条件別サマリーはスキップされました。")

                    detail_download = detail_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    summary_download = summary_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button("📥 summary CSV", summary_download, file_name="evaluator_summary.csv", mime="text/csv")
                    with dl_col2:
                        st.download_button("📥 detail CSV", detail_download, file_name="evaluator_detail.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Evaluator実行中にエラーが発生しました: {e}")

    with st.expander("📈 Feature Importance（LightGBM特徴量重要度）", expanded=False):
        st.caption("保存済みの data/race_html/*.html からレース結果表を読み取り、AIへ渡すデータ要素から result_rank を除外してLightGBMを一度学習します。result_time など結果後にしか分からない列は、実運用予測では引き続きリークに注意してください。")
        fi_col1, fi_col2 = st.columns([2, 1])
        with fi_col1:
            fi_html_dir = st.text_input("レースHTML保存先", value="data/race_html", key="fi_html_dir")
        with fi_col2:
            fi_target = st.selectbox("目的変数", ["top3", "win"], index=0, format_func=lambda x: "3着以内" if x == "top3" else "1着")

        if st.button("LightGBMで学習して重要度を表示", key="run_feature_importance"):
            try:
                fi_result = run_feature_importance(html_dir=fi_html_dir, target_col=fi_target)
                importance_view = fi_result.importance.copy()
                importance_view["display_name"] = importance_view.get("label", importance_view["feature"])
                chart_df = importance_view.set_index("display_name")[["importance_pct"]]

                st.success(f"学習完了: {fi_result.dataset['race_id'].nunique()}レース / {len(fi_result.dataset)}頭")
                st.bar_chart(chart_df)
                st.dataframe(
                    importance_view[["feature", "label", "importance_pct", "gain_importance", "split_importance"]],
                    use_container_width=True
                )
                st.caption(f"保存先: {fi_result.output_csv} / {fi_result.output_svg}")
            except Exception as e:
                st.error(f"特徴量重要度の計算に失敗しました: {e}")

    if st.button("バックテスト実行"):
        if bt_effective_budget <= 0:
            st.warning("1レースあたりの最大予算を入力してください。")
            st.stop()
        if not bt_models:
            st.warning("検証するモデルを1つ以上選択してください。")
            st.stop()
        bt_race_ids = [r.strip() for r in bt_race_ids_str.split('\n') if r.strip()]
        if not bt_race_ids: st.warning("レースIDを入力してください。")
        else:
            csv_file = "backtest_results.csv"
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', encoding='utf-8-sig') as f:
                    f.write("Timestamp,RaceID,Model,TotalBet,TotalReturn,Hits,HitRate,ReturnRate,BetsJSON\n")
            
            prog_text = st.empty()
            prog_bar = st.progress(0)
            results_log = []
            all_tk_stats = {}

            for i, r_id in enumerate(bt_race_ids):
                prog_text.write(f"検証中: レースID {r_id} ({i+1}/{len(bt_race_ids)})")
                race_date = get_race_date(r_id)
                payouts = scrape_payouts(r_id)
                race_info = scraper.scrape_race_info(r_id)
                h_ids = scraper.get_horse_ids_from_race(r_id)
                if not h_ids: continue
                
                h_ids = h_ids[:bt_max_horses] # トークン節約のため頭数を絞る
                
                horses_data = []
                for hid in h_ids:
                    ped = scraper.scrape_horse_ped(hid)
                    res_df = scraper.scrape_race_results_dedicated(hid)
                    if not res_df.empty:
                        res_df['日付'] = pd.to_datetime(res_df['日付'], errors='coerce')
                        # バックテストでは予測基準日ではなく、レース開催日より前のデータを使用
                        res_df = res_df[res_df['日付'] < pd.Timestamp(race_date)] 
                    horses_data.append({"id": hid, "pedigree": ped, "results": res_df})
                
                race_info_lines = []
                if "race_id" in bt_selected_features:
                    race_info_lines.append(f"・対象レースID: {r_id}")
                if "race_name" in bt_selected_features:
                    race_info_lines.append(f"・レース名: {race_info['name']}")
                if "race_condition" in bt_selected_features:
                    race_info_lines.append(f"・コース・天候等: {race_info['data']}")

                ctx = "--- \n"
                if race_info_lines:
                    ctx += "【レース基本情報】\n" + "\n".join(race_info_lines) + "\n\n"
                if bt_user_opinion.strip():
                    ctx += f"【ユーザーからの特記事項・予想意見】\n{bt_user_opinion}\n※上記のユーザー意見を、今回の予想の重要な根拠の一つとして加味してください。\n\n"

                if bt_use_backtest_ml:
                    try:
                        ml_result = train_backtest_trend_model(r_id, max_horse_number=bt_max_horses)
                        if ml_result.context_text:
                            ctx += ml_result.context_text + "\n\n"
                    except Exception as e:
                        st.warning(f"バックテストML傾向を作成できませんでした: {e}")

                ctx += "【出走馬詳細】\n"

                shutuba_df = scraper.scrape_shutuba_table(r_id)
                bt_volatility = analyze_race_volatility(shutuba_df)
                volatility_prompt = format_volatility_for_prompt(bt_volatility)
                if volatility_prompt:
                    ctx += f"{volatility_prompt}\n\n"
                if bt_use_shutuba_table and not shutuba_df.empty:
                    selected_shutuba_df = get_selected_shutuba_df(shutuba_df, bt_selected_features)
                    ctx += "[出馬表]\n" + selected_shutuba_df.to_csv(index=False, sep='|') + "\n"

                bt_jockey_day_form_df = None
                if bt_use_jockey_day_form:
                    prog_text.write(f"{r_id} の騎手当日成績を取得中...")
                    bt_jockey_day_results_df = scraper.scrape_jockey_day_results(r_id)
                    bt_jockey_day_form_df = build_jockey_day_form_table(bt_jockey_day_results_df, shutuba_df)
                    jockey_prompt = format_jockey_day_form_for_prompt(
                        bt_jockey_day_form_df,
                        bt_jockey_day_results_df["レース番号"].nunique() if not bt_jockey_day_results_df.empty else 0,
                    )
                    if jockey_prompt:
                        ctx += jockey_prompt + "\n"
                    
                is_classic = any(keyword in race_info['name'] for keyword in ["桜花賞", "皐月賞", "優駿牝馬", "オークス", "東京優駿", "日本ダービー", "菊花賞"])
                classic_instruction = ""
                if is_classic and bt_use_blood:
                    classic_instruction = "※本レースは3歳クラシック競走のため、未知の距離・条件への対応力が重要になります。そのため、他の要素よりも「血統背景（特に母父や配合傾向）」を2倍重視して予想を行ってください。\n"

                for horse in horses_data:
                    ped = horse['pedigree']
                    res_df = horse['results']
                    
                    horse_header = f"\n[{ped['name']}]"
                    blood_parts = []
                    if "blood_sire" in bt_selected_features:
                        blood_parts.append(f"父:{ped['sire']}")
                    if "blood_dam" in bt_selected_features:
                        blood_parts.append(f"母:{ped['dam']}")
                    if "blood_broodmare_sire" in bt_selected_features:
                        blood_parts.append(f"母父:{ped['broodmare_sire']}")
                    if blood_parts:
                        horse_header += " " + " ".join(blood_parts)
                    ctx += horse_header + "\n"
                    
                    if res_df.empty:
                        ctx += "データなし\n"
                        continue
                    
                    if bt_use_analysis:
                        weight_info = ""
                        if "analysis_latest_weight" in bt_selected_features and '馬体重' in res_df.columns and res_df['馬体重'].notna().any():
                            latest_weight_str = res_df['馬体重'].dropna().iloc[0]
                            weight = re.match(r'(\d+)', str(latest_weight_str))
                            if weight: weight_info = f" 体重:{weight.group(1)}"

                        analysis_parts = []
                        if "analysis_running_style" in bt_selected_features:
                            analysis_parts.append(f"脚質:{analyze_running_style(res_df)}")
                        if "analysis_track_preference" in bt_selected_features:
                            analysis_parts.append(f"馬場:{analyze_track_preference(res_df)}")
                        if weight_info:
                            analysis_parts.append(weight_info.strip())
                        if analysis_parts:
                            ctx += " ".join(analysis_parts) + "\n"
                    
                    if bt_use_past_results:
                        existing_cols = get_selected_result_columns(res_df, bt_selected_features)
                        if not existing_cols:
                            continue
                        
                        ctx += res_df[existing_cols].head(10).to_csv(index=False, sep='|') + "\n"
                        
                        short_df = res_df[existing_cols].copy()
                        if '日付' in short_df.columns:
                            short_df['日付'] = pd.to_datetime(short_df['日付'], errors='coerce').dt.strftime('%y/%m/%d')
                        if 'レース名' in short_df.columns:
                            short_df['レース名'] = short_df['レース名'].astype(str).str.replace('ステークス', 'S').str.replace('カップ', 'C').str[:6]
                        if '着順' in short_df.columns:
                            short_df['着順'] = short_df['着順'].astype(str).str.extract(r'(\d+)')[0].fillna(short_df['着順'])
                        if '距離' in short_df.columns:
                            short_df['距離'] = short_df['距離'].astype(str).str.replace('m', '', regex=False)
                            
                        ctx += short_df.head(10).to_csv(index=False, header=False, sep=',') + "\n"

                prompt_parts = [
                    AI_PERSONA_PROMPT,
                    f"以下のデータを単一レース予想と同じ水準で複合的に評価し、レース分析を行ってください（分析はあなたの内部で行い、出力には含めないでください）。最大予算:{bt_effective_budget}円。",
                    "買い方、券種、点数、配分は完全にあなたの判断に委ねます。期待値が低い場合は0円または一部だけ賭ける判断も許可します。",
                    "金額は必ず100円単位にしてください（馬券は10円単位の調整ができないため）。50円、150円、250円など100の倍数ではない金額は絶対に出さないでください。",
                    classic_instruction,
                    "【出力要件】",
                    *OUTPUT_FORMAT_INSTRUCTIONS,
                    "券種・組み合わせ・金額は完全にあなたの判断で決めてください。",
                    "・買い目データ（※必ず以下のJSONフォーマットでテキストの最後に記述すること）",
                    "```json",
                    '{ "bets": [ {"type": "3連複", "combo": "1-2-3", "amount": 500} ] }', # 例示フォーマット
                    "```",
                    "【出走馬詳細】",
                    ctx
                ]
                prompt = "\n".join(prompt_parts)
                for model in bt_models:
                    prog_text.write(f"{r_id} を {model} で予想中...")
                    ai_res, bets = "", []
                    try:
                        if model == "ルールベース (AI不使用)":
                            rule_result = run_rule_based_analysis(
                                horses_data,
                                shutuba_df,
                                race_info,
                                r_id,
                                include_blood=bt_use_blood,
                                jockey_day_form_df=bt_jockey_day_form_df,
                            )
                            rule_result["_volatility"] = bt_volatility
                            ai_res = generate_rule_based_summary(
                                rule_result,
                                bt_effective_budget,
                                [],
                                "おまかせ",
                            )
                            bets = extract_bets_from_text(ai_res)
                        else:
                            ai_res = ask_gemini(prompt) if model == "Gemini" else ask_chatgpt(prompt)
                            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', ai_res, re.DOTALL | re.IGNORECASE)
                            if match: bets = json.loads(match.group(1)).get("bets", [])
                    except Exception as e:
                        st.error(f"モデル '{model}' の処理中にエラーが発生しました: {e}")
                        continue
                        
                    t_bet, t_ret, hits, tk_stats = calculate_return(bets, payouts)
                    for tk, tv in tk_stats.items():
                        if tk not in all_tk_stats: all_tk_stats[tk] = {"bet": 0, "return": 0, "hits": 0}
                        all_tk_stats[tk]["bet"] += tv["bet"]; all_tk_stats[tk]["return"] += tv["return"]; all_tk_stats[tk]["hits"] += tv["hits"]
                    
                    h_rate = (hits / len(bets) * 100) if bets else 0
                    r_rate = (t_ret / t_bet * 100) if t_bet > 0 else 0
                    with open(csv_file, 'a', encoding='utf-8-sig') as f:
                        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},{r_id},{model},{t_bet},{t_ret},{hits},{h_rate:.1f}%,{r_rate:.1f}%,{json.dumps(bets, ensure_ascii=False)}\n')
                    results_log.append({"RaceID": r_id, "Model": model, "TotalBet": t_bet, "TotalReturn": t_ret, "Hits": hits, "HitRate": f"{h_rate:.1f}%", "ReturnRate": f"{r_rate:.1f}%"})
                prog_bar.progress((i + 1) / len(bt_race_ids))
                
            prog_text.write("✅ バックテスト完了！")
            st.subheader("📊 バックテスト結果サマリー")
            if results_log:
                df_res = pd.DataFrame(results_log)
                grp = df_res.groupby("Model").agg({"TotalBet": "sum", "TotalReturn": "sum"}).reset_index()
                grp["ReturnRate"] = (grp["TotalReturn"] / grp["TotalBet"] * 100).fillna(0).apply(lambda x: f"{x:.1f}%")
                st.write("■ モデル別成績"); st.dataframe(grp, use_container_width=True)
                tk_df = pd.DataFrame.from_dict(all_tk_stats, orient='index').reset_index().rename(columns={"index": "券種", "bet": "TotalBet", "return": "TotalReturn", "hits": "Hits"})
                if not tk_df.empty:
                    tk_df["ReturnRate"] = (tk_df["TotalReturn"] / tk_df["TotalBet"] * 100).fillna(0).apply(lambda x: f"{x:.1f}%")
                    st.write("■ 券種別成績"); st.dataframe(tk_df, use_container_width=True)
                st.write("■ 詳細履歴"); st.dataframe(df_res, use_container_width=True)
            with open(csv_file, 'rb') as f: st.download_button("📥 CSVをダウンロード", f, file_name="backtest_results.csv")
    st.stop()

if app_mode == "予想履歴":
    st.title("📂 予想履歴")
    st.markdown("これまでにAIが予想した内容の履歴を確認できます。")
    
    log_file = "prediction_log.csv"
    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file, encoding='utf-8-sig')
            if df_log.empty:
                st.info("予想履歴はまだありません。")
            else:
                df_log = df_log.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
                
                st.dataframe(
                    df_log[["Timestamp", "RaceID", "Model", "Budget"]], 
                    use_container_width=True
                )
                
                st.markdown("### 📝 詳細確認")
                selected_idx = st.selectbox(
                    "確認したい予想を選択してください", 
                    df_log.index, 
                    format_func=lambda x: f"{df_log.loc[x, 'Timestamp']} - レースID: {df_log.loc[x, 'RaceID']} ({df_log.loc[x, 'Model']})"
                )
                
                if selected_idx is not None:
                    row = df_log.loc[selected_idx]
                    st.markdown(f"**日時:** {row['Timestamp']} | **レースID:** {row['RaceID']} | **モデル:** {row['Model']} | **予算:** {row['Budget']}円")
                    
                    with st.expander("AIへの指示（プロンプト）"):
                        st.text(row.get("Prompt", "データなし"))
                        
                    st.markdown("#### 🎯 AIの予想結果")
                    st.write(row.get("Response", "データなし"))
        except Exception as e:
            st.error(f"履歴の読み込み中にエラーが発生しました: {e}")
    else:
        st.info("予想履歴はまだありません。")
    st.stop()

if app_mode == "My収支":
    st.title("💰 My収支")
    st.caption(
        "単一レース予想の「結果照合」でチェックを入れて保存した予想と、"
        "手動で入力した実際の購入結果をもとに、月別・年別・券種別の収支を確認できます。"
    )

    with st.expander("✏️ 実際の購入結果を手動で追加", expanded=False):
        st.caption("AIの提案どおりに買わなかった場合や、アプリ外で購入した馬券もここから記録できます。")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            manual_date = st.date_input("購入日（レース施行日）", value=pd.Timestamp.now(), key="manual_balance_date")
            manual_race_id = st.text_input("Race ID（任意）", key="manual_balance_race_id")
        with mc2:
            manual_race_name = st.text_input("レース名（任意）", key="manual_balance_race_name")
            manual_ticket_type = st.selectbox(
                "券種", list(bet_plans.TICKET_ARITY.keys()) + ["その他"], key="manual_balance_ticket_type"
            )
        with mc3:
            manual_bet = st.number_input("購入金額（円）", min_value=0, step=100, value=0, key="manual_balance_bet")
            manual_return = st.number_input("払戻金額（円）", min_value=0, step=100, value=0, key="manual_balance_return")
        manual_memo = st.text_input("メモ（任意）", key="manual_balance_memo")

        if st.button("My収支に追加", key="manual_balance_add"):
            if manual_bet <= 0:
                st.warning("購入金額を入力してください。")
            else:
                hit = 1 if manual_return > 0 else 0
                balance.append_entries([{
                    "race_date": manual_date.strftime("%Y/%m/%d"),
                    "race_id": manual_race_id,
                    "race_name": manual_race_name,
                    "source": "手動入力",
                    "ticket_type": manual_ticket_type,
                    "bet_count": 1,
                    "hits": hit,
                    "total_bet": int(manual_bet),
                    "total_return": int(manual_return),
                    "ticket_stats": json.dumps(
                        {manual_ticket_type: {"bet": int(manual_bet), "return": int(manual_return), "hits": hit}},
                        ensure_ascii=False,
                    ),
                    "memo": manual_memo,
                }])
                st.success("My収支に追加しました。")
                st.rerun()

    balance_df = balance.load_balance_df()

    if balance_df.empty:
        st.info("まだMy収支の記録がありません。単一レース予想の「結果照合」でチェックを入れて保存するか、上の手動入力から追加してください。")
        st.stop()

    total_bet_all = balance_df["total_bet"].sum()
    total_return_all = balance_df["total_return"].sum()
    total_profit_all = total_return_all - total_bet_all
    total_return_rate = (total_return_all / total_bet_all * 100) if total_bet_all > 0 else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("総投資額", f"{int(total_bet_all):,}円")
    sc2.metric("総払戻額", f"{int(total_return_all):,}円")
    sc3.metric("収支", f"{int(total_profit_all):+,}円")
    sc4.metric("回収率", f"{total_return_rate:.1f}%")

    period_label = st.radio("集計単位", ["月別", "年別"], horizontal=True, key="balance_period_mode")
    freq = "M" if period_label == "月別" else "Y"
    period_df = balance.summarize_by_period(balance_df, freq=freq)

    if period_df.empty:
        st.info("集計できる日付データがありません。")
    else:
        st.markdown(f"#### {period_label}収支")
        st.bar_chart(period_df.set_index("period")[["収支"]])
        display_df = period_df.rename(columns={"period": "期間"}).copy()
        display_df["回収率"] = display_df["回収率"].astype(str) + "%"
        st.dataframe(display_df, use_container_width=True)

    st.markdown("#### 🎫 券種別の収支（全期間）")
    ticket_summary_df = balance.summarize_by_ticket_type(balance_df)
    if ticket_summary_df.empty:
        st.caption("券種別のデータがありません。")
    else:
        ticket_display = ticket_summary_df.copy()
        ticket_display["回収率"] = ticket_display["回収率"].astype(str) + "%"
        st.dataframe(ticket_display, use_container_width=True)

    with st.expander("📋 記録一覧（削除も可能）", expanded=False):
        list_df = balance_df.sort_values("race_date", ascending=False).copy()
        list_df["race_date"] = list_df["race_date"].dt.strftime("%Y/%m/%d")
        show_cols = ["entry_id", "race_date", "race_id", "race_name", "source", "ticket_type", "total_bet", "total_return", "profit", "memo"]
        st.dataframe(
            list_df[show_cols].rename(columns={
                "entry_id": "ID", "race_date": "日付", "race_id": "レースID", "race_name": "レース名",
                "source": "元", "ticket_type": "券種", "total_bet": "購入額", "total_return": "払戻額",
                "profit": "収支", "memo": "メモ",
            }),
            use_container_width=True,
        )

        delete_id = st.selectbox("削除する記録のID", [""] + list_df["entry_id"].tolist(), key="balance_delete_select")
        if st.button("この記録を削除", key="balance_delete_button"):
            if delete_id:
                balance.delete_entries([delete_id])
                st.success("削除しました。")
                st.rerun()
            else:
                st.warning("削除する記録を選択してください。")

    st.stop()

st.markdown(
    '<h1 class="app-title">競馬予想AI</h1>',
    unsafe_allow_html=True,
)

if "race_id_input" not in st.session_state:
    st.session_state.race_id_input = ""
if "pending_race_id_input" in st.session_state:
    st.session_state.race_id_input = st.session_state.pop("pending_race_id_input")

race_id = st.text_input("Race IDを入力 (例: 202405020811)", key="race_id_input")
prediction_date = st.date_input(
    "予想の基準日（この日より前のデータのみ使用）", 
    value=pd.Timestamp.now(),
    min_value=pd.Timestamp("1980-01-01")
)

if "race_info_preview" not in st.session_state:
    st.session_state.race_info_preview = None

if st.button("このRace IDのレース情報を確認", use_container_width=True):
    if not race_id:
        st.warning("Race IDを入力してください")
    else:
        with st.spinner("レース情報を取得中..."):
            st.session_state.race_info_preview = scraper.scrape_race_info(race_id)

if st.session_state.race_info_preview and st.session_state.race_info_preview.get("race_id") == race_id:
    preview = st.session_state.race_info_preview
    st.info(f"{preview.get('name', '')} / {preview.get('data', '')}")

# --- データの保存場所を準備（コードの上のほう、ボタンより前に書いておく） ---
if "all_horse_data" not in st.session_state:
    st.session_state.all_horse_data = None
if "pedigree_list" not in st.session_state:
    st.session_state.pedigree_list = None
if "shutuba_table" not in st.session_state:
    st.session_state.shutuba_table = None
if "latest_predictions" not in st.session_state:
    st.session_state.latest_predictions = {}
if "result_check_cache" not in st.session_state:
    st.session_state.result_check_cache = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "race_info" not in st.session_state:
    st.session_state.race_info = None
if "bet_plan_mode" not in st.session_state:
    st.session_state.bet_plan_mode = "おまかせ"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "full_ai_reports" not in st.session_state:
    st.session_state.full_ai_reports = {}
if "past_trend_df" not in st.session_state:
    st.session_state.past_trend_df = None
if "race_volatility" not in st.session_state:
    st.session_state.race_volatility = None
if "jockey_day_form" not in st.session_state:
    st.session_state.jockey_day_form = None
if "jockey_day_races" not in st.session_state:
    st.session_state.jockey_day_races = 0
if "hard_race_scan_results" not in st.session_state:
    st.session_state.hard_race_scan_results = []
if "bet_plan_blocks" not in st.session_state:
    st.session_state.bet_plan_blocks = []
if "gamiri_check_result" not in st.session_state:
    st.session_state.gamiri_check_result = None
if "generated_ai_prompt" not in st.session_state:
    st.session_state.generated_ai_prompt = None

with st.expander("当日レースID一覧から「堅い」レースを探す", expanded=False):
    st.caption("当日のレースIDを複数貼り付けると、出馬表と過去5年傾向から荒れ具合を判定します。堅いレースはそのまま総合予想と買い目提案へ進めます。")
    day_race_ids_text = st.text_area(
        "当日レースID（改行 or カンマ区切り）",
        value="",
        height=100,
        key="hard_race_ids_input",
        placeholder="例:\n202405020811\n202405020812"
    )

    if st.button("堅いレースを判定", key="scan_hard_races", use_container_width=True):
        day_race_ids = _parse_ids_input(day_race_ids_text)
        if not day_race_ids:
            st.warning("レースIDを入力してください。")
        else:
            scan_rows = []
            scan_progress = st.progress(0)
            scan_status = st.empty()
            for idx, rid in enumerate(day_race_ids, start=1):
                scan_status.write(f"判定中: {rid} ({idx}/{len(day_race_ids)})")
                try:
                    scan_shutuba_df = scraper.scrape_shutuba_table(rid)
                    scan_race_info = scraper.scrape_race_info(rid)
                    scan_past_trend_df = scraper.scrape_past_top3_trend(rid, years=5)
                    volatility = analyze_race_volatility(scan_shutuba_df, scan_past_trend_df)
                    scan_rows.append({
                        "race_id": rid,
                        "レース名": scan_race_info.get("name", ""),
                        "条件": scan_race_info.get("data", ""),
                        "荒れ指数": volatility["score"],
                        "判定": volatility["level"],
                        "方針": volatility["summary"],
                        "根拠": " / ".join(volatility.get("reasons", [])),
                    })
                except Exception as e:
                    scan_rows.append({
                        "race_id": rid,
                        "レース名": "",
                        "条件": "",
                        "荒れ指数": None,
                        "判定": "取得失敗",
                        "方針": "",
                        "根拠": str(e),
                    })
                scan_progress.progress(idx / len(day_race_ids))
            scan_status.empty()
            scan_progress.empty()
            st.session_state.hard_race_scan_results = scan_rows

    if st.session_state.hard_race_scan_results:
        scan_df = pd.DataFrame(st.session_state.hard_race_scan_results)
        hard_df = scan_df[scan_df["判定"] == "堅い"].copy()
        st.dataframe(scan_df, use_container_width=True)

        if hard_df.empty:
            st.info("今回の一覧には、判定が「堅い」のレースは見つかりませんでした。")
        else:
            st.success(f"堅いレース候補: {len(hard_df)}件")
            selected_hard_race = st.selectbox(
                "総合予想へ進むレース",
                hard_df["race_id"].tolist(),
                format_func=lambda rid: f"{rid} - {hard_df.loc[hard_df['race_id'] == rid, 'レース名'].iloc[0]}",
                key="selected_hard_race_id"
            )
            if st.button("この堅いレースで総合予想へ進む", key="load_selected_hard_race", use_container_width=True):
                st.session_state.pending_race_id_input = selected_hard_race
                with st.spinner("選択した堅いレースの詳細データを取得中..."):
                    race_data = collect_single_race_data(selected_hard_race, prediction_date)
                if race_data is None:
                    st.error("出走馬が見つかりませんでした。")
                else:
                    save_single_race_data(race_data)
                    st.rerun()

# --- AI予想開始ボタン ---
if st.button("データ取得開始"):
    if ai_choice in ["Gemini", "両方で比較"] and not API_KEY_GEMINI:
        st.error("Gemini APIキーがありません")
    elif ai_choice in ["ChatGPT", "両方で比較"] and not API_KEY_OPENAI:
        st.error("OpenAI APIキーがありません")
    elif not race_id:
        st.warning("Race IDを入力してください")
    else:
        with st.spinner("出走馬データを収集中..."):
            race_data = collect_single_race_data(race_id, prediction_date)
            if race_data is None:
                st.error("出走馬が見つかりませんでした。")
            else:
                save_single_race_data(race_data)
                st.rerun()

# --- ここから表示フェーズ（データがあるときだけ自動で表示される） ---
if st.session_state.all_horse_data:
    # 🆕 レース情報の表示
    if st.session_state.race_info:
        r_info = st.session_state.race_info
        st.markdown(f"## 🏆 {r_info['name']}")
        st.markdown(f"**{r_info['data']}**")
        race_meta = []
        if r_info.get("start_time"):
            race_meta.append(f"発走: {r_info['start_time']}")
        if r_info.get("course"):
            race_meta.append(f"条件: {r_info['course']}")
        if r_info.get("place"):
            race_meta.append(f"競馬場: {r_info['place']}")
        if r_info.get("entries"):
            race_meta.append(f"頭数: {r_info['entries']}")
        if race_meta:
            st.caption(" / ".join(race_meta))
        st.markdown("---")

    # 🆕 取得した公式出馬表の表示
    if st.session_state.shutuba_table is not None and not st.session_state.shutuba_table.empty:
        st.subheader("📊 本レース出馬表（馬番・騎手・斤量・馬体重など）")
        st.dataframe(st.session_state.shutuba_table, use_container_width=True)

    if st.session_state.race_volatility:
        volatility = st.session_state.race_volatility
        st.subheader("🎯 荒れ具合と推奨購入点数")
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.metric("荒れ指数", f"{volatility['score']}/100")
        c2.metric("判定", volatility["level"])
        c3.info(volatility["summary"])
        if volatility.get("reasons"):
            st.caption("根拠: " + " / ".join(volatility["reasons"]))
        st.dataframe(pd.DataFrame(volatility["ticket_rows"]), use_container_width=True)

    st.subheader("⚔️ 出走馬の対戦表（過去同一レース比較）")
    race_compare_count = st.slider("表示する比較レース数", min_value=3, max_value=12, value=8, step=1)
    race_comparison_df = _build_race_comparison_table(
        st.session_state.all_horse_data,
        st.session_state.shutuba_table,
        max_races=race_compare_count
    )
    if race_comparison_df.empty:
        st.caption("今回の出走馬同士で共通出走したレースが見つかりませんでした。")
    else:
        st.caption("同じレースに出走していた馬の着順・タイムを横並びで比較できます（1〜3着は色付き）。")
        st.dataframe(_style_race_comparison(race_comparison_df), use_container_width=True, height=600)

    if st.session_state.past_trend_df is not None and not st.session_state.past_trend_df.empty:
        st.subheader("🕰️ 同レース過去5年（1〜3着の傾向）")
        st.caption("馬番・騎手・コーナー通過順・人気を重点表示")
        st.dataframe(st.session_state.past_trend_df, use_container_width=True)

    st.subheader("🏇 騎手の当日成績")
    jockey_form_df = st.session_state.jockey_day_form
    if jockey_form_df is None or jockey_form_df.empty or not (jockey_form_df["当日騎乗"] > 0).any():
        st.caption("同日・同競馬場で本レースより前の結果が見つかりませんでした（第1レース、または当日の結果が未確定です）。")
    else:
        st.caption(
            f"同日・同競馬場の1R〜{st.session_state.jockey_day_races}Rの結果から集計。"
            "着別度数は1着-2着-3着-着外、人気差平均は(人気-着順)の平均でプラスほど人気以上に好走。"
            "この表はAI予想のプロンプトにも含まれます。"
        )
        st.dataframe(jockey_form_df, use_container_width=True, hide_index=True)

    # ① 出馬表（血統情報）
    st.subheader("🧬 出馬表（血統情報）")
    df_pedigree = pd.DataFrame(st.session_state.pedigree_list)
    df_pedigree = df_pedigree.rename(columns={
        "horse_id": "馬ID", "name": "馬名", "sire": "父", "dam": "母", "broodmare_sire": "母父"
    })
    st.dataframe(df_pedigree, use_container_width=True)

    # 各馬の詳細情報
    st.subheader("🐎 各馬の詳細情報と直近戦績")
    for horse in st.session_state.all_horse_data:
        with st.expander(f"{horse['pedigree']['name']} (ID: {horse['id']})"):
            running_style = analyze_running_style(horse['results'])
            track_preference = analyze_track_preference(horse['results'])
            st.write(f"**父:** {horse['pedigree']['sire']} / **母:** {horse['pedigree']['dam']} / **母父:** {horse['pedigree']['broodmare_sire']}")
            st.write(f"**脚質:** {running_style} | **馬場適性:** {track_preference}")
            if not horse['results'].empty:
                st.dataframe(horse['results'].head(10), use_container_width=True)
            else:
                st.write("戦績データがありません。")

    st.markdown("---")
    st.subheader("🚀 AI総合予想と賭け方提案")
    st.caption("血統、各馬の過去10戦、出馬表をAIへ渡してレース分析を行います。買い方プランを指定しない場合は、最大予算内でAIが自由に買い目を提案します。")

    with st.expander("🎯 買い方セレクト（券種・方式・目標点数・配分）", expanded=True):
        render_bet_plan_selector()

    plan_blocks = st.session_state.bet_plan_blocks
    plan_total = sum(bet_plans.block_total_amount(b) for b in plan_blocks)

    full_cols = st.columns([1, 1])
    with full_cols[0]:
        full_max_budget = st.number_input(
            "最大予算 (円) ※買い方プラン未指定時にAIが自由提案する上限",
            min_value=0,
            max_value=1000000,
            value=1000,
            step=100,
            key="full_ai_max_budget",
            disabled=bool(plan_blocks),
        )
    with full_cols[1]:
        use_full_budget = st.checkbox(
            "最大予算をすべて使う",
            value=False,
            key="full_ai_use_full_budget",
            disabled=bool(plan_blocks),
        )

    if plan_blocks:
        st.success(f"買い方プランが指定されています（見積合計{plan_total}円）。この点数・配分を目安にAIが馬番を選んで買い目を提案します。")
    else:
        st.caption("買い方プランが未指定のため、AIが最大予算内で買い方を自由に提案します。")

    if st.button("血統・過去10戦・出馬表からAI分析して買い方を提案", use_container_width=True):
        if ai_choice == "ルールベース (AI不使用)":
            st.warning("AI総合予想はGeminiまたはChatGPTを選んでください。")
        elif ai_choice in ["Gemini", "両方で比較"] and not API_KEY_GEMINI:
            st.error("Gemini APIキーがありません")
        elif ai_choice in ["ChatGPT", "両方で比較"] and not API_KEY_OPENAI:
            st.error("OpenAI APIキーがありません")
        else:
            st.session_state.full_ai_reports = run_full_ai_betting_analysis(
                race_id,
                int(full_max_budget),
                plan_blocks,
                use_full_budget,
            )
            for model_name, response in st.session_state.full_ai_reports.items():
                st.session_state.latest_predictions[model_name] = {
                    "race_id": race_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bets": extract_bets_from_text(response),
                    "response": response,
                }
            st.rerun()

    if st.session_state.full_ai_reports:
        st.markdown("### AI総合予想結果")
        for model_name, response in st.session_state.full_ai_reports.items():
            with st.expander(f"{model_name} の分析・推奨賭け金・買い目", expanded=True):
                st.write(response)

    st.markdown("---")
    st.subheader("📋 他のAI（Claudeなど）に渡すプロンプトを出力")
    st.caption(
        "GeminiやChatGPTのAPIを使わず、血統・過去10戦・出馬表など取得済みデータをすべて含んだプロンプトをテキストで出力します。"
        "これをコピーしてClaudeなど任意のAIチャットに貼り付けるだけで、同じ分析・買い目提案を依頼できます。"
    )
    if st.button("📋 AIに渡すプロンプトを出力", use_container_width=True, key="generate_shareable_prompt"):
        st.session_state.generated_ai_prompt = build_full_ai_betting_prompt(
            race_id,
            int(full_max_budget),
            plan_blocks,
            use_full_budget,
        )
        st.rerun()

    if st.session_state.generated_ai_prompt:
        st.text_area(
            "生成されたプロンプト（全選択してコピーし、Claudeなど任意のAIに貼り付けてください）",
            value=st.session_state.generated_ai_prompt,
            height=400,
            key="generated_ai_prompt_display",
        )
        st.download_button(
            "📥 プロンプトをテキストファイルでダウンロード",
            data=st.session_state.generated_ai_prompt,
            file_name=f"ai_prompt_{race_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 オッズ確認・ガミリチェック（購入前）")
        st.caption(
            "現在のオッズを取得し、買い目のどれかが的中しても購入合計額を下回ってしまう"
            "「ガミリ」が起きていないかを購入前にチェックします。"
        )
        gamiri_target_models = [
            model_name for model_name, pred in st.session_state.latest_predictions.items()
            if pred.get("race_id") == race_id and pred.get("bets")
        ]
        if not gamiri_target_models:
            st.caption("買い目JSONが取得できているAI予想がないため、チェックできません。")
        else:
            gamiri_model = st.selectbox("チェックする予想", gamiri_target_models, key="gamiri_model_select")
            if st.button("最新オッズを取得してガミリをチェック", key="gamiri_check_button"):
                target_bets = st.session_state.latest_predictions[gamiri_model]["bets"]
                ticket_types_needed = sorted({b.get("type", "") for b in target_bets if b.get("type")})
                with st.spinner("最新オッズを取得中..."):
                    odds_data = scraper.scrape_realtime_odds(race_id, ticket_types_needed)
                rows, total_bet = check_gamiri(target_bets, odds_data)
                st.session_state.gamiri_check_result = {
                    "model": gamiri_model, "race_id": race_id, "rows": rows, "total_bet": total_bet,
                }

            gamiri_result = st.session_state.get("gamiri_check_result")
            if gamiri_result and gamiri_result.get("model") == gamiri_model and gamiri_result.get("race_id") == race_id:
                rows = gamiri_result["rows"]
                total_bet = gamiri_result["total_bet"]
                if not rows:
                    st.info("有効な買い目がありません。")
                else:
                    df_rows = pd.DataFrame(rows)
                    display_df = df_rows.rename(columns={
                        "type": "券種", "combo": "組み合わせ", "amount": "購入額",
                        "odds": "現在オッズ", "est_return": "的中時の推定払戻", "profit": "購入合計との損益",
                    }).copy()
                    display_df["判定"] = df_rows["is_gamiri"].map(
                        lambda x: "⚠️ ガミリ" if x is True else ("✅ 問題なし" if x is False else "オッズ取得不可")
                    )
                    st.dataframe(display_df.drop(columns=["is_gamiri"]), use_container_width=True)
                    st.caption(f"購入合計: {total_bet}円（ワイド・複勝は最低オッズで保守的に計算しています）")

                    gamiri_rows = df_rows[df_rows["is_gamiri"] == True]
                    not_found_rows = df_rows[df_rows["is_gamiri"].isna()]
                    if not gamiri_rows.empty:
                        st.warning(
                            f"⚠️ {len(gamiri_rows)}点で、的中しても購入合計{total_bet}円を下回る「ガミリ」の"
                            "可能性があります。配分金額の見直しをおすすめします。"
                        )
                    elif not_found_rows.empty:
                        st.success("現在のオッズでは、どの買い目が的中しても購入合計を下回らない見込みです。")
                    if not not_found_rows.empty:
                        st.caption(f"{len(not_found_rows)}点はオッズが取得できませんでした（発売前・対象外の可能性があります）。")

    data_context = build_full_ai_betting_prompt(
        race_id,
        int(st.session_state.get("full_ai_max_budget", 1000)),
        plan_blocks,
        st.session_state.get("full_ai_use_full_budget", False),
    )

    # --- 💬 AIと相談機能の追加 ---
    st.markdown("---")
    st.subheader("💬 AIアシスタントと相談して決める")
    st.caption("集めたデータや予想結果をもとに、AIアシスタントと対話しながら最終的な買い目を検討できます。")
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if chat_prompt := st.chat_input(""):
        st.session_state.chat_messages.append({"role": "user", "content": chat_prompt})
        with st.chat_message("user"):
            st.markdown(chat_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                chat_context = AI_PERSONA_PROMPT + "\n"
                chat_context += "ユーザーの質問に答え、共に勝ち筋を探る共有型コンサルタントとして、冷静に最適な買い目を検討してください。\n\n"
                chat_context += f"【対象レースのデータ・コンテキスト】\n{data_context}\n\n"
                chat_context += "【これまでの会話】\n"
                for m in st.session_state.chat_messages[:-1]:
                    role_name = "ユーザー" if m["role"] == "user" else "アシスタント"
                    chat_context += f"{role_name}: {m['content']}\n"
                chat_context += f"ユーザー: {chat_prompt}\nアシスタント:"
                
                try:
                    # 「両方で比較」を選んでいる場合はGeminiが応答します
                    chat_ai = "ChatGPT" if ai_choice == "ChatGPT" else "Gemini"
                    response = ask_chatgpt(chat_context) if chat_ai == "ChatGPT" else ask_gemini(chat_context)
                        
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"返答の生成中にエラーが発生しました: {e}")

    has_prediction_for_current_race = any(
        pred.get("race_id") == race_id
        for pred in st.session_state.latest_predictions.values()
    )

    if has_prediction_for_current_race:
        st.subheader("💴 結果照合（過去レース限定・任意実行）")
        st.caption(
            "予想後にボタンを押すと、結果・払戻ページを参照して的中と回収金額を計算します。"
            "My収支に反映したい予想にチェックを入れて保存すると、月別・年別の収支に集計されます。"
        )
        if st.button("結果ページで的中チェックを実行"):
            if not race_id:
                st.warning("Race IDを入力してください。")
            else:
                result_payload = scraper.scrape_race_result_page(race_id)
                if not result_payload.get("ok"):
                    st.error("結果ページの取得に失敗しました。過去レースIDかどうかを確認してください。")
                else:
                    payouts = result_payload.get("payouts", {})
                    if not payouts:
                        st.warning("払戻情報を取得できませんでした。")
                    else:
                        st.success(f"結果ページを確認しました: {result_payload.get('url')}")

                    race_date = scraper.get_race_date(race_id)
                    race_name = (st.session_state.race_info or {}).get("name", "")
                    race_results = {}
                    for model_name, pred in st.session_state.latest_predictions.items():
                        if pred.get("race_id") != race_id:
                            continue

                        bets = pred.get("bets", [])
                        if not bets:
                            st.info(f"{model_name}: 買い目JSONが見つからないため、金額集計をスキップしました。")
                            continue

                        total_bet, total_return, hits, ticket_stats = calculate_return(bets, payouts)
                        race_results[model_name] = {
                            "race_date": race_date,
                            "race_name": race_name,
                            "bet_count": len(bets),
                            "hits": hits,
                            "total_bet": total_bet,
                            "total_return": total_return,
                            "ticket_stats": ticket_stats,
                            "saved": False,
                        }
                    st.session_state.result_check_cache[race_id] = race_results

        cached_results = st.session_state.result_check_cache.get(race_id)
        if cached_results:
            for model_name, r in cached_results.items():
                hit_rate = (r["hits"] / r["bet_count"] * 100) if r["bet_count"] else 0
                return_rate = (r["total_return"] / r["total_bet"] * 100) if r["total_bet"] > 0 else 0
                st.markdown(f"#### {model_name} 的中結果")
                st.write(f"購入合計: {r['total_bet']}円 / 払戻合計: {r['total_return']}円 / 回収率: {return_rate:.1f}% / 的中率: {hit_rate:.1f}%")
                if r["ticket_stats"]:
                    ticket_df = pd.DataFrame.from_dict(r["ticket_stats"], orient='index').reset_index().rename(
                        columns={"index": "券種", "bet": "購入額", "return": "払戻額", "hits": "的中数"}
                    )
                    ticket_df["回収率"] = (ticket_df["払戻額"] / ticket_df["購入額"] * 100).fillna(0).round(1).astype(str) + "%"
                    st.dataframe(ticket_df, use_container_width=True)

                if r["saved"]:
                    st.success("✅ My収支に追加済みです。")
                else:
                    r["include_in_balance"] = st.checkbox(
                        f"この{model_name}の結果をMy収支に追加する",
                        key=f"balance_chk_{race_id}_{model_name}",
                    )

            to_save = [
                (model_name, r) for model_name, r in cached_results.items()
                if not r["saved"] and r.get("include_in_balance")
            ]
            if to_save and st.button("チェックした結果をMy収支に保存", key=f"balance_save_{race_id}"):
                entries = []
                for model_name, r in to_save:
                    entries.append({
                        "race_date": r["race_date"],
                        "race_id": race_id,
                        "race_name": r["race_name"],
                        "source": model_name,
                        "ticket_type": ",".join(r["ticket_stats"].keys()) if r["ticket_stats"] else "",
                        "bet_count": r["bet_count"],
                        "hits": r["hits"],
                        "total_bet": r["total_bet"],
                        "total_return": r["total_return"],
                        "ticket_stats": json.dumps(r["ticket_stats"], ensure_ascii=False),
                        "memo": "",
                    })
                    r["saved"] = True
                balance.append_entries(entries)
                st.success(f"{len(entries)}件をMy収支に追加しました。")
                st.rerun()

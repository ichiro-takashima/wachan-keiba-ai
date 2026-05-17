import streamlit as st
import pandas as pd
import time
import os
import re
import json
import csv
import scraper
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
from tickets import (
    BETTING_METHOD_OPTIONS,
    TICKET_TYPE_OPTIONS,
    build_ticket_plan_text,
    calculate_return,
    create_ticket_request,
    extract_bets_from_text,
    parse_ids_input as _parse_ids_input,
    ticket_type_requires_combo_count,
    ticket_type_supports_betting_method,
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


# --- Streamlit UI ---
st.sidebar.title("🏇 メニュー")
app_mode = st.sidebar.radio("モード選択", ["単一レース予想", "バックテスト", "予想履歴"], index=0)

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

    st.markdown("#### 買い方設定 (バックテスト用)")
    if "bt_custom_ticket_requests" not in st.session_state:
        st.session_state.bt_custom_ticket_requests = [create_ticket_request()]
    else:
        normalized_requests = []
        for req in st.session_state.bt_custom_ticket_requests:
            normalized_req = create_ticket_request()
            normalized_req.update(req if isinstance(req, dict) else {})
            if normalized_req.get("betting_method") not in BETTING_METHOD_OPTIONS:
                normalized_req["betting_method"] = "通常"
            normalized_requests.append(normalized_req)
        st.session_state.bt_custom_ticket_requests = normalized_requests or [create_ticket_request()]

    bt_bet_plan_mode = st.radio(
        "買い方を選んでください",
        ["おまかせ", "カスタム"],
        key="bt_bet_plan_mode",
        horizontal=True
    )

    if bt_bet_plan_mode == "カスタム":
        st.caption("券種ごとの条件を追加できます。単勝・複勝以外は方式（通常／フォーメーション／ながし／ボックス）も指定できます。")
        bt_remove_request_index = None
        for idx, req in enumerate(st.session_state.bt_custom_ticket_requests):
            st.markdown(f"**条件 {idx + 1}**")
            cols = st.columns([1.5, 1.2, 1.0, 0.9, 1.0, 0.6])
            default_type = req.get("ticket_type", "ワイド")
            ticket_type = cols[0].selectbox(
                "券種",
                TICKET_TYPE_OPTIONS,
                index=TICKET_TYPE_OPTIONS.index(default_type) if default_type in TICKET_TYPE_OPTIONS else 0,
                key=f"bt_ticket_type_{idx}"
            )
            default_method = req.get("betting_method", "通常")
            if ticket_type_supports_betting_method(ticket_type):
                betting_method = cols[1].selectbox(
                    "方式",
                    BETTING_METHOD_OPTIONS,
                    index=BETTING_METHOD_OPTIONS.index(default_method) if default_method in BETTING_METHOD_OPTIONS else 0,
                    key=f"bt_ticket_method_{idx}"
                )
            else:
                cols[1].markdown("方式指定なし")
                betting_method = "通常"
            request_budget = cols[2].number_input(
                "予算 (円)",
                min_value=100,
                step=100,
                value=int(req.get("budget", 500) or 500),
                key=f"bt_ticket_budget_{idx}"
            )
            if ticket_type_requires_combo_count(ticket_type):
                combo_count = cols[3].number_input(
                    "組数",
                    min_value=1,
                    step=1,
                    value=int(req.get("combo_count", 3) or 3),
                    key=f"bt_ticket_combo_{idx}"
                )
            else:
                cols[3].markdown("組数指定なし")
                combo_count = 0
            axis_horse = cols[4].text_input(
                "軸馬(任意)",
                value=req.get("axis_horse", ""),
                key=f"bt_ticket_axis_{idx}",
                help="カンマ区切りで入力（例: 1,3）"
            )
            
            if cols[5].button("削除", key=f"bt_remove_ticket_{idx}"):
                bt_remove_request_index = idx

            st.session_state.bt_custom_ticket_requests[idx] = {
                "ticket_type": ticket_type,
                "budget": int(request_budget),
                "combo_count": int(combo_count),
                "betting_method": betting_method,
                "axis_horse": axis_horse
            }

        action_cols = st.columns(2)
        if action_cols[0].button("条件を追加", key="bt_add_ticket_request"):
            st.session_state.bt_custom_ticket_requests.append(create_ticket_request())
            st.rerun()
        if action_cols[1].button("最後の条件を削除", key="bt_remove_last_ticket_request"):
            if len(st.session_state.bt_custom_ticket_requests) > 1:
                st.session_state.bt_custom_ticket_requests.pop()
                st.rerun()

        if bt_remove_request_index is not None:
            st.session_state.bt_custom_ticket_requests.pop(bt_remove_request_index)
            if not st.session_state.bt_custom_ticket_requests:
                st.session_state.bt_custom_ticket_requests = [create_ticket_request()]
            st.rerun()

        bt_custom_total_budget = sum(req.get("budget", 0) for req in st.session_state.bt_custom_ticket_requests)
        st.info(f"カスタム条件の合計予算: {bt_custom_total_budget}円")
        bt_budget = bt_custom_total_budget
    else:
        bt_budget = st.number_input("1レースあたりの予算 (円)", min_value=100, step=100, value=1000, key="bt_overall_budget")
        st.caption("買い方希望がない場合はこちら。現状の予想モデルで券種と配分をおまかせ提案します。")

    bt_ticket_plan_text, bt_effective_budget = build_ticket_plan_text(
        bt_bet_plan_mode,
        st.session_state.get("bt_custom_ticket_requests", []),
        bt_budget
    )

    bt_use_backtest_ml = st.checkbox(
        "バックテスト結果からML傾向を学習して予想に反映する",
        value=True,
        key="bt_use_backtest_ml",
        help="backtest_results.csv の買い目履歴を使い、同一レースID系統で回収率が高かった馬券種・馬番構成をプロンプトへ追加します。"
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
        if bt_bet_plan_mode == "カスタム" and bt_effective_budget <= 0:
            st.warning("カスタムの買い方条件を1件以上、予算ありで設定してください。")
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

                # バックテスト用の買い目構築ルールをプロンプトに追加
                ctx += f"【買い目構築ルール】\n{bt_ticket_plan_text}\n\n"

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
                        
                        ctx += res_df[existing_cols].head(3).to_csv(index=False, sep='|') + "\n"
                        
                        short_df = res_df[existing_cols].copy()
                        if '日付' in short_df.columns:
                            short_df['日付'] = pd.to_datetime(short_df['日付'], errors='coerce').dt.strftime('%y/%m/%d')
                        if 'レース名' in short_df.columns:
                            short_df['レース名'] = short_df['レース名'].astype(str).str.replace('ステークス', 'S').str.replace('カップ', 'C').str[:6]
                        if '着順' in short_df.columns:
                            short_df['着順'] = short_df['着順'].astype(str).str.extract(r'(\d+)')[0].fillna(short_df['着順'])
                        if '距離' in short_df.columns:
                            short_df['距離'] = short_df['距離'].astype(str).str.replace('m', '', regex=False)
                            
                        ctx += short_df.head(5).to_csv(index=False, header=False, sep=',') + "\n"

                prompt_parts = [
                    f"あなたはプロの競馬予想家です。不要な前置き・一般論は省略し、合計300トークン以内で簡潔に予想を出力してください。予算:{bt_effective_budget}円。",
                    classic_instruction,
                    "【出力要件】",
                    "・出力: 本命1、対抗1、穴を複数頭",
                    "・対象: 上位評価7頭のみ詳述（根拠は各馬2行以内）、それ以外は1行評価",
                    "・買い目データ（※必ず以下のJSONフォーマットでテキストの最後に記述すること）",
                    "```json",
                    '{ "bets": [ {{"type": "3連複", "combo": "1-2-3", "amount": 500}} ] }', # 例示フォーマット
                    "```",
                    "【出走馬詳細】",
                    ctx # ctxには既に買い目構築ルールが含まれている
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
                            )
                            rule_result["_volatility"] = bt_volatility
                            ai_res = generate_rule_based_summary(
                                rule_result,
                                bt_effective_budget,
                                st.session_state.get("bt_custom_ticket_requests", []),
                                bt_bet_plan_mode,
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

st.markdown(
    '<h1 class="app-title">A horse racing prediction AI</h1>',
    unsafe_allow_html=True,
)

race_id = st.text_input("Race IDを入力 (例: 202405020811)")
prediction_date = st.date_input(
    "予想の基準日（この日より前のデータのみ使用）", 
    value=pd.Timestamp.now(),
    min_value=pd.Timestamp("1980-01-01")
)

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
if "custom_ticket_requests" not in st.session_state:
    st.session_state.custom_ticket_requests = [create_ticket_request()]
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "past_trend_df" not in st.session_state:
    st.session_state.past_trend_df = None
if "race_volatility" not in st.session_state:
    st.session_state.race_volatility = None

# --- AI予想開始ボタン ---
if st.button("データ取得開始"):
    if not API_KEY_GEMINI:
        st.error("APIキーがありません")
    elif not race_id:
        st.warning("Race IDを入力してください")
    else:
        with st.spinner("出走馬データを収集中..."):
            h_ids = scraper.get_horse_ids_from_race(race_id)
            if not h_ids:
                st.error("出走馬が見つかりませんでした。")
            else:
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
                        cutoff_date = pd.Timestamp(prediction_date)
                        results_df = results_df[results_df['日付'] < cutoff_date]
                        results_df['日付'] = results_df['日付'].dt.strftime('%Y/%m/%d')
                    
                    temp_horse_data.append({"id": hid, "pedigree": ped_data, "results": results_df})
                    progress_bar.progress((i + 1) / len(h_ids))

                # 出馬表のスクレイピングを実行
                status_text.write("📊 出馬表（レース情報）を取得中...")
                shutuba_df = scraper.scrape_shutuba_table(race_id)
                race_info = scraper.scrape_race_info(race_id)
                past_trend_df = scraper.scrape_past_top3_trend(race_id, years=5)

                # データをセッションに保存して再起動
                st.session_state.all_horse_data = temp_horse_data
                st.session_state.pedigree_list = temp_pedigree_list
                st.session_state.shutuba_table = shutuba_df
                st.session_state.race_info = race_info
                st.session_state.past_trend_df = past_trend_df
                st.session_state.race_volatility = analyze_race_volatility(shutuba_df, past_trend_df)
                st.session_state.chat_messages = [] # 別のレースを予想する際にチャット履歴をリセット
                st.session_state.analysis_results = {}
                st.rerun()

# --- ここから表示フェーズ（データがあるときだけ自動で表示される） ---
if st.session_state.all_horse_data:
    # 🆕 レース情報の表示
    if st.session_state.race_info:
        r_info = st.session_state.race_info
        st.markdown(f"## 🏆 {r_info['name']}")
        st.markdown(f"**{r_info['data']}**")
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
                st.dataframe(horse['results'].head(5), use_container_width=True)
            else:
                st.write("戦績データがありません。")

    st.subheader(f"🤖 {ai_choice} によるレース分析")

    selected_features = render_ai_data_feature_selector(
        "race",
        "AIに渡す情報を細かく絞り込むことで、特定のデータに引っ張られる過学習やノイズを防ぎます。"
    )
    use_backtest_ml = st.checkbox(
        "バックテスト結果からML傾向を学習して予想に反映する",
        value=True,
        key="race_use_backtest_ml",
        help="backtest_results.csv の買い目履歴を使い、同一レースID系統の買い目傾向を予想プロンプトへ追加します。"
    )
    use_past_trend = feature_enabled(selected_features, "past_trend_rows", "past_trend_summary")
    use_shutuba_table = feature_enabled(selected_features, "shutuba_umaban", "shutuba_waku", "shutuba_name", "shutuba_sex_age", "shutuba_jockey", "shutuba_weight_carried", "shutuba_odds", "shutuba_popularity", "shutuba_stable", "shutuba_body_weight")
    use_blood = feature_enabled(selected_features, "blood_sire", "blood_dam", "blood_broodmare_sire")
    use_analysis = feature_enabled(selected_features, "analysis_running_style", "analysis_track_preference", "analysis_latest_weight")
    use_past_results = any(key.startswith("result_") for key in selected_features)

    col_op1, col_op2 = st.columns([4, 1])
    with col_op1:
        st.markdown("**✍️ あなたの予想・注目馬（AIへの指示や意見があれば入力してください）**")

    user_opinion = st.text_area(
        "あなたの予想・注目馬",
        placeholder="例: 1番の馬の逃げ残りに期待。雨が降っているので外枠有利。",
        label_visibility="collapsed"
    )

    r_info = st.session_state.race_info or {"name": "不明", "data": "不明"}
    race_info_lines = []
    if "race_id" in selected_features:
        race_info_lines.append(f"・対象レースID: {race_id}")
    if "race_name" in selected_features:
        race_info_lines.append(f"・レース名: {r_info['name']}")
    if "race_condition" in selected_features:
        race_info_lines.append(f"・コース・天候等: {r_info['data']}")

    data_context = "---\n"
    if race_info_lines:
        data_context += "【レース基本情報】\n" + "\n".join(race_info_lines) + "\n"

    if user_opinion.strip():
        data_context += f"\n【ユーザーからの特記事項・予想意見】\n{user_opinion}\n※上記のユーザー意見を、今回の予想の重要な根拠の一つとして加味してください。\n"

    volatility_prompt = format_volatility_for_prompt(st.session_state.race_volatility)
    if volatility_prompt:
        data_context += f"\n{volatility_prompt}\n"

    if use_backtest_ml:
        try:
            max_horse_number = None
            if st.session_state.shutuba_table is not None and not st.session_state.shutuba_table.empty:
                max_horse_number = len(st.session_state.shutuba_table)
            ml_result = train_backtest_trend_model(race_id, max_horse_number=max_horse_number)
            if ml_result.context_text:
                data_context += "\n" + ml_result.context_text + "\n"
                with st.expander("バックテストML傾向", expanded=False):
                    st.text(ml_result.context_text)
                    if not ml_result.candidates.empty:
                        st.dataframe(
                            ml_result.candidates[["ticket_type", "combo", "trend_score", "avg_return_rate", "samples", "hit_races"]].head(20),
                            use_container_width=True
                        )
        except Exception as e:
            st.warning(f"バックテストML傾向を作成できませんでした: {e}")

    data_context += f"""

【出走馬詳細】"""

    if use_past_trend and st.session_state.past_trend_df is not None and not st.session_state.past_trend_df.empty:
        trend_df = st.session_state.past_trend_df.copy()
        trend_df["着順_num"] = pd.to_numeric(trend_df["着順"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        trend_df["人気_num"] = pd.to_numeric(trend_df["人気"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

        win_df = trend_df[trend_df["着順_num"] == 1]
        avg_win_pop = win_df["人気_num"].dropna().mean() if not win_df.empty else None
        common_umaban = trend_df["馬番"].value_counts().head(3).to_dict()

        trend_summary = []
        if avg_win_pop is not None:
            trend_summary.append(f"過去5年の勝ち馬人気平均: {avg_win_pop:.2f}")
        if common_umaban:
            trend_summary.append(f"馬番出現上位: {common_umaban}")

        if "past_trend_rows" in selected_features:
            data_context += "\n\n【同レース過去5年（1〜3着）】\n"
            data_context += trend_df[["年", "着順", "馬番", "騎手", "通過", "人気"]].to_csv(index=False, sep='|')
        if "past_trend_summary" in selected_features and trend_summary:
            data_context += "要約: " + " / ".join(trend_summary) + "\n"
        data_context += "この過去傾向を今回予想の根拠として優先的に評価してください。\n"

    # 取得した出馬表（騎手や斤量）のデータをAIプロンプトに追加
    if use_shutuba_table and st.session_state.shutuba_table is not None and not st.session_state.shutuba_table.empty:
        selected_shutuba_df = get_selected_shutuba_df(st.session_state.shutuba_table, selected_features)
        data_context += "\n[出馬表]\n"
        data_context += selected_shutuba_df.to_csv(index=False, sep='|') + "\n"

    for horse in st.session_state.all_horse_data:
        p = horse['pedigree']
        results_df = horse['results']
        
        horse_header = f"\n[{p['name']}]"
        blood_parts = []
        if "blood_sire" in selected_features:
            blood_parts.append(f"父:{p['sire']}")
        if "blood_dam" in selected_features:
            blood_parts.append(f"母:{p['dam']}")
        if "blood_broodmare_sire" in selected_features:
            blood_parts.append(f"母父:{p['broodmare_sire']}")
        if blood_parts:
            horse_header += " " + " ".join(blood_parts)
        data_context += horse_header + "\n"
        
        if results_df.empty:
            data_context += "データなし\n"
            continue

        if use_analysis:
            weight_info = ""
            if "analysis_latest_weight" in selected_features and '馬体重' in results_df.columns and results_df['馬体重'].notna().any():
                latest_weight_str = results_df['馬体重'].dropna().iloc[0]
                weight = re.match(r'(\d+)', str(latest_weight_str))
                if weight: weight_info = f" 体重:{weight.group(1)}"

            analysis_parts = []
            if "analysis_running_style" in selected_features:
                analysis_parts.append(f"脚質:{analyze_running_style(results_df)}")
            if "analysis_track_preference" in selected_features:
                analysis_parts.append(f"馬場:{analyze_track_preference(results_df)}")
            if weight_info:
                analysis_parts.append(weight_info.strip())
            if analysis_parts:
                data_context += " ".join(analysis_parts) + "\n"
        
        if use_past_results:
            existing_cols = get_selected_result_columns(results_df, selected_features)
            if not existing_cols:
                continue
            data_context += results_df[existing_cols].head(3).to_csv(index=False, sep='|') + "\n"
            
            # ③過去データを「数値のみ」に圧縮（単発予想時）
            short_df = results_df[existing_cols].copy()
            if '日付' in short_df.columns:
                short_df['日付'] = pd.to_datetime(short_df['日付'], errors='coerce').dt.strftime('%y/%m/%d')
            if 'レース名' in short_df.columns:
                short_df['レース名'] = short_df['レース名'].astype(str).str.replace('ステークス', 'S').str.replace('カップ', 'C').str[:6]
            if '着順' in short_df.columns:
                short_df['着順'] = short_df['着順'].astype(str).str.extract(r'(\d+)')[0].fillna(short_df['着順'])
            if '距離' in short_df.columns:
                short_df['距離'] = short_df['距離'].astype(str).str.replace('m', '', regex=False)
                
            data_context += short_df.head(5).to_csv(index=False, header=False, sep=',') + "\n"
    
    if st.button("多角分析を実行"):
        def run_perspectives(ai_name, ask_func, context):
            race_name = st.session_state.race_info.get("name", "") if st.session_state.race_info else ""
            is_classic = any(keyword in race_name for keyword in ["桜花賞", "皐月賞", "優駿牝馬", "オークス", "東京優駿", "日本ダービー", "菊花賞"])
            
            perspectives = []
            if use_blood:
                blood_focus = "「血統（父、母、母父の傾向や血統背景）」を最重視"
                if is_classic:
                    blood_focus += "（※本レースはクラシック競走のため、未知の距離への対応力を測るべく、特に母父や配合傾向などの血統背景を通常以上に深く考察してください）"
                perspectives.append(("🩸 血統重視", blood_focus))
            elif is_classic:
                st.info("血統情報が未選択のため、クラシック競走でも血統重視分析はスキップします。")
            perspectives.extend([
                ("📊 指数・データ重視", "「過去の戦績、着順、タイム、馬場適性、近走馬体重」を最重視"),
                ("🏇 展開重視", "「脚質、枠順、今回のメンバー構成（逃げ先行馬の数）」を最重視")
            ])
            
            results_dict = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (title, focus) in enumerate(perspectives):
                status_text.write(f"🔍 {ai_name} が {title} で予想中...")
                p = f"""あなたはプロの競馬予想家です。
以下の出走馬データとレース情報を元に、{focus}してレースを予想してください。

【出力要件】
1. この視点から見たレースの見解
2. 上位7頭の予想印（◎, ○, ▲, △, 注, 穴, 大穴など）とその明確な根拠

{context}
"""
                try:
                    res = ask_func(p)
                    results_dict[title] = res
                except Exception as e:
                    st.error(f"{title} 分析エラー: {e}")
                    results_dict[title] = "エラーのため取得できませんでした。"
                
                progress_bar.progress((idx + 1) / len(perspectives))
            
            status_text.write("✅ 分析完了！")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            return results_dict

        st.session_state.analysis_results = {}
        if ai_choice == "ルールベース (AI不使用)":
            rule_result = run_rule_based_analysis(
                st.session_state.all_horse_data, 
                st.session_state.shutuba_table,
                st.session_state.race_info,
                race_id,
                include_blood=use_blood
            )
            rule_result["_volatility"] = st.session_state.race_volatility
            st.session_state.analysis_results["ルールベース (AI不使用)"] = rule_result
        if ai_choice in ["Gemini", "両方で比較"]:
            st.session_state.analysis_results["Gemini"] = run_perspectives("Gemini", ask_gemini, data_context)
        if ai_choice in ["ChatGPT", "両方で比較"]:
            st.session_state.analysis_results["ChatGPT"] = run_perspectives("ChatGPT", ask_chatgpt, data_context)
        
        st.rerun()

    if st.session_state.analysis_results:
        st.markdown("### 🔍 多角分析結果")
        for ai_name, results_dict in st.session_state.analysis_results.items():
            st.markdown(f"#### {ai_name}")
            for title, res in results_dict.items():
                if title.startswith("_"):
                    continue
                with st.expander(f"👁️ {title} の予想結果", expanded=False):
                    st.write(res)

        st.markdown("---")
        st.markdown("#### 買い方設定")
        bet_plan_mode = st.radio(
            "買い方を選んでください",
            ["おまかせ", "カスタム"],
            key="bet_plan_mode",
            horizontal=True
        )

        if bet_plan_mode == "カスタム":
            st.caption("券種ごとの条件を追加できます。単勝・複勝以外は方式（通常／フォーメーション／ながし／ボックス）も指定できます。")
            remove_request_index = None
            for idx, req in enumerate(st.session_state.custom_ticket_requests):
                st.markdown(f"**条件 {idx + 1}**")
                cols = st.columns([1.5, 1.2, 1.0, 0.9, 1.0, 0.6])
                default_type = req.get("ticket_type", "ワイド")
                ticket_type = cols[0].selectbox(
                    "券種",
                    TICKET_TYPE_OPTIONS,
                    index=TICKET_TYPE_OPTIONS.index(default_type) if default_type in TICKET_TYPE_OPTIONS else 0,
                    key=f"ticket_type_{idx}"
                )
                default_method = req.get("betting_method", "通常")
                if ticket_type_supports_betting_method(ticket_type):
                    betting_method = cols[1].selectbox(
                        "方式",
                        BETTING_METHOD_OPTIONS,
                        index=BETTING_METHOD_OPTIONS.index(default_method) if default_method in BETTING_METHOD_OPTIONS else 0,
                        key=f"ticket_method_{idx}"
                    )
                else:
                    cols[1].markdown("方式指定なし")
                    betting_method = "通常"
                request_budget = cols[2].number_input(
                    "予算 (円)",
                    min_value=100,
                    step=100,
                    value=int(req.get("budget", 500) or 500),
                    key=f"ticket_budget_{idx}"
                )
                if ticket_type_requires_combo_count(ticket_type):
                    combo_count = cols[3].number_input(
                        "組数",
                        min_value=1,
                        step=1,
                        value=int(req.get("combo_count", 3) or 3),
                        key=f"ticket_combo_{idx}"
                    )
                else:
                    cols[3].markdown("組数指定なし")
                    combo_count = 0
                axis_horse = cols[4].text_input(
                    "軸馬(任意)",
                    value=req.get("axis_horse", ""),
                    key=f"ticket_axis_{idx}",
                    help="カンマ区切りで入力（例: 1,3）"
                )
                
                if cols[5].button("削除", key=f"remove_ticket_{idx}"):
                    remove_request_index = idx

                st.session_state.custom_ticket_requests[idx] = {
                    "ticket_type": ticket_type,
                    "budget": int(request_budget),
                    "combo_count": int(combo_count),
                    "betting_method": betting_method,
                    "axis_horse": axis_horse
                }

            action_cols = st.columns(2)
            if action_cols[0].button("条件を追加", key="add_ticket_request"):
                st.session_state.custom_ticket_requests.append(create_ticket_request())
                st.rerun()
            if action_cols[1].button("最後の条件を削除", key="remove_last_ticket_request"):
                if len(st.session_state.custom_ticket_requests) > 1:
                    st.session_state.custom_ticket_requests.pop()
                    st.rerun()

            if remove_request_index is not None:
                st.session_state.custom_ticket_requests.pop(remove_request_index)
                if not st.session_state.custom_ticket_requests:
                    st.session_state.custom_ticket_requests = [create_ticket_request()]
                st.rerun()

            custom_total_budget = sum(req.get("budget", 0) for req in st.session_state.custom_ticket_requests)
            st.info(f"カスタム条件の合計予算: {custom_total_budget}円")
            budget = custom_total_budget
        else:
            budget = st.number_input("予算 (円)", min_value=100, step=100, value=1000, key="overall_budget")
            st.caption("買い方希望がない場合はこちら。現状の予想モデルで券種と配分をおまかせ提案します。")

        ticket_plan_text, effective_budget = build_ticket_plan_text(
            bet_plan_mode,
            st.session_state.custom_ticket_requests,
            budget
        )

        if st.button("買い目を提案する"):
            if bet_plan_mode == "カスタム" and effective_budget <= 0:
                st.warning("カスタムの買い方条件を1件以上、予算ありで設定してください。")
                st.stop()

            race_name = st.session_state.race_info.get("name", "") if st.session_state.race_info else ""
            is_classic = any(keyword in race_name for keyword in ["桜花賞", "皐月賞", "優駿牝馬", "オークス", "東京優駿", "日本ダービー", "菊花賞"])
            classic_instruction = ""
            if is_classic and use_blood:
                classic_instruction = "※本レースは3歳クラシック競走のため、未知の距離・条件への対応力が重要になります。「血統重視の予想」の評価ウェイトを他の2つの視点の【2倍】にして、総合評価および買い目の構築を行ってください。\n\n"

            for ai_name, results_dict in st.session_state.analysis_results.items():
                if ai_name == "ルールベース (AI不使用)":
                    final_res = generate_rule_based_summary(results_dict, effective_budget, st.session_state.custom_ticket_requests, bet_plan_mode)
                    
                    st.markdown(f"### 🏆 ルールベース の最終結論（偏差値総合評価）")
                    st.write(final_res)

                    parsed_bets = extract_bets_from_text(final_res)
                    st.session_state.latest_predictions[ai_name] = {
                        "race_id": race_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "bets": parsed_bets,
                        "response": final_res
                    }
                    continue

                ask_func = ask_gemini if ai_name == "Gemini" else ask_chatgpt
                with st.spinner(f"{ai_name} が共通項を抽出し、最終結論を生成中..."):
                    perspective_sections = []
                    if use_blood:
                        perspective_sections.append(
                            "■ 血統重視の予想\n"
                            f"{results_dict.get('🩸 血統重視', '')}"
                        )
                    perspective_sections.extend([
                        "■ 指数・データ重視の予想\n"
                        f"{results_dict.get('📊 指数・データ重視', '')}",
                        "■ 展開重視の予想\n"
                        f"{results_dict.get('🏇 展開重視', '')}",
                    ])
                    perspective_count = len(perspective_sections)
                    perspective_text = "\n\n".join(perspective_sections)
                    summary_prompt = f"""あなたは総合競馬予想のスペシャリストです。
以下の{perspective_count}つの異なる視点からの予想結果を分析し、共通項を抽出して最終的な予想と買い目を出力してください。

{classic_instruction}【{perspective_count}つの視点からの予想結果】
{perspective_text}

【最終出力要件】
1. 分析の共通項（どの馬が複数の視点で高く評価されているか、その理由）
2. 最終的な総合予想印（◎, ○, ▲, △, 注, 消）と総合評価の根拠
3. レースの波乱度判定（「堅い」「標準」「荒れ気味」「大荒れ警戒」のいずれか）とその理由
4. 荒れ具合診断を踏まえた購入点数の判断（券種ごとの実際の点数も明記）
5. 予算{effective_budget}円の範囲での具体的な買い目（馬券種、組み合わせ（すべて馬番で書くこと）、金額配分）と、その買い方を選んだ理由
   - 「大荒れ警戒」または強い波乱想定の場合は、期待値のある穴馬または最上位評価馬の単勝を予算の20〜40%程度入れ、残りでワイド・馬連・3連複を組み立てること。
6. 買い目データ（※システムの自動集計用。必ず以下のJSONフォーマットでテキストの最後に記述すること）
```json
{{ "bets": [ {{"type": "馬連", "combo": "1-2", "amount": 500}} ] }}
```

【買い目構築ルール】
{ticket_plan_text}
"""
                try:
                    final_res = ask_func(summary_prompt)

                    # 予想結果ログ保存機能
                    log_file = "prediction_log.csv"
                    log_exists = os.path.exists(log_file)
                    log_data = [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        race_id,
                        ai_name,
                        effective_budget,
                        summary_prompt,
                        final_res
                    ]
                    with open(log_file, 'a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not log_exists:
                            writer.writerow(["Timestamp", "RaceID", "Model", "Budget", "Prompt", "Response"])
                        writer.writerow(log_data)

                    st.markdown(f"### 🏆 {ai_name} の最終結論（共通項抽出）")
                    st.write(final_res)

                    parsed_bets = extract_bets_from_text(final_res)
                    st.session_state.latest_predictions[ai_name] = {
                        "race_id": race_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "bets": parsed_bets,
                        "response": final_res
                    }
                except Exception as e:
                    st.error(f"最終結論 生成エラー: {e}")

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
                chat_context = "あなたは優秀な競馬予想アシスタントです。ユーザーの質問に答え、相談に乗りながら最適な買い目を検討してください。\n\n"
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
        st.caption("予想後にボタンを押すと、結果・払戻ページを参照して的中と回収金額を計算します。")
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

                    for model_name, pred in st.session_state.latest_predictions.items():
                        if pred.get("race_id") != race_id:
                            continue

                        bets = pred.get("bets", [])
                        if not bets:
                            st.info(f"{model_name}: 買い目JSONが見つからないため、金額集計をスキップしました。")
                            continue

                        total_bet, total_return, hits, ticket_stats = calculate_return(bets, payouts)
                        hit_rate = (hits / len(bets) * 100) if bets else 0
                        return_rate = (total_return / total_bet * 100) if total_bet > 0 else 0
                        st.markdown(f"#### {model_name} 的中結果")
                        st.write(f"購入合計: {total_bet}円 / 払戻合計: {total_return}円 / 回収率: {return_rate:.1f}% / 的中率: {hit_rate:.1f}%")
                        if ticket_stats:
                            ticket_df = pd.DataFrame.from_dict(ticket_stats, orient='index').reset_index().rename(
                                columns={"index": "券種", "bet": "購入額", "return": "払戻額", "hits": "的中数"}
                            )
                            ticket_df["回収率"] = (ticket_df["払戻額"] / ticket_df["購入額"] * 100).fillna(0).round(1).astype(str) + "%"
                            st.dataframe(ticket_df, use_container_width=True)

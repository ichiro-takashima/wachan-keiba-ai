import streamlit as st
import google.generativeai as genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import io
import os
import re
import json
import csv
import scraper
from datetime import datetime

# --- カスタムCSSの読み込み ---
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# --- 設定 ---
TARGET_MODEL = "gemini-2.5-flash"
# コマンドプロンプトで setx した値を取得
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
if API_KEY_GEMINI is None:
    st.sidebar.error("⚠️ APIキーが読み込めていません。再度 setx コマンドを実行し、PCを再起動してください。")
API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")
if API_KEY_OPENAI is None:
    st.sidebar.error("⚠️ APIキーが読み込めていません。再度 setx コマンドを実行し、PCを再起動してください。")

ai_choice = st.radio("使用するAIを選択", ["Gemini", "ChatGPT", "両方で比較"], horizontal=True)
# --- 分析用ヘルパー関数 ---
def ask_gemini(prompt):
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.GenerativeModel(TARGET_MODEL)
    response = model.generate_content(prompt)
    return response.text

def ask_chatgpt(prompt):
    client = OpenAI(api_key=API_KEY_OPENAI)
    response = client.chat.completions.create(
        model="gpt-4o", # または gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def _find_matching_column(df, candidates):
    """DataFrameから候補リストに一致する最初の列名を見つける"""
    if df is None:
        return None
        
    # 1. 完全一致
    for col in candidates:
        if col in df.columns:
            return col
            
    # 2. 空白・改行無視の一致
    for col in candidates:
        clean_col = col.replace(" ", "").replace("　", "").replace("\n", "").lower()
        for actual_col in df.columns:
            col_str = str(actual_col[-1] if isinstance(actual_col, tuple) else actual_col)
            clean_actual = col_str.replace(" ", "").replace("　", "").replace("\n", "").lower()
            if clean_col == clean_actual:
                return actual_col
                
    # 3. 部分一致
    for col in candidates:
        clean_col = col.replace(" ", "").replace("　", "").replace("\n", "").lower()
        for actual_col in df.columns:
            col_str = str(actual_col[-1] if isinstance(actual_col, tuple) else actual_col)
            clean_actual = col_str.replace(" ", "").replace("　", "").replace("\n", "").lower()
            if clean_col in clean_actual:
                return actual_col
    return None


def _get_available_columns(df, candidate_map):
    """
    候補名の中から実在する列だけを重複なく返す。
    「Unnamed」列であっても、中身が重要そうな場合は予備として含める。
    """
    selected_cols = []
    for candidates in candidate_map:
        col = _find_matching_column(df, candidates)
        if col and col not in selected_cols:
            selected_cols.append(col)
    
    # もし「通過」や「ペース」がまだ見つかっておらず、Unnamed列が存在する場合の救済処置
    # (scraper.py側の修正で解決するはずですが、二段構えにします)
    for col in df.columns:
        col_str = str(col)
        if "Unnamed" in col_str and col not in selected_cols:
            # 中身が空でない列は念のためAIに渡す候補に入れる
            if df[col].notna().any():
                selected_cols.append(col)
                
    return selected_cols

def analyze_running_style(results_df):
    """通過順位のデータから脚質を判定する"""
    if results_df is None or results_df.empty:
        return "不明"
    # 'コーナー'も検索候補に追加し、列名の揺れに対応
    corner_col = _find_matching_column(results_df, ['通過', 'コーナー通過順位', 'コーナー'])
    if not corner_col:
        # 列が見つからない場合にデバッグ情報をUIに表示
        st.warning(f"脚質判定に必要な「通過」列が見つかりませんでした。取得された列名を確認してください: {results_df.columns.tolist()}")
        return "不明"

    corner_pos_series = results_df[corner_col].dropna()
    # 直近5走のデータを使用
    last_5_races = corner_pos_series.head(5)
    if last_5_races.empty:
        return "不明"

    avg_positions = []
    for pos_str in last_5_races:
        # "1-2-2-3"などの文字列からすべての通過順位を抽出
        positions = [int(p) for p in re.findall(r'\d+', str(pos_str))]
        if positions:
            # そのレースにおける道中の平均順位を計算
            race_avg = sum(positions) / len(positions)
            avg_positions.append(race_avg)

    if not avg_positions:
        return "不明"

    avg_pos = sum(avg_positions) / len(avg_positions)

    if avg_pos <= 2.5:
        return "逃げ"
    elif avg_pos <= 5.5:
        return "先行"
    elif avg_pos <= 10.5:
        return "差し"
    else:
        return "追込"

def analyze_track_preference(results_df):
    """戦績から馬場適性を分析する"""
    if results_df is None or results_df.empty:
        return "データなし"

    track_col = _find_matching_column(results_df, ['馬場', '馬場状態'])
    rank_col = _find_matching_column(results_df, ['着順', '着'])

    if not track_col or not rank_col:
        return "データなし"

    df = results_df.copy()
    df['着順_num'] = df[rank_col].astype(str).str.extract(r'(\d+)')[0]
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')

    good_track = df[df[track_col] == '良']
    good_track_in_money = good_track[good_track['着順_num'] <= 3].shape[0]
    heavy_track = df[df[track_col].isin(['稍', '重', '不'])]
    heavy_track_in_money = heavy_track[heavy_track['着順_num'] <= 3].shape[0]

    if good_track.shape[0] == 0 and heavy_track.shape[0] == 0: return "経験なし"
    if good_track_in_money > heavy_track_in_money: return f"良馬場巧者({good_track_in_money}回)"
    elif heavy_track_in_money > good_track_in_money: return f"道悪巧者({heavy_track_in_money}回)"
    elif good_track_in_money > 0: return "馬場不問"
    else: return "傾向なし"

def normalize_ticket_type(t):
    return str(t).replace("三連", "3連").strip()

def normalize_combo(combo, ticket_type):
    normalized = str(combo).replace('ー', '-').replace(' ', '')
    if ticket_type in ["馬連", "ワイド", "三連複"]:
        return "-".join(sorted(normalized.split('-')))
    normalized = str(combo).replace('ー', '-').replace(' ', '').replace('→', '-').replace('>', '-')
    t_type = normalize_ticket_type(ticket_type)
    if t_type in ["馬連", "ワイド", "3連複"]:
        parts = [p for p in normalized.split('-') if p]
        try: return "-".join(sorted(parts, key=int))
        except: return "-".join(sorted(parts))
    return normalized


TICKET_TYPE_OPTIONS = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
COMBO_COUNT_REQUIRED_TYPES = {"枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"}
BETTING_METHOD_OPTIONS = ["通常", "フォーメーション", "ながし", "ボックス"]


def create_ticket_request():
    return {"ticket_type": "ワイド", "budget": 500, "combo_count": 3, "betting_method": "通常"}


def ticket_type_requires_combo_count(ticket_type):
    return ticket_type in COMBO_COUNT_REQUIRED_TYPES


def ticket_type_supports_betting_method(ticket_type):
    return ticket_type not in ["単勝", "複勝"]


def build_ticket_plan_text(plan_mode, ticket_requests, fallback_budget):
    if plan_mode != "カスタム":
        return (
            "波乱度の判定に基づき、必ず以下のルールで買い目を構築してください。\n"
            "・判定が「堅い」の場合: 上位2頭を軸にした3連複2頭軸流しを提案すること。\n"
            "・判定が「標準」の場合: 馬連4頭BOX（6点）と、それに対応する3連複フォーメーションを提案すること。\n"
            "・判定が「荒れる」の場合: 単勝2点と、その穴馬から上位人気へのワイド流しを提案すること。\n"
            "・金額配分は100円単位で、合計は予算以内に収めること。"
        ), fallback_budget

    valid_requests = []
    for req in ticket_requests:
        ticket_type = req.get("ticket_type")
        request_budget = int(req.get("budget", 0) or 0)
        combo_count = int(req.get("combo_count", 0) or 0)
        betting_method = req.get("betting_method", "通常")
        if not ticket_type or request_budget <= 0:
            continue
        if betting_method not in BETTING_METHOD_OPTIONS:
            betting_method = "通常"
        valid_requests.append({
            "ticket_type": ticket_type,
            "budget": request_budget,
            "combo_count": combo_count,
            "betting_method": betting_method
        })

    total_budget = sum(req["budget"] for req in valid_requests)
    lines = ["ユーザーの買い方希望に従って、以下の条件を満たす買い目を提案してください。"]
    for idx, req in enumerate(valid_requests, start=1):
        line = f"・条件{idx}: {req['ticket_type']}を{req['budget']}円分"
        if ticket_type_supports_betting_method(req["ticket_type"]):
            line += f"、方式は{req['betting_method']}"
        if ticket_type_requires_combo_count(req["ticket_type"]):
            line += f"、組数は{req['combo_count']}点ちょうど"
        lines.append(line)
    lines.extend([
        "・上記で指定された券種以外は提案しないこと。",
        "・各条件の予算を超えないこと。",
        "・券種ごとに指定された方式（通常／フォーメーション／ながし／ボックス）を守ること。",
        "・組数指定がある券種は、指定された組数ちょうどで買い目を構成すること。",
        "・金額配分は100円単位にすること。"
    ])
    return "\n".join(lines), total_budget


def format_ticket_plan_for_context(plan_mode, ticket_requests, fallback_budget):
    if plan_mode != "カスタム":
        return f"買い方希望: なし（おまかせ）。全体予算{fallback_budget}円で、現在の予想モデルに基づいて提案。"

    valid_requests = []
    for req in ticket_requests:
        ticket_type = req.get("ticket_type")
        request_budget = int(req.get("budget", 0) or 0)
        combo_count = int(req.get("combo_count", 0) or 0)
        betting_method = req.get("betting_method", "通常")
        if not ticket_type or request_budget <= 0:
            continue
        line = f"・{ticket_type}: {request_budget}円"
        if ticket_type_supports_betting_method(ticket_type):
            if betting_method not in BETTING_METHOD_OPTIONS:
                betting_method = "通常"
            line += f" / {betting_method}"
        if ticket_type_requires_combo_count(ticket_type):
            line += f" / {combo_count}点"
        valid_requests.append(line)

    total_budget = sum(int(req.get("budget", 0) or 0) for req in ticket_requests)
    return "買い方希望:\n" + "\n".join(valid_requests) + f"\n合計予算: {total_budget}円"


def _parse_ids_input(raw_text):
    if not raw_text:
        return []
    tokens = re.split(r"[\s,]+", raw_text.strip())
    return [token for token in tokens if token]


def calculate_return(bets, payouts):
    total_bet, total_return, hits = 0, 0, 0
    ticket_stats = {}
    # 払戻しのキー表記揺れを吸収 (三連複 -> 3連複)
    norm_payouts = {normalize_ticket_type(k): v for k, v in payouts.items()}

    for bet in bets:
        raw_type = bet.get("type", "")
        t_type = normalize_ticket_type(raw_type)
        bet_combo = str(bet.get("combo", ""))
        try:
            amount = int(bet.get("amount", 0))
        except Exception:
            continue

        if raw_type not in ticket_stats:
            ticket_stats[raw_type] = {"bet": 0, "return": 0, "hits": 0}
        ticket_stats[raw_type]["bet"] += amount
        total_bet += amount

        is_hit = False
        if t_type in norm_payouts:
            for payout in norm_payouts[t_type]:
                if normalize_combo(bet_combo, t_type) == normalize_combo(payout["combo"], t_type):
                    return_amount = int((amount / 100) * payout["pay"])
                    total_return += return_amount
                    ticket_stats[raw_type]["return"] += return_amount
                    is_hit = True
                    break

        if is_hit:
            hits += 1
            ticket_stats[raw_type]["hits"] += 1

    return total_bet, total_return, hits, ticket_stats


def extract_bets_from_text(ai_text):
    if not ai_text:
        return []
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', ai_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        payload = json.loads(match.group(1))
        bets = payload.get("bets", [])
        return bets if isinstance(bets, list) else []
    except Exception:
        return []


def build_reflection_text(predicted_bets, payouts):
    predicted_ticket_types = {bet.get("type", "") for bet in predicted_bets if bet.get("type")}
    payout_ticket_types = set(payouts.keys())
    missing_ticket_types = sorted(list(payout_ticket_types - predicted_ticket_types))

    hit_combos = set()
    predicted_combos_by_type = {}
    for bet in predicted_bets:
        ticket_type = bet.get("type", "")
        combo = bet.get("combo", "")
        if not ticket_type or not combo:
            continue
        predicted_combos_by_type.setdefault(ticket_type, set()).add(normalize_combo(combo, ticket_type))

    uncovered_examples = []
    for ticket_type, payout_list in payouts.items():
        predicted_set = predicted_combos_by_type.get(ticket_type, set())
        for payout in payout_list:
            payout_combo = normalize_combo(payout.get("combo", ""), ticket_type)
            if payout_combo in predicted_set:
                hit_combos.add(f"{ticket_type}:{payout_combo}")
            elif len(uncovered_examples) < 5:
                uncovered_examples.append(f"{ticket_type} {payout.get('combo', '')}")

    reflection_lines = []
    if missing_ticket_types:
        reflection_lines.append("・今回買っていないのに的中が出ていた券種: " + "、".join(missing_ticket_types))
    if uncovered_examples:
        reflection_lines.append("・買い目に入っていなかった主な的中組み合わせ: " + " / ".join(uncovered_examples))

    if not reflection_lines:
        reflection_lines.append("・券種と組み合わせの観点では、予想は結果に近い構成でした。")

    reflection_lines.append("・次回改善案: ◎○の軸は維持しつつ、ワイド/三連複の保険を1〜2点追加して取りこぼしを減らす。")
    return "\n".join(reflection_lines)

# --- Streamlit UI ---
st.sidebar.title("🏇 メニュー")
app_mode = st.sidebar.radio("モード選択", ["単一レース予想", "バックテスト"], index=0)

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

    def normalize_combo(c, b_type):
        c = str(c).replace('ー', '-').replace(' ', '')
        if b_type in ["馬連", "ワイド", "三連複"]:
            return "-".join(sorted(c.split('-')))
        c = str(c).replace('ー', '-').replace(' ', '').replace('→', '-').replace('>', '-')
        norm_type = str(b_type).replace("三連", "3連").strip()
        if norm_type in ["馬連", "ワイド", "3連複"]:
            parts = [p for p in c.split('-') if p]
            try: return "-".join(sorted(parts, key=int))
            except: return "-".join(sorted(parts))
        return c

    def calculate_return(bets, payouts):
        total_bet, total_ret, hits = 0, 0, 0
        tk_stats = {}
        norm_payouts = {str(k).replace("三連", "3連").strip(): v for k, v in payouts.items()}

        for bet in bets:
            b_type_orig = bet.get("type", "")
            b_type = str(b_type_orig).replace("三連", "3連").strip()
            b_combo = str(bet.get("combo", ""))
            try: b_amount = int(bet.get("amount", 0))
            except: continue
            
            if b_type_orig not in tk_stats: tk_stats[b_type_orig] = {"bet": 0, "return": 0, "hits": 0}
            tk_stats[b_type_orig]["bet"] += b_amount
            total_bet += b_amount
            
            is_hit = False
            if b_type in norm_payouts:
                for p in norm_payouts[b_type]:
                    if normalize_combo(b_combo, b_type) == normalize_combo(p["combo"], b_type):
                        ret_amt = int((b_amount / 100) * p["pay"])
                        total_ret += ret_amt
                        tk_stats[b_type_orig]["return"] += ret_amt
                        is_hit = True
                        break
            if is_hit:
                hits += 1
                tk_stats[b_type_orig]["hits"] += 1
        return total_bet, total_ret, hits, tk_stats

    st.title("🔄 AIバックテスト機能")
    st.markdown("過去のレースIDを入力し、AIの予想精度（的中率・回収率）を検証します。結果はCSVに保存されます。")
    bt_models = st.multiselect("検証するAIモデル", ["Gemini", "ChatGPT"], default=["Gemini"])
    
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
    col1, col2 = st.columns(2)
    with col1:
        bt_budget = st.number_input("1レースあたりの予算 (円)", value=1000)
    with col2:
        bt_max_horses = st.number_input("最大送信頭数（トークン節約用）", min_value=5, max_value=18, value=12, help="馬番の大きい外枠の馬から除外されます。全頭送るとトークンを消費します。")

    bt_user_opinion = st.text_area(
        "✍️ あなたの予想・注目馬（AIへの指示や意見があれば入力してください）",
        placeholder="例: 1番の馬の逃げ残りに期待。雨が降っているので外枠有利。",
        key="bt_user_opinion"
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
            cols = st.columns([1.6, 1.3, 1.1, 1.0, 0.8])
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
            if cols[4].button("削除", key=f"bt_remove_ticket_{idx}"):
                bt_remove_request_index = idx

            st.session_state.bt_custom_ticket_requests[idx] = {
                "ticket_type": ticket_type,
                "budget": int(request_budget),
                "combo_count": int(combo_count),
                "betting_method": betting_method
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
        if bt_custom_total_budget != bt_budget:
            st.caption(f"上の全体予算は {bt_budget}円ですが、カスタム条件の合計 {bt_custom_total_budget}円 を優先して買い目提案に使います。")

    bt_ticket_plan_text, bt_effective_budget = build_ticket_plan_text(
        bt_bet_plan_mode,
        st.session_state.get("bt_custom_ticket_requests", []),
        bt_budget
    )

    if st.button("バックテスト実行"):
        if bt_bet_plan_mode == "カスタム" and bt_effective_budget <= 0:
            st.warning("カスタムの買い方条件を1件以上、予算ありで設定してください。")
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
                        res_df = res_df[res_df['日付'] < pd.Timestamp(race_date)]
                    horses_data.append({"id": hid, "pedigree": ped, "results": res_df})
                
                ctx = f"--- \n【レース基本情報】\n・対象レースID: {r_id}\n・レース名: {race_info['name']}\n・コース・天候等: {race_info['data']}\n\n"
                if bt_user_opinion.strip():
                    ctx += f"【ユーザーからの特記事項・予想意見】\n{bt_user_opinion}\n※上記のユーザー意見を、今回の予想の重要な根拠の一つとして加味してください。\n\n"
                ctx += "【出走馬詳細】\n"

                shutuba_df = scraper.scrape_shutuba_table(r_id)
                if not shutuba_df.empty:
                    ctx += "[出馬表]\n" + shutuba_df.to_csv(index=False, sep='|') + "\n"
                    
                for horse in horses_data:
                    ped = horse['pedigree']
                    res_df = horse['results']
                    ctx += f"\n[{ped['name']}] 父:{ped['sire']} 母父:{ped['broodmare_sire']}\n"
                    if res_df.empty:
                        ctx += "データなし\n"
                        continue
                    
                    weight_info = ""
                    if '馬体重' in res_df.columns and res_df['馬体重'].notna().any():
                        latest_weight_str = res_df['馬体重'].dropna().iloc[0]
                        weight = re.match(r'(\d+)', str(latest_weight_str))
                        if weight: weight_info = f" 体重:{weight.group(1)}"

                    running_style = analyze_running_style(res_df)
                    track_preference = analyze_track_preference(res_df)
                    ctx += f"脚質:{running_style} 馬場:{track_preference}{weight_info}\n"
                    
                    existing_cols = _get_available_columns(res_df, [
                        ['日付'],
                        ['レース名', 'レース'],
                        ['着順', '着'],
                        ['距離'],
                        ['馬場'],
                        ['タイム'],
                        ['上り', '上がり', '上り3F', '上がり3F'],
                        ['ペース', 'ﾍﾟｰｽ'],
                        ['通過', 'コーナー通過順', 'コーナー'],
                        ['馬体重']
                    ])
                    
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

                prompt = f"""あなたはプロの競馬予想家です。出走馬データから総合的な予想を行い買い目を出力してください。予算:{bt_effective_budget}円。
【出力要件】
1. レース見解と予想印
2. 買い目（※必ず以下のJSONフォーマットでテキストの最後に記述すること）
```json
{{ "bets": [ {{"type": "馬連", "combo": "1-2", "amount": 500}} ] }}
```
【出走馬詳細】
{ctx}
"""
                for model in bt_models:
                    prog_text.write(f"{r_id} を {model} で予想中...")
                    ai_res, bets = "", []
                    try:
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

st.title("🏇 わーちゃんのレース予想AI")

race_id = st.text_input("Race IDを入力 (例: 202405020811)")
prediction_date = st.date_input(
    "予想の基準日（この日より前のデータのみ使用）", 
    value=pd.Timestamp.now(),
    min_value=pd.Timestamp("1980-01-01")
)
budget = st.number_input("予算 (円)", value=1000)

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
else:
    normalized_requests = []
    for req in st.session_state.custom_ticket_requests:
        normalized_req = create_ticket_request()
        normalized_req.update(req if isinstance(req, dict) else {})
        if normalized_req.get("betting_method") not in BETTING_METHOD_OPTIONS:
            normalized_req["betting_method"] = "通常"
        normalized_requests.append(normalized_req)
    st.session_state.custom_ticket_requests = normalized_requests or [create_ticket_request()]
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

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

                # データをセッションに保存して再起動
                st.session_state.all_horse_data = temp_horse_data
                st.session_state.pedigree_list = temp_pedigree_list
                st.session_state.shutuba_table = shutuba_df
                st.session_state.race_info = race_info
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
        st.subheader("📊 本レース出馬表（馬番・騎手・斤量など）")
        st.dataframe(st.session_state.shutuba_table, use_container_width=True)

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

    user_opinion = st.text_area(
        "✍️ あなたの予想・注目馬（AIへの指示や意見があれば入力してください）",
        placeholder="例: 1番の馬の逃げ残りに期待。雨が降っているので外枠有利。"
    )

    r_info = st.session_state.race_info or {"name": "不明", "data": "不明"}
    data_context = f"""---
【レース基本情報】
・対象レースID: {race_id}
・レース名: {r_info['name']}
・コース・天候等: {r_info['data']}
"""

    if user_opinion.strip():
        data_context += f"\n【ユーザーからの特記事項・予想意見】\n{user_opinion}\n※上記のユーザー意見を、今回の予想の重要な根拠の一つとして加味してください。\n"

    data_context += f"""

【出走馬詳細】"""

    # 取得した出馬表（騎手や斤量）のデータをAIプロンプトに追加
    if st.session_state.shutuba_table is not None and not st.session_state.shutuba_table.empty:
        data_context += "\n[出馬表]\n"
        data_context += st.session_state.shutuba_table.to_csv(index=False, sep='|') + "\n"

    for horse in st.session_state.all_horse_data:
        p = horse['pedigree']
        results_df = horse['results']
        data_context += f"\n[{p['name']}] 父:{p['sire']} 母父:{p['broodmare_sire']}\n"
        
        if results_df.empty:
            data_context += "データなし\n"
            continue

        weight_info = ""
        if '馬体重' in results_df.columns and results_df['馬体重'].notna().any():
            latest_weight_str = results_df['馬体重'].dropna().iloc[0]
            weight = re.match(r'(\d+)', str(latest_weight_str))
            if weight: weight_info = f" 体重:{weight.group(1)}"

        running_style = analyze_running_style(results_df)
        track_preference = analyze_track_preference(results_df)
        data_context += f"脚質:{running_style} 馬場:{track_preference}{weight_info}\n"
        
        existing_cols = _get_available_columns(results_df, [
            ['日付'],
            ['レース名', 'レース'],
            ['着順', '着'],
            ['距離'],
            ['馬場'],
            ['タイム'],
            ['上り', '上がり', '上り3F', '上がり3F'],
            ['ペース', 'ﾍﾟｰｽ'],
            ['通過', 'コーナー通過順', 'コーナー'],
            ['馬体重']
        ])
        data_context += results_df[existing_cols].head(3).to_csv(index=False, sep='|')
        
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
            
        data_context += short_df.head(5).to_csv(index=False, header=False, sep=',')
    
    if st.button("AIで多角分析を実行"):
        def run_perspectives(ai_name, ask_func, context):
            perspectives = [
                ("🩸 血統重視", "「血統（父、母、母父の傾向や血統背景）」を最重視"),
                ("📊 指数・データ重視", "「過去の戦績、着順、タイム、馬場適性、近走馬体重」を最重視"),
                ("🏇 展開重視", "「脚質、枠順、今回のメンバー構成（逃げ先行馬の数）」を最重視")
            ]
            
            results_dict = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (title, focus) in enumerate(perspectives):
                status_text.write(f"🔍 {ai_name} が {title} で予想中...")
                p = f"""あなたはプロの競馬予想家です。
以下の出走馬データとレース情報を元に、{focus}してレースを予想してください。

【出力要件】
1. この視点から見たレースの見解
2. 上位5頭の予想印（◎, ○, ▲, △, ☆）とその明確な根拠

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
                cols = st.columns([1.6, 1.3, 1.1, 1.0, 0.8])
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
                if cols[4].button("削除", key=f"remove_ticket_{idx}"):
                    remove_request_index = idx

                st.session_state.custom_ticket_requests[idx] = {
                    "ticket_type": ticket_type,
                    "budget": int(request_budget),
                    "combo_count": int(combo_count),
                    "betting_method": betting_method
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
            if custom_total_budget != budget:
                st.caption(f"上の全体予算は {budget}円ですが、カスタム条件の合計 {custom_total_budget}円 を優先して買い目提案に使います。")
        else:
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

            for ai_name, results_dict in st.session_state.analysis_results.items():
                ask_func = ask_gemini if ai_name == "Gemini" else ask_chatgpt
                with st.spinner(f"{ai_name} が共通項を抽出し、最終結論を生成中..."):
                    summary_prompt = f"""あなたは総合競馬予想のスペシャリストです。
以下の3つの異なる視点からの予想結果を分析し、共通項を抽出して最終的な予想と買い目を出力してください。

【3つの視点からの予想結果】
■ 血統重視の予想
{results_dict.get('🩸 血統重視', '')}

■ 指数・データ重視の予想
{results_dict.get('📊 指数・データ重視', '')}

■ 展開重視の予想
{results_dict.get('🏇 展開重視', '')}

【最終出力要件】
1. 分析の共通項（どの馬が複数の視点で高く評価されているか、その理由）
2. 最終的な総合予想印（◎, ○, ▲, △, 注, 消）と総合評価の根拠
3. レースの波乱度判定（「堅い」「標準」「荒れる」のいずれか）とその理由
4. 予算{effective_budget}円の範囲での具体的な買い目（馬券種、組み合わせ（すべて馬番で書くこと）、金額配分）と、その買い方を選んだ理由
5. 買い目データ（※システムの自動集計用。必ず以下のJSONフォーマットでテキストの最後に記述すること）
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
                        reflection = build_reflection_text(bets, payouts)

                        st.markdown(f"#### {model_name} 的中結果")
                        st.write(f"購入合計: {total_bet}円 / 払戻合計: {total_return}円 / 回収率: {return_rate:.1f}% / 的中率: {hit_rate:.1f}%")
                        if ticket_stats:
                            ticket_df = pd.DataFrame.from_dict(ticket_stats, orient='index').reset_index().rename(
                                columns={"index": "券種", "bet": "購入額", "return": "払戻額", "hits": "的中数"}
                            )
                            ticket_df["回収率"] = (ticket_df["払戻額"] / ticket_df["購入額"] * 100).fillna(0).round(1).astype(str) + "%"
                            st.dataframe(ticket_df, use_container_width=True)

                        st.markdown("**次回予想に活かす振り返り**")
                        st.write(reflection)
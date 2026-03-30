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

def analyze_running_style(results_df):
    """通過順位のデータから脚質を判定する"""
    if results_df is None or results_df.empty:
        return "不明"
    corner_col = _find_matching_column(results_df, ['通過', 'コーナー通過順位'])
    if not corner_col:
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

def calculate_base_score(results_df, pace_prediction):
    """脚質、過去実績、近走着順から基礎スコアと関連情報を算出する"""
    default_return = {
        "score": 0.0,
        "detail": "0点 (データなし)",
        "running_style": "不明",
        "track_preference": "データなし"
    }
    if results_df is None or results_df.empty:
        return default_return

    try:
        style = analyze_running_style(results_df)
        track_pref = analyze_track_preference(results_df)

        score = 0
        details = []

        # 1. 近走着順スコア (Max 50点)
        recent_score = 0
        rank_col = _find_matching_column(results_df, ['着順', '着'])
        if rank_col:
            recent_5 = results_df[rank_col].dropna().head(5)
            for rank in recent_5:
                match = re.search(r'(\d+)', str(rank))
                if match:
                    r = int(match.group(1))
                    if r == 1: recent_score += 10
                    elif r == 2: recent_score += 8
                    elif r == 3: recent_score += 6
                    elif r == 4: recent_score += 4
                    elif r == 5: recent_score += 2
        recent_score = min(recent_score, 50)
        score += recent_score
        details.append(f"近走:{recent_score}")

        # 2. 通算実績スコア (Max 20点)
        track_score = 0
        if rank_col:
            in_money = 0
            for rank in results_df[rank_col]:
                match = re.search(r'(\d+)', str(rank))
                if match and int(match.group(1)) <= 3:
                    in_money += 1
            track_score = min(in_money * 5, 20)
        score += track_score
        details.append(f"実績:{track_score}")

        # 3. 展開マッチスコア (Max 30点)
        pace_score = 0
        if style != "不明":
            if "ハイペース" in pace_prediction:
                pace_score = 30 if style in ["差し", "追込"] else 10
            elif "スローペース" in pace_prediction:
                pace_score = 30 if style in ["逃げ", "先行"] else 10
            else: # ミドルペース
                pace_score = 20
        score += pace_score
        details.append(f"展開:{pace_score}")

        return {
            "score": float(score),
            "detail": f"{score}点 ({', '.join(details)})",
            "running_style": style,
            "track_preference": track_pref
        }
    except Exception as e:
        st.warning(f"基礎スコアの計算中にエラーが発生しました: {e}")
        return default_return

# --- Streamlit UI ---
st.sidebar.title("🏇 メニュー")
app_mode = st.sidebar.radio("モード選択", ["単一レース予想", "バックテスト"], index=0)

if app_mode == "バックテスト":
    @st.cache_data(ttl=3600, show_spinner=False)
    def scrape_payouts(race_id): return scraper.scrape_payouts(race_id)

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_race_date(race_id): return scraper.get_race_date(race_id)

    def normalize_combo(c, b_type):
        c = str(c).replace('ー', '-').replace(' ', '')
        if b_type in ["馬連", "ワイド", "三連複"]:
            return "-".join(sorted(c.split('-')))
        return c

    def calculate_return(bets, payouts):
        total_bet, total_ret, hits = 0, 0, 0
        tk_stats = {}
        for bet in bets:
            b_type = bet.get("type", "")
            b_combo = str(bet.get("combo", ""))
            try: b_amount = int(bet.get("amount", 0))
            except: continue
            
            if b_type not in tk_stats: tk_stats[b_type] = {"bet": 0, "return": 0, "hits": 0}
            tk_stats[b_type]["bet"] += b_amount
            total_bet += b_amount
            
            is_hit = False
            if b_type in payouts:
                for p in payouts[b_type]:
                    if normalize_combo(b_combo, b_type) == normalize_combo(p["combo"], b_type):
                        ret_amt = int((b_amount / 100) * p["pay"])
                        total_ret += ret_amt
                        tk_stats[b_type]["return"] += ret_amt
                        is_hit = True
                        break
            if is_hit:
                hits += 1
                tk_stats[b_type]["hits"] += 1
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

    if st.button("バックテスト実行"):
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
                h_ids = scraper.get_horse_ids_from_race(r_id)
                if not h_ids: continue
                
                h_ids = h_ids[:bt_max_horses] # トークン節約のため頭数を絞る
                    
                ctx = f"【レースID】{r_id} 【レース日】{race_date}\n"
                shutuba_df = scraper.scrape_shutuba_table(r_id)
                if not shutuba_df.empty: ctx += "[出馬表]\n" + shutuba_df.to_csv(index=False, sep='|') + "\n"
                    
                for hid in h_ids:
                    ped = scraper.scrape_horse_ped(hid)
                    res_df = scraper.scrape_race_results_dedicated(hid)
                    if not res_df.empty:
                        res_df['日付'] = pd.to_datetime(res_df['日付'], errors='coerce')
                        res_df = res_df[res_df['日付'] < pd.Timestamp(race_date)]
                        res_df['日付'] = res_df['日付'].dt.strftime('%y/%m/%d') # 年を2桁に短縮
                    ctx += f"[{ped['name']}] 父:{ped['sire']} 母父:{ped['broodmare_sire']}\n"
                    if res_df.empty: ctx += "データなし\n"
                    if res_df.empty: ctx += "-\n"
                    else:
                        cols = [c for c in ['日付', 'レース名', '着順', '距離', '馬場', 'タイム', '通過'] if c in res_df.columns]
                        ctx += res_df[cols].head(3).to_csv(index=False, sep='|') + "\n"
                        # ③過去データを「数値のみ」に圧縮
                        short_df = res_df.copy()
                        if 'レース名' in short_df.columns:
                            short_df['レース名'] = short_df['レース名'].astype(str).str.replace('ステークス', 'S').str.replace('カップ', 'C').str[:6]
                        if '着順' in short_df.columns:
                            short_df['着順'] = short_df['着順'].astype(str).str.extract(r'(\d+)')[0].fillna(short_df['着順'])
                        if '距離' in short_df.columns:
                            short_df['距離'] = short_df['距離'].astype(str).str.replace('m', '', regex=False)
                            
                        cols = [c for c in ['日付', 'レース名', '着順', '距離', '馬場', 'タイム', '通過'] if c in short_df.columns]
                        # ヘッダーなし、カンマ区切りで究極まで圧縮
                        ctx += short_df[cols].head(5).to_csv(index=False, header=False, sep=',') + "\n"

                prompt = f"""あなたはプロの競馬予想家です。出走馬データから総合的な予想を行い買い目を出力してください。予算:{bt_budget}円。
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
    value=pd.Timestamp.now()
)
budget = st.number_input("予算 (円)", value=1000)

# --- データの保存場所を準備（コードの上のほう、ボタンより前に書いておく） ---
if "all_horse_data" not in st.session_state:
    st.session_state.all_horse_data = None
if "pedigree_list" not in st.session_state:
    st.session_state.pedigree_list = None
if "shutuba_table" not in st.session_state:
    st.session_state.shutuba_table = None

# --- AI予想開始ボタン ---
if st.button("AI予想を開始"):
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

                # データをセッションに保存して再起動
                st.session_state.all_horse_data = temp_horse_data
                st.session_state.pedigree_list = temp_pedigree_list
                st.session_state.shutuba_table = shutuba_df
                st.rerun()

# --- ここから表示フェーズ（データがあるときだけ自動で表示される） ---
if st.session_state.all_horse_data:
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

    # --- 展開予想ロジック（表示とAIプロンプトで共通使用） ---
    all_running_styles = []
    for horse in st.session_state.all_horse_data:
        if not horse['results'].empty:
            style = analyze_running_style(horse['results'])
            all_running_styles.append(style)

    num_front_runners = all_running_styles.count('逃げ')
    num_leaders = all_running_styles.count('先行')
    if num_front_runners >= 2 or (num_front_runners == 1 and num_leaders >= 3):
        race_pace_prediction = "ハイペース予想。複数の逃げ・先行馬が競り合い、前方の争いが激化しそうです。これにより、後半に脚を溜められる差し・追込馬に有利な展開となる可能性があります。"
    elif num_front_runners == 0 and num_leaders <= 2:
        race_pace_prediction = "スローペース予想。明確な逃げ馬がおらず、牽制しあって落ち着いた流れになりそうです。瞬発力や決め手のある馬が有利で、前残りの展開も考えられます。"
    else:
        race_pace_prediction = "ミドルペース予想。平均的なペース構成で、各馬の実力がストレートに反映されやすいでしょう。"

    st.info(f"🏁 **システム展開予想:** {race_pace_prediction}")

    # ② 基礎スコアの計算と補完
    for horse in st.session_state.all_horse_data:
        if "base_score" not in horse or not isinstance(horse.get("base_score"), dict):
            horse["base_score"] = calculate_base_score(horse['results'], race_pace_prediction)

    # ③ 基礎スコア一覧表の表示
    score_data = []
    for horse in st.session_state.all_horse_data:
        shutuba_info = {}
        if st.session_state.shutuba_table is not None and not st.session_state.shutuba_table.empty:
            horse_name = horse['pedigree']['name']
            entry = st.session_state.shutuba_table[st.session_state.shutuba_table['馬名'] == horse_name]
            if not entry.empty:
                shutuba_info = entry.iloc[0].to_dict()

        score_info = horse["base_score"]
        score_data.append({
            "馬番": shutuba_info.get("馬番", "-"),
            "馬名": horse['pedigree']['name'],
            "スコア": score_info['score'],
            "詳細": score_info['detail'],
            "脚質": score_info['running_style'],
            "馬場適性": score_info['track_preference']
        })

    if score_data:
        st.subheader("📊 全馬基礎スコア一覧")
        score_df = pd.DataFrame(score_data)
        score_df['スコア'] = pd.to_numeric(score_df['スコア'], errors='coerce').fillna(0)
        st.dataframe(score_df.sort_values("スコア", ascending=False), use_container_width=True)

    # ④ 各馬の詳細情報
    st.subheader("🐎 各馬の詳細情報と直近戦績")
    for horse in st.session_state.all_horse_data:
        score_info = horse["base_score"]
        with st.expander(f"{horse['pedigree']['name']} (ID: {horse['id']}) - 基礎スコア: {score_info['score']:.0f}点"):
            st.write(f"**父:** {horse['pedigree']['sire']} / **母:** {horse['pedigree']['dam']} / **母父:** {horse['pedigree']['broodmare_sire']}")
            st.write(f"**📊 ルールベース基礎スコア:** {score_info['detail']} | **脚質:** {score_info['running_style']} | **馬場適性:** {score_info['track_preference']}")
            if not horse['results'].empty:
                st.dataframe(horse['results'].head(5), use_container_width=True)
            else:
                st.write("戦績データがありません。")

    # ③ AIによる分析
    st.subheader(f"🤖 {ai_choice} によるレース分析")
    
    # ユーザーの意見を入力するボックスを追加
    user_opinion = st.text_area("✍️ あなたの予想・注目馬（AIへの指示や意見があれば入力してください）", placeholder="例: 1番の馬の逃げ残りに期待。雨が降っているので外枠有利。")

    # --- プロンプト用データコンテキスト生成 ---
    data_context = f"""---
【レース基本情報】
・対象レースID: {race_id}
"""

    if user_opinion.strip():
        data_context += f"\n【ユーザーからの特記事項・予想意見】\n{user_opinion}\n※上記のユーザー意見を、今回の予想の重要な根拠の一つとして加味してください。\n"

    data_context += f"""

【システム展開予想】
{race_pace_prediction}

【出走馬詳細】"""

    # 取得した出馬表（騎手や斤量）のデータをAIプロンプトに追加
    if st.session_state.shutuba_table is not None and not st.session_state.shutuba_table.empty:
        data_context += "\n[出馬表]\n"
        data_context += st.session_state.shutuba_table.to_csv(index=False, sep='|') + "\n"

    for horse in st.session_state.all_horse_data:
        p = horse['pedigree']
        results_df = horse['results']
        score_info = horse['base_score'] # 補完済みのスコア情報を利用
        data_context += f"\n[{p['name']}] 父:{p['sire']} 母父:{p['broodmare_sire']}\n"
        
        if results_df.empty:
            data_context += "データなし\n"
            continue

        weight_info = ""
        if '馬体重' in results_df.columns and results_df['馬体重'].notna().any():
            latest_weight_str = results_df['馬体重'].dropna().iloc[0]
            weight = re.match(r'(\d+)', str(latest_weight_str))
            if weight: weight_info = f" 体重:{weight.group(1)}"

        data_context += f"脚質:{score_info['running_style']} 馬場:{score_info['track_preference']}{weight_info} 基礎スコア:{score_info['detail']}\n"
        
        summary_cols = ['日付', 'レース名', '着順', '距離', '馬場', 'タイム', '上り', '通過']
        existing_cols = [col for col in summary_cols if col in results_df.columns]
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
    
    if st.button("AI多角分析を実行"):
        def run_multi_perspective(ai_name, ask_func, context):
            perspectives = [
                ("🩸 血統重視", "「血統（父、母、母父の傾向や血統背景）」を最重視"),
                ("📊 指数・データ重視", "「過去の戦績、着順、タイム、馬場適性、近走馬体重」を最重視"),
                ("🏇 展開重視", "「脚質、枠順、システムによる展開予想、今回のメンバー構成（逃げ先行馬の数）」を最重視")
            ]
            
            results_dict = {}
            st.markdown(f"#### 🔍 {ai_name} による多角分析プロセス")
            
            # 各視点からの予想を実行
            for title, focus in perspectives:
                with st.spinner(f"{ai_name} が {title} で予想中..."):
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
                        with st.expander(f"👁️ {title} の予想結果表示"):
                            st.write(res)
                    except Exception as e:
                        st.error(f"{title} 分析エラー: {e}")
                        results_dict[title] = "エラーのため取得できませんでした。"
                        
            # 共通項抽出と最終結論の生成
            with st.spinner(f"{ai_name} が共通項を抽出し、最終結論を生成中..."):
                summary_prompt = f"""あなたは総合競馬予想のスペシャリストです。
以下の3つの異なる視点からの予想結果を分析し、共通項を抽出して最終的な予想と買い目を出力してください。

【3つの視点からの予想結果】
■ 血統重視の予想
{results_dict['🩸 血統重視']}

■ 指数・データ重視の予想
{results_dict['📊 指数・データ重視']}

■ 展開重視の予想
{results_dict['🏇 展開重視']}

【最終出力要件】
1. 分析の共通項（どの馬が複数の視点で高く評価されているか、その理由）
2. 最終的な総合予想印（◎, ○, ▲, △, 注, 消）と総合評価の根拠
3. レースの波乱度判定（「堅い」「標準」「荒れる」のいずれか）とその理由
4. 予算{budget}円の範囲での具体的な買い目（馬券種、組み合わせ（すべて馬番で書くこと）、金額配分）と、その買い方を選んだ理由

【買い目構築ルール】
波乱度の判定に基づき、必ず以下のルールで買い目を構築してください。
・判定が「堅い」場合：上位2頭を軸にした三連複2頭軸流しを提案せよ。
・判定が「標準」場合：馬連4頭BOX（6点）と、それに対応する三連複フォーメーションを提案せよ。
・判定が「荒れる」場合：単勝2点と、その穴馬から上位人気へのワイド流しを提案せよ。
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
                        budget,
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
                except Exception as e:
                    st.error(f"最終結論 生成エラー: {e}")

        if ai_choice == "Gemini" or ai_choice == "両方で比較":
            run_multi_perspective("Gemini", ask_gemini, data_context)

        if ai_choice == "ChatGPT" or ai_choice == "両方で比較":
            run_multi_perspective("ChatGPT", ask_chatgpt, data_context)
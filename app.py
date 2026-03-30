import streamlit as st
import google.generativeai as genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
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

def analyze_running_style(corner_pos_series):
    """通過順位のデータから脚質を判定する"""
    if corner_pos_series.empty or corner_pos_series.isnull().all():
        return "不明"
    
    # 直近5走のデータを使用
    last_5_races = corner_pos_series.dropna().head(5)
    if last_5_races.empty:
        return "不明"

    avg_positions = []
    for pos_str in last_5_races:
        try:
            # '1-2-3-4' -> 最初のコーナー位置を脚質の指標として使う
            first_corner_pos = int(str(pos_str).split('-')[0])
            avg_positions.append(first_corner_pos)
        except (ValueError, IndexError):
            continue
            
    if not avg_positions:
        return "不明"
        
    # 平均コーナー位置
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
    if results_df.empty or '馬場' not in results_df.columns or '着順' not in results_df.columns:
        return "データなし"
    
    results_df['着順_num'] = pd.to_numeric(results_df['着順'], errors='coerce')
    good_track = results_df[results_df['馬場'] == '良']
    good_track_in_money = good_track[good_track['着順_num'] <= 3].shape[0]
    heavy_track = results_df[results_df['馬場'].isin(['稍', '重', '不'])]
    heavy_track_in_money = heavy_track[heavy_track['着順_num'] <= 3].shape[0]
    
    if good_track.shape[0] == 0 and heavy_track.shape[0] == 0: return "経験なし"
    if good_track_in_money > heavy_track_in_money: return f"良馬場巧者({good_track_in_money}回)"
    elif heavy_track_in_money > good_track_in_money: return f"道悪巧者({heavy_track_in_money}回)"
    elif good_track_in_money > 0: return "馬場不問"
    else: return "傾向なし"

def calculate_base_score(results_df, pace_prediction):
    """脚質、馬場適性、近走着順から100点満点の基礎スコアを算出する"""
    if results_df.empty:
        return 0, "データなし"
    
    score = 0
    details = []

    # 1. 近走着順スコア (Max 50点)
    recent_score = 0
    if '着順' in results_df.columns:
        recent_5 = results_df['着順'].head(5)
        for rank in recent_5:
            try:
                r = int(re.search(r'(\d+)', str(rank)).group(1))
                if r == 1: recent_score += 10
                elif r == 2: recent_score += 8
                elif r == 3: recent_score += 6
                elif r == 4: recent_score += 4
                elif r == 5: recent_score += 2
            except: pass
    recent_score = min(recent_score, 50)
    score += recent_score; details.append(f"近走:{recent_score}")

    # 2. 馬場実績スコア (Max 20点)
    track_score = 0
    if '着順' in results_df.columns:
        track_score = min(sum(1 for r in results_df['着順'] if re.search(r'(\d+)', str(r)) and int(re.search(r'(\d+)', str(r)).group(1)) <= 3) * 5, 20)
    score += track_score; details.append(f"馬場:{track_score}")

    # 3. 展開マッチスコア (Max 30点)
    style = analyze_running_style(results_df['通過']) if '通過' in results_df.columns else "不明"
    pace_score = 30 if ("ハイペース" in pace_prediction and style in ["差し", "追込"]) or ("スローペース" in pace_prediction and style in ["逃げ", "先行"]) else (20 if "ミドルペース" in pace_prediction and style in ["先行", "差し"] else 10)
    if style == "不明": pace_score = 0
    score += pace_score; details.append(f"展開:{pace_score}")

    return score, f"{score}点 ({', '.join(details)})"

def scrape_horse_ped(horse_id):

    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    try:

        time.sleep(1)

        response = requests.get(url, headers=headers, timeout=10)

        response.encoding = 'euc-jp'

        soup = BeautifulSoup(response.text, 'html.parser')



        # 馬名のクリーンアップ取得

        try:

            raw_title = soup.title.text

            horse_name = raw_title.split(" | ")[0].split(" (")[0].strip()

        except: horse_name = "不明"



        def get_clean_name(td_element):

            if not td_element: return "不明"

            a_tag = td_element.find('a')

            raw_text = a_tag.text if a_tag else td_element.text

            return raw_text.strip().split('\n')[0].strip()



        sire, dam, bms = "不明", "不明", "不明"

        blood_table = soup.find("table", class_="blood_table")

        if blood_table:

            parents_16 = blood_table.find_all("td", rowspan="16")

            grandparents_8 = blood_table.find_all("td", rowspan="8")

            if len(parents_16) >= 2:

                sire = get_clean_name(parents_16[0]); dam = get_clean_name(parents_16[1])

            if len(grandparents_8) >= 3:

                bms = get_clean_name(grandparents_8[2])



        return {"horse_id": horse_id, "name": horse_name, "sire": sire, "dam": dam, "broodmare_sire": bms}

    except Exception as e:

        return {"horse_id": horse_id, "name": "不明", "sire": "不明", "dam": "不明", "broodmare_sire": "不明"}

def scrape_race_results_dedicated(horse_id):
    # 💡 あなたが発見した「戦績専用URL」を使用！
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        time.sleep(1) # サーバーへの配慮（1秒待機）
        
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-jp'
        
        # HTML解析（馬名取得用）
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ① 馬名の取得
        # 専用ページはタイトルが「〇〇の競走成績 | 競走馬データ」となっているはずなので、そこから抜き出す
        try:
            horse_name = soup.title.text.split("の競走成績")[0].strip()
        except AttributeError:
            horse_name = "不明"

        # ② 戦績表の取得（Pandasの最強機能を使用）
        # 専用ページならHTMLが綺麗なので、pd.read_htmlが一発で表を認識してくれます！
        dfs = pd.read_html(io.StringIO(response.text))
        
        if len(dfs) > 0:
            # 通常、ページ内の最初の表(インデックス0)が戦績表です
            df = dfs[0]
            
            # 念のため、取得した表が本当に戦績表か（「日付」や「レース名」の列があるか）確認
            if '日付' in df.columns or 'レース名' in df.columns:
                # 誰の戦績かわかるように、表の左端に「馬ID」と「馬名」の列を追加
                df.insert(0, 'horse_id', horse_id)
                df.insert(1, 'horse_name', horse_name)
                return df
                
        print(f"  -> ⚠️ {horse_name} の戦績表が見つかりませんでした。")
        return pd.DataFrame() # 見つからなかった場合は空の表を返す

    except Exception as e:
        print(f"  -> ⚠️ {horse_id} の取得中にエラーが発生しました: {e}")
        return pd.DataFrame()

def get_horse_ids_from_race(race_id):
    """
    レースIDから出走馬のIDリストを自動取得する（出馬表・結果ページ両対応）
    """
    # 1. まずは過去のレース結果 (db.netkeiba.com) をチェック
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        time.sleep(1)
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # レース結果表から「馬番」と「馬ID」のペアを取得してソートする（カンニング防止）
        horse_list = []
        table = soup.find('table', class_='race_table_01')
        if table:
            for tr in table.find_all('tr')[1:]:  # ヘッダーをスキップ
                tds = tr.find_all('td')
                if len(tds) > 3:
                    try:
                        umaban = int(tds[2].text.strip())
                        a_tag = tds[3].find('a')
                        if a_tag and 'href' in a_tag.attrs:
                            match = re.search(r'/horse/(\d{10})', a_tag['href'])
                            if match:
                                horse_list.append((umaban, match.group(1)))
                    except:
                        pass
            if horse_list:
                horse_list.sort(key=lambda x: x[0]) # 馬番で昇順ソート
                return [h[1] for h in horse_list]
        
        # db.netkeiba で見つからない場合は、未来のレース (race.netkeiba.com) をチェック
        url_future = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        time.sleep(1)
        res_future = requests.get(url_future, headers=headers, timeout=10)
        res_future.encoding = 'euc-jp'
        soup_future = BeautifulSoup(res_future.text, 'html.parser')
        
        horse_ids = []
        for l in soup_future.find_all('a', href=True):
            match = re.search(r'/horse/(\d{10})', l['href'])
            if match:
                horse_ids.append(match.group(1))
        unique_ids = list(dict.fromkeys(horse_ids))
        return unique_ids[:18]
        
    except Exception as e:
        st.error(f"データの抽出中にエラーが発生しました: {e}")
        return []

def scrape_shutuba_table(race_id):
    """
    指定されたURLから出馬表データ（馬番、斤量、騎手、馬体重など）をスクレイピングする
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        time.sleep(1)
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-jp'
        
        # PandasでHTML内の表を一括取得
        dfs = pd.read_html(io.StringIO(response.text))
        
        for df in dfs:
            # 列名がマルチインデックス（複数行）になっている場合、一番下の行を列名にする
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(col[-1]) for col in df.columns]
            
            # 「馬名」列が含まれている表を出馬表とみなす
            if '馬名' in df.columns:
                # 空行などを削除
                df = df.dropna(subset=['馬名'])
                
                # 取得したい主要な列（サイトの表記揺れに対応するため部分一致で探す）
                target_cols = ['枠', '馬番', '馬名', '性齢', '斤量', '騎手', '厩舎', '馬体重']
                target_cols = ['枠', '馬番', '馬名', '性齢', '斤量', '騎手', '厩舎', '馬体重', '人気', 'オッズ']
                keep_cols = []
                for actual_col in df.columns:
                    for tc in target_cols:
                        if tc in actual_col.replace(" ", "") and actual_col not in keep_cols:
                            keep_cols.append(actual_col)
                            break
                if keep_cols:
                    return df[keep_cols]
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

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
    bt_race_ids_str = st.text_area("レースIDリスト（改行区切り）", "202405020811\n202305020811")
    bt_budget = st.number_input("1レースあたりの予算 (円)", value=1000)
    
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
                h_ids = get_horse_ids_from_race(r_id)
                if not h_ids: continue
                
                h_ids = h_ids[:bt_max_horses] # トークン節約のため頭数を絞る
                    
                ctx = f"【レースID】{r_id} 【レース日】{race_date}\n"
                shutuba_df = scrape_shutuba_table(r_id)
                if not shutuba_df.empty: ctx += "[出馬表]\n" + shutuba_df.to_csv(index=False, sep='|') + "\n"
                    
                for hid in h_ids:
                    ped = scrape_horse_ped(hid)
                    res_df = scrape_race_results_dedicated(hid)
                    if not res_df.empty:
                        res_df['日付'] = pd.to_datetime(res_df['日付'], errors='coerce')
                        res_df = res_df[res_df['日付'] < pd.Timestamp(race_date)]
                        res_df['日付'] = res_df['日付'].dt.strftime('%Y/%m/%d')
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
                        st.error(f"{model}エラー: {e}")
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
            h_ids = get_horse_ids_from_race(race_id)
            if not h_ids:
                st.error("出走馬が見つかりませんでした。")
            else:
                temp_horse_data = []
                temp_pedigree_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, hid in enumerate(h_ids):
                    status_text.write(f"🔍 {i+1}/{len(h_ids)}頭目 (ID: {hid}) を取得中...")
                    
                    ped_data = scrape_horse_ped(hid)
                    temp_pedigree_list.append(ped_data)
                    
                    results_df = scrape_race_results_dedicated(hid)
                    
                    if not results_df.empty:
                        results_df['日付'] = pd.to_datetime(results_df['日付'], errors='coerce')
                        cutoff_date = pd.Timestamp(prediction_date)
                        results_df = results_df[results_df['日付'] < cutoff_date]
                        results_df['日付'] = results_df['日付'].dt.strftime('%Y/%m/%d')
                    
                    temp_horse_data.append({"id": hid, "pedigree": ped_data, "results": results_df})
                    progress_bar.progress((i + 1) / len(h_ids))

                # 出馬表のスクレイピングを実行
                status_text.write("📊 出馬表（レース情報）を取得中...")
                shutuba_df = scrape_shutuba_table(race_id)

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
    st.subheader("� 出馬表（血統情報）")
    df_pedigree = pd.DataFrame(st.session_state.pedigree_list)
    df_pedigree = df_pedigree.rename(columns={
        "horse_id": "馬ID", "name": "馬名", "sire": "父", "dam": "母", "broodmare_sire": "母父"
    })
    st.dataframe(df_pedigree, use_container_width=True)

    # --- 展開予想ロジック（表示とAIプロンプトで共通使用） ---
    all_running_styles = []
    for horse in st.session_state.all_horse_data:
        if not horse['results'].empty and '通過' in horse['results'].columns:
            style = analyze_running_style(horse['results']['通過'])
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

    # ② 各馬の戦績と基礎スコア
    st.subheader("🐎 各馬の血統と直近戦績")
    for horse in st.session_state.all_horse_data:
        name = horse['pedigree']['name']
        results_df = horse['results']
        score, score_details = calculate_base_score(results_df, race_pace_prediction)
        with st.expander(f"{name} (ID: {horse['id']}) - 基礎スコア: {score}点"):
            st.write(f"**父:** {horse['pedigree']['sire']} / **母:** {horse['pedigree']['dam']} / **母父:** {horse['pedigree']['broodmare_sire']}")
            st.write(f"**📊 ルールベース基礎スコア:** {score_details}")
            if not results_df.empty:
                st.dataframe(results_df.head(5), use_container_width=True)
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
        data_context += f"\n[{p['name']}] 父:{p['sire']} 母父:{p['broodmare_sire']}\n"
        
        if results_df.empty:
            data_context += "データなし\n"
            continue

        running_style = analyze_running_style(results_df['通過'] if '通過' in results_df.columns else pd.Series())
        track_pref = analyze_track_preference(results_df)
        score, score_details = calculate_base_score(results_df, race_pace_prediction)
        
        weight_info = ""
        if '馬体重' in results_df.columns and results_df['馬体重'].notna().any():
            latest_weight_str = results_df['馬体重'].dropna().iloc[0]
            weight = re.match(r'(\d+)', str(latest_weight_str))
            if weight: weight_info = f" 体重:{weight.group(1)}"

        data_context += f"脚質:{running_style} 馬場:{track_pref}{weight_info} 基礎スコア:{score_details}\n"
        
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
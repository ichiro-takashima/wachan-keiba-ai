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

# 1頭分のデータをAI用の短い1行テキストにする関数（トークン節約）
def compact_for_ai(horse):
    p = horse['pedigree']
    if horse['results'].empty:
        return f"・{p['name']}({p['sire']}/{p['broodmare_sire']}): データなし"
        
    # 必要な列だけ抽出
    cols = ['日付', '着順', '距離']
    existing_cols = [c for c in cols if c in horse['results'].columns]
    res = horse['results'][existing_cols].head(3).copy()
    
    # 着順から数字だけ抽出（「1着」→「1」）
    if '着順' in res.columns:
        res['着順'] = res['着順'].astype(str).str.extract(r'(\d+)')[0].fillna(res['着順'])
        
    # データをギュッと凝縮（改行を消して | 区切りに）
    res_text = res.to_string(index=False, header=False).replace('\n', ' | ')
    res_text = re.sub(r'\s+', ' ', res_text) # 連続する空白を圧縮
    return f"・{p['name']}({p['sire']}/{p['broodmare_sire']}): {res_text}"

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
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    
    try:
        time.sleep(1)
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'euc-jp'
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 馬のページへのリンク（/horse/数字/）を抽出
        links = soup.find_all('a', href=True)
        horse_ids = []
        
        for l in links:
            match = re.search(r'/horse/(\d{10})', l['href'])
            if match:
                horse_ids.append(match.group(1))
        
        # 重複を削除しつつ、順番を維持
        unique_ids = list(dict.fromkeys(horse_ids))
        
        # db.netkeiba で見つからない場合は、未来のレース (race.netkeiba.com) をチェック
        if not unique_ids:
            url_future = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            time.sleep(1)
            res_future = requests.get(url_future, headers=headers, timeout=10)
            res_future.encoding = 'euc-jp'
            soup_future = BeautifulSoup(res_future.text, 'html.parser')
            
            for l in soup_future.find_all('a', href=True):
                match = re.search(r'/horse/(\d{10})', l['href'])
                if match:
                    horse_ids.append(match.group(1))
            unique_ids = list(dict.fromkeys(horse_ids))

        # 1レースは最大18頭なので、余計なリンク（掲示板など）を排除するため制限
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

    # ② 各馬の戦績
    st.subheader("🐎 各馬の血統と直近戦績")
    for horse in st.session_state.all_horse_data:
        name = horse['pedigree']['name']
        with st.expander(f"{name} (ID: {horse['id']})"):
            st.write(f"**父:** {horse['pedigree']['sire']} / **母:** {horse['pedigree']['dam']} / **母父:** {horse['pedigree']['broodmare_sire']}")
            if not horse['results'].empty:
                st.dataframe(horse['results'].head(5), use_container_width=True)
            else:
                st.write("戦績データがありません。")

    # ③ AIによる分析
    st.subheader(f"🤖 {ai_choice} によるレース分析")
    
    # --- トークン節約のためのデータ圧縮 ---
    ai_data_text = ""
    for horse in st.session_state.all_horse_data:
        ai_data_text += compact_for_ai(horse) + "\n"
        
    prompt = f"""あなたはプロの競馬予想家です。以下の制約を厳守し、簡潔に回答してください。

【制約】
・挨拶、前置き、結びの言葉は一切不要。
・各馬の評価は1行（50文字以内）にまとめる。
・的中率と回収率のバランスを重視した買い目を出す。

【レース情報】
レースID: {race_id} / 予算: {budget}円

【出走馬データ（馬名/父/母父/直近3走成績）】
{ai_data_text}

【出力フォーマット】
■有力馬評価
・[馬名]: 評価理由
（以下、注目馬のみ数頭）

■展開予想
（1行で記述）

■推奨買い目
・[券種]: [組み合わせ] (金額)
（予算内に収めること）
"""

    if st.button("AI予想を実行 (節約・高精度版)"):
        if ai_choice == "Gemini" or ai_choice == "両方で比較":
            with st.spinner("Geminiが分析中..."):
                try:
                    res = ask_gemini(prompt)
                    st.markdown("### 🟦 Geminiの予想")
                    st.write(res)
                except Exception as e:
                    st.error(f"Gemini分析エラー: {e}")

        if ai_choice == "ChatGPT" or ai_choice == "両方で比較":
            with st.spinner("ChatGPTが分析中..."):
                try:
                    res = ask_chatgpt(prompt)
                    st.markdown("### 🟩 ChatGPTの予想")
                    st.write(res)
                except Exception as e:
                    st.error(f"ChatGPT分析エラー: {e}")
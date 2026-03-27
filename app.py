import streamlit as st
import google.generativeai as genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import io
import os
import re

# --- 設定 ---
TARGET_MODEL = "gemini-2.5-flash"
# コマンドプロンプトで setx した値を取得
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY is None:
    st.sidebar.error("⚠️ APIキーが読み込めていません。再度 setx コマンドを実行し、PCを再起動してください。")

# --- 分析用ヘルパー関数 ---
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

# --- AI予想開始ボタン ---
if st.button("AI予想を開始"):
    if not API_KEY:
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

                # データをセッションに保存して再起動
                st.session_state.all_horse_data = temp_horse_data
                st.session_state.pedigree_list = temp_pedigree_list
                st.rerun()

# --- ここから表示フェーズ（データがあるときだけ自動で表示される） ---
if st.session_state.all_horse_data:
    # ① 出馬表
    st.subheader("📋 出馬表（血統情報）")
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

    # ③ Geminiによる分析
    st.subheader("🤖 Geminiによるレース分析と買い目予想")
    with st.spinner("AIが分析中..."):
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(TARGET_MODEL)
        
        # --- プロンプト生成ロジック ---
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

        prompt = f"""あなたはプロの競馬予想家です。
以下の出走馬データとレース展開予想を元に、レースID: {race_id} の最終的な予想をしてください。

# レース展開予想
{race_pace_prediction}

# 出走馬 詳細分析"""

        for horse in st.session_state.all_horse_data:
            p = horse['pedigree']
            results_df = horse['results']
            prompt += f"\n---\n## 馬名: {p['name']} (父: {p['sire']}, 母父: {p['broodmare_sire']})\n"
            
            if results_df.empty:
                prompt += "戦績データがありません（新馬戦など）。\n"
                continue

            running_style = analyze_running_style(results_df['通過'] if '通過' in results_df.columns else pd.Series())
            track_pref = analyze_track_preference(results_df)
            prompt += f"- **脚質**: {running_style}\n- **馬場適性**: {track_pref}\n"

            if '馬体重' in results_df.columns and results_df['馬体重'].notna().any():
                latest_weight_str = results_df['馬体重'].dropna().iloc[0]
                weight = re.match(r'(\d+)', str(latest_weight_str))
                if weight: prompt += f"- **近走馬体重**: {weight.group(1)}kg前後\n"

            prompt += "- **直近戦績サマリー**:\n"
            summary_cols = ['日付', 'レース名', '着順', '距離', '馬場', 'タイム', '上り', '通過']
            existing_cols = [col for col in summary_cols if col in results_df.columns]
            prompt += results_df[existing_cols].head(3).to_string(index=False) + "\n"

        prompt += f"""
# 最終的な指示
上記の詳細な分析とレース展開予想を踏まえ、予算{budget}円の範囲で、以下の形式で回答してください。
1. **レースの総括**: 展開予想を元に、どの脚質の馬が有利になるか簡潔に説明。
2. **有力馬3頭の評価**: 最も有力と考える馬を3頭挙げ、その理由を分析内容と関連付けて説明。
3. **買い目提案**: 予算内で、単勝(狙い4-10倍)・馬連(狙い10-30倍)・ワイド(狙い5-15倍)を各1点ずつ組み合わせた具体的な買い目と金額配分を提案。オッズはAIが推測すること。
"""
        
        try:
            response = model.generate_content(prompt)
            st.write(response.text)
        except Exception as e:
            st.error(f"分析エラー: {e}")
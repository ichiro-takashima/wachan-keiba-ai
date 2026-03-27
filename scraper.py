import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import io

# --- 1. 血統データを取得する関数 (完成版) ---
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

# --- 2. 戦績データを取得する関数 ---
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

# --- 2. メインの実行部分 ---
if __name__ == "__main__":
    # イクイノックスとリバティアイランドでテスト
    horse_ids = ["2019105219", "2020103656"]
    
    print("--- 戦績取得開始 ---")
    all_results = []
    
    for horse_id in horse_ids:
        print(f"馬ID: {horse_id} の戦績を取得中...")
        df_result = scrape_race_results_dedicated(horse_id)
        
        if not df_result.empty:
            all_results.append(df_result)
            print(f"  -> {df_result['horse_name'].iloc[0]} の戦績を {len(df_result)} 件取得しました！")

    # 取得したすべての馬の戦績を1つの大きな表（DataFrame）に結合
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        print("\n--- 取得結果（最初の5行を表示） ---")
        # ターミナルで横幅が省略されないように設定
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print(final_df.head())
        print("\n...")
        print("--- 取得結果（最後の5行を表示） ---")
        print(final_df.tail())
    else:
        print("戦績データが取得できませんでした。")

# --- 2. メインの実行部分 ---
if __name__ == "__main__":
    horse_ids = ["2019105219", "2020103656"]
    
    print("--- 戦績取得開始 ---")
    all_results = []
    
    for horse_id in horse_ids:
        print(f"馬ID: {horse_id} の戦績を取得中...")
        df_result = scrape_race_results_dedicated(horse_id)
        
        if not df_result.empty:
            all_results.append(df_result)
            print(f"  -> {df_result['horse_name'].iloc[0]} の戦績を {len(df_result)} 件取得しました！")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        print("\n--- 取得結果（最初の5行と最後の5行を表示） ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print(final_df.head())
        print("...")
        print(final_df.tail())
    else:
        print("戦績データが取得できませんでした。")
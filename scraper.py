import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

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
def scrape_race_results(horse_id):
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        time.sleep(1)
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-jp'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 戦績テーブルをクラス名で特定
        race_table = soup.find("table", class_="db_h_race_results")
        
        if race_table:
            # 【ここを修正】
            # テーブルのHTML文字列を直接渡し、明示的に最初のテーブル [0] を取得
            # flavor='bs4' を指定することで、BeautifulSoupの解析結果をそのまま利用します
            dfs = pd.read_html(str(race_table), flavor='bs4')
            if len(dfs) > 0:
                df = dfs[0]
                df.insert(0, 'horse_id', horse_id)
                return df
        
        print(f"  -> {horse_id} の戦績テーブルがHTML内に見つかりませんでした。")
        return None
    except Exception as e:
        print(f"  -> {horse_id} 戦績取得エラー: {e}")
        return None

# --- 3. メイン実行部分 ---
if __name__ == "__main__":
    horse_ids = ["2019105219", "2020103656", "2019104740", "2019105283"]
    
    ped_results = []
    race_results = []
    
    print("--- 取得開始 ---")
    for horse_id in horse_ids:
        # ① 血統を取得
        ped_data = scrape_horse_ped(horse_id)
        ped_results.append(ped_data)
        print(f"【{ped_data['name']}】の血統を取得完了")
        
        # ② 戦績を取得
        race_df = scrape_race_results(horse_id)
        if race_df is not None:
            race_results.append(race_df)
            print(f"  -> 戦績 {len(race_df)} レース分を取得")

    # --- 4. 結果の表示と保存 ---
    df_ped_final = pd.DataFrame(ped_results)
    df_race_final = pd.concat(race_results, ignore_index=True) if race_results else pd.DataFrame()

    print("\n--- 血統データ一覧 ---")
    print(df_ped_final.to_string(index=False))

    print("\n--- 戦績データ（最初の10行） ---")
    print(df_race_final.head(10).to_string())

    # 必要ならCSV保存もできます
    # df_ped_final.to_csv("horse_pedigrees.csv", index=False, encoding="utf_8_sig")
    # df_race_final.to_csv("horse_race_results.csv", index=False, encoding="utf_8_sig")
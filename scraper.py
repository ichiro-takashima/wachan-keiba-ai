import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def scrape_horse_data(horse_id):
    # 馬の個別ページURL
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    
    # サイトにアクセス（Bot弾きを避けるためUser-Agentを設定することが多いです）
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    # 文字コードをEUC-JPに設定（文字化け対策）
    response.encoding = 'euc-jp'
    
    # HTMLを解析
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # --- 1. 馬名の取得例 ---
    try:
        horse_name = soup.find("div", class_="horse_title").h1.text.strip()
        print(f"取得中: {horse_name}")
    except AttributeError:
        print("馬名が見つかりませんでした。")
        return None

    # --- 2. 血統データの取得（テーブルの解析が必要） ---
    # class="blood_table" などのテーブルを探すことになります。
    # blood_table = soup.find("table", class_="blood_table")
    # ここに血統のパース処理を書く
    
    # --- 3. 戦績データの取得 ---
    # 戦績のテーブルを探して取得する
    # past_races = soup.find("table", class_="db_h_race_results")
    # ここに戦績のパース処理を書く

    # サーバーへの負荷軽減（必須）
    time.sleep(2) 

    return {
        "name": horse_name,
        # "pedigree": ...,
        # "results": ...
    }

if __name__ == "__main__":
    # 例: イクイノックスの馬ID（2020102966）
    test_id = "2020102966"
    data = scrape_horse_data(test_id)
    print(data)
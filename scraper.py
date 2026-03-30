import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import io
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- HTTPセッション（リトライ設定付き） ---
session = requests.Session()
# サーバーエラー等で失敗した場合、最大3回リトライ（1秒, 2秒, 4秒と待機時間を増やす）
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.mount('http://', HTTPAdapter(max_retries=retries))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# --- 1. 血統データを取得する関数 (完成版) ---
def scrape_horse_ped(horse_id):
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
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

    try:
        time.sleep(1) # サーバーへの配慮（1秒待機）
        
        response = session.get(url, headers=HEADERS, timeout=10)
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
    """レースIDから出走馬のIDリストを自動取得する（出馬表・結果ページ両対応）"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        time.sleep(1)
        res = session.get(url, headers=HEADERS, timeout=10)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        horse_list = []
        table = soup.find('table', class_='race_table_01')
        if table:
            for tr in table.find_all('tr')[1:]:
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
                horse_list.sort(key=lambda x: x[0])
                return [h[1] for h in horse_list]
        
        url_future = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        time.sleep(1)
        res_future = session.get(url_future, headers=HEADERS, timeout=10)
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
        print(f"データの抽出中にエラーが発生しました: {e}")
        return []

def scrape_shutuba_table(race_id):
    """指定されたURLから出馬表データ（馬番、斤量、騎手など）を取得する"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = 'euc-jp'
        dfs = pd.read_html(io.StringIO(response.text))
        
        for df in dfs:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(col[-1]) for col in df.columns]
            if '馬名' in df.columns:
                df = df.dropna(subset=['馬名'])
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
    except Exception:
        return pd.DataFrame()

def scrape_payouts(race_id):
    """レースの払戻金データを取得する"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = 'euc-jp'
        soup = BeautifulSoup(response.text, 'html.parser')
        payouts = {}
        for table in soup.find_all('table', class_='pay_table_01'):
            for tr in table.find_all('tr'):
                th = tr.find('th')
                tds = tr.find_all('td')
                if th and len(tds) >= 2:
                    ticket_type = th.text.strip()
                    combos = list(tds[0].stripped_strings)
                    pays = list(tds[1].stripped_strings)
                    if ticket_type not in payouts:
                        payouts[ticket_type] = []
                    for c, p in zip(combos, pays):
                        pay_val = p.replace(',', '').replace('円', '')
                        if pay_val.isdigit():
                            payouts[ticket_type].append({"combo": c, "pay": int(pay_val)})
        return payouts
    except: return {}

def get_race_date(race_id):
    """レース開催日を取得する"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
        title = soup.title.text
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title)
        if match: return f"{match.group(1)}/{match.group(2).zfill(2)}/{match.group(3).zfill(2)}"
    except: pass
    return pd.Timestamp.now().strftime('%Y/%m/%d')
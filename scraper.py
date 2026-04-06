import io
import re
import time
import unicodedata
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _normalize_label(value):
    return (
        unicodedata.normalize("NFKC", str(value))
        .replace(" ", "")
        .replace("　", "")
        .replace("\n", "")
        .strip()
    )


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(col[-1]) for col in df.columns]
    return df


def _rename_columns(df, alias_map):
    rename_map = {}
    normalized_columns = {col: _normalize_label(col) for col in df.columns}
    for canonical, aliases in alias_map.items():
        normalized_aliases = {_normalize_label(alias) for alias in aliases}
        for actual_col, normalized_col in normalized_columns.items():
            if normalized_col in normalized_aliases:
                rename_map[actual_col] = canonical
                break
    return df.rename(columns=rename_map) if rename_map else df


def _infer_special_columns(df):
    """
    列名が不正確な場合、データの中身のパターンから
    「通過順位」と「ペース」の列を特定してリネームする
    """
    inference_patterns = {
        # 通過: "9-9-12-12" や "11-10-9-6" など
        "通過": r"^(\d{1,2}-)+\d{1,2}$",
        # ペース: "36.5-36.6" など
        "ペース": r"^\d{2}\.\d-\d{2}\.\d$"
    }
    
    for target_col, pattern in inference_patterns.items():
        for col in df.columns:
            sample_data = df[col].dropna().astype(str).head(10)
            if not sample_data.empty and any(re.match(pattern, value.strip()) for value in sample_data):
                # 重複回避：既に存在するターゲット列名がある場合はリネーム
                if target_col in df.columns and col != target_col:
                    df = df.rename(columns={target_col: f"old_{target_col}"})
                df = df.rename(columns={col: target_col})
                break
    return df


def _prepare_race_result_df(df):
    alias_map = {
        "日付": ["日付"],
        "開催": ["開催"],
        "天気": ["天気"],
        "R": ["R"],
        "レース名": ["レース名", "レース"],
        "映像": ["映像"],
        "頭数": ["頭数"],
        "枠番": ["枠番", "枠"],
        "馬番": ["馬番"],
        "オッズ": ["オッズ"],
        "人気": ["人気"],
        "着順": ["着順", "着"],
        "騎手": ["騎手"],
        "斤量": ["斤量"],
        "距離": ["距離"],
        "馬場": ["馬場"],
        "タイム": ["タイム"],
        "着差": ["着差"],
        "通過": ["通過", "コーナー通過順位", "コーナー"],
        "ペース": ["ペース", "ﾍﾟｰｽ"],
        "上り": ["上り", "上がり", "上り3F"],
        "馬体重": ["馬体重"],
        "勝ち馬(2着馬)": ["勝ち馬(2着馬)"],
        "賞金": ["賞金"],
    }
    df = _flatten_columns(df)
    df = _rename_columns(df, alias_map)
    df = _infer_special_columns(df)
    return df


def _looks_like_race_result_table(df):
    normalized_columns = {_normalize_label(col) for col in _flatten_columns(df).columns}
    return "日付" in normalized_columns or "レース名" in normalized_columns


def scrape_horse_ped(horse_id):
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = "euc-jp"
        soup = BeautifulSoup(response.text, "html.parser")

        try:
            raw_title = soup.title.text
            horse_name = raw_title.split(" | ")[0].split(" (")[0].strip()
        except Exception:
            horse_name = "不明"

        def get_clean_name(td_element):
            if not td_element:
                return "不明"
            a_tag = td_element.find("a")
            raw_text = a_tag.text if a_tag else td_element.text
            return raw_text.strip().split("\n")[0].strip()

        sire, dam, bms = "不明", "不明", "不明"
        blood_table = soup.find("table", class_="blood_table")
        if blood_table:
            parents_16 = blood_table.find_all("td", rowspan="16")
            grandparents_8 = blood_table.find_all("td", rowspan="8")
            if len(parents_16) >= 2:
                sire = get_clean_name(parents_16[0])
                dam = get_clean_name(parents_16[1])
            if len(grandparents_8) >= 3:
                bms = get_clean_name(grandparents_8[2])

        return {
            "horse_id": horse_id,
            "name": horse_name,
            "sire": sire,
            "dam": dam,
            "broodmare_sire": bms,
        }
    except Exception:
        return {"horse_id": horse_id, "name": "不明", "sire": "不明", "dam": "不明", "broodmare_sire": "不明"}


def scrape_race_results_dedicated(horse_id):
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = "euc-jp"
        soup = BeautifulSoup(response.text, "html.parser")

        dfs = pd.read_html(io.StringIO(response.text))
        result_df = None
        for candidate_df in dfs:
            if _looks_like_race_result_table(candidate_df):
                result_df = _prepare_race_result_df(candidate_df)
                break

        if result_df is None:
            return pd.DataFrame()

        result_df.insert(0, "horse_id", horse_id)
        return result_df
    except Exception:
        return pd.DataFrame()


def get_horse_ids_from_race(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        time.sleep(1)
        res = session.get(url, headers=HEADERS, timeout=10)
        res.encoding = "euc-jp"
        soup = BeautifulSoup(res.text, "html.parser")

        horse_list = []
        table = soup.find("table", class_="race_table_01")
        if table:
            for tr in table.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if len(tds) > 3:
                    try:
                        pop_text = tds[1].text.strip()
                        popularity = int(pop_text) if pop_text.isdigit() else 99
                        umaban = int(tds[2].text.strip())
                        a_tag = tds[3].find("a")
                        if a_tag and "href" in a_tag.attrs:
                            match = re.search(r"/horse/(\d{10})", a_tag["href"])
                            if match:
                                horse_list.append({"pop": popularity, "umaban": umaban, "id": match.group(1)})
                    except Exception: pass
            if horse_list:
                horse_list.sort(key=lambda x: x["pop"])
                return [h["id"] for h in horse_list]
        
        # データベースにない場合は出馬表から
        url_future = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        res_future = session.get(url_future, headers=HEADERS, timeout=10)
        res_future.encoding = "euc-jp"
        soup_future = BeautifulSoup(res_future.text, "html.parser")
        horse_ids = []
        for link in soup_future.find_all("a", href=True):
            match = re.search(r"/horse/(\d{10})", link["href"])
            if match: horse_ids.append(match.group(1))
        return list(dict.fromkeys(horse_ids))[:18]
    except Exception:
        return []


def scrape_shutuba_table(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        # JavaScriptがオッズと人気を描画するまで待機
        time.sleep(3)

        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")

        data = []
        table_rows = soup.find_all("tr", class_="HorseList")

        for row in table_rows:
            # --- 1. 馬番の取得 ---
            umaban_tag = row.find("td", class_=re.compile(r"Umaban\d"))
            umaban = umaban_tag.get_text(strip=True) if umaban_tag else ""
            
            if not umaban:
                continue

            # --- 2. 枠番 ---
            waku_tag = row.find("td", class_=re.compile(r"Waku\d"))
            waku = waku_tag.get_text(strip=True) if waku_tag else ""

            # --- 3. 馬名 ---
            horse_a = row.find("a", href=re.compile(r"/horse/"))
            horse_name = re.sub(r'\s+', '', horse_a.text) if horse_a else ""

            # --- 4. 性齢と斤量 ---
            seirei_td = row.find("td", class_="Barei")
            seirei = seirei_td.get_text(strip=True) if seirei_td else ""

            tds = row.find_all("td")
            weight = tds[5].get_text(strip=True) if len(tds) > 5 else ""

            # --- 5. 騎手 ---
            jockey_td = row.select_one(".Jockey")
            jockey = jockey_td.get_text(strip=True) if jockey_td else ""

            # --- 6. オッズと人気 ---
            # オッズの抽出
            odds_span = row.find("span", id=re.compile(r"odds[-_]"))
            if odds_span:
                odds = odds_span.get_text(strip=True)
            else:
                odds_td = row.find("td", class_=re.compile(r"Odds|txt_r", re.IGNORECASE))
                odds = odds_td.get_text(strip=True) if odds_td else ""

            # 人気の抽出
            ninki_span = row.find("span", id=re.compile(r"ninki[-_]"))
            if not ninki_span:
                ninki_span = row.find("span", class_=re.compile(r"Popularity|Popular|Ninki", re.IGNORECASE))
                
            if ninki_span:
                ninki = ninki_span.get_text(strip=True)
            else:
                ninki_td = row.find("td", class_=re.compile(r"Popularity|r3ml", re.IGNORECASE))
                ninki = ninki_td.get_text(strip=True) if ninki_td else ""

            data.append({
                "枠": waku,
                "馬番": umaban,
                "馬名": horse_name,
                "性齢": seirei,
                "斤量": weight,
                "騎手": jockey,
                "オッズ": odds,
                "人気": ninki
            })

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    finally:
        driver.quit()


def scrape_payouts(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = "euc-jp"
        soup = BeautifulSoup(response.text, "html.parser")
        payouts = {}
        for table in soup.find_all("table", class_="pay_table_01"):
            for tr in table.find_all("tr"):
                th = tr.find("th")
                tds = tr.find_all("td")
                if th and len(tds) >= 2:
                    ticket_type = th.text.strip()
                    combos = list(tds[0].stripped_strings)
                    pays = list(tds[1].stripped_strings)
                    if ticket_type not in payouts: payouts[ticket_type] = []
                    for combo, pay in zip(combos, pays):
                        pay_val = pay.replace(",", "").replace("円", "")
                        if pay_val.isdigit():
                            payouts[ticket_type].append({"combo": combo, "pay": int(pay_val)})
        return payouts
    except Exception: return {}


def scrape_race_result_page(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = "euc-jp"

        payouts = {}
        soup = BeautifulSoup(response.text, "html.parser")
        for table in soup.find_all("table", class_="pay_table_01"):
            for tr in table.find_all("tr"):
                th = tr.find("th")
                tds = tr.find_all("td")
                if not th or len(tds) < 2:
                    continue
                ticket_type = th.text.strip()
                combos = list(tds[0].stripped_strings)
                pays = list(tds[1].stripped_strings)
                if ticket_type not in payouts:
                    payouts[ticket_type] = []
                for combo, pay in zip(combos, pays):
                    pay_val = pay.replace(",", "").replace("円", "")
                    if pay_val.isdigit():
                        payouts[ticket_type].append({"combo": combo, "pay": int(pay_val)})

        return {"ok": True, "url": url, "payouts": payouts}
    except Exception:
        return {"ok": False, "url": url, "payouts": {}}


def scrape_race_info(race_id):
    # 1. 現在のレースページ(race.netkeiba.com)から取得を試みる
    url_shutuba = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        res = session.get(url_shutuba, headers=HEADERS, timeout=10)
        res.encoding = "euc-jp"
        soup = BeautifulSoup(res.text, "html.parser")
        
        name_tag = soup.find("div", class_="RaceName")
        if name_tag and name_tag.text.strip():
            race_name = name_tag.text.strip()
            race_data = ""
            data_tag = soup.find("div", class_="RaceData01")
            if data_tag:
                race_data = data_tag.text.replace("\n", " ").strip()
                race_data = re.sub(r'\s+', ' ', race_data)
            return {"name": race_name, "data": race_data}
    except Exception:
        pass
        
    # 2. 過去のレースの場合はデータベース(db.netkeiba.com)から取得する
    url_db = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        res = session.get(url_db, headers=HEADERS, timeout=10)
        res.encoding = "euc-jp"
        soup = BeautifulSoup(res.text, "html.parser")
        
        dl = soup.find("dl", class_="racedata")
        if dl:
            h1 = dl.find("h1")
            race_name = h1.text.strip() if h1 else ""
            p = dl.find("diary_snap_cut") or dl.find("p")
            race_data = p.text.replace("\xa0", " ").strip() if p else ""
            race_data = re.sub(r'\s+', ' ', race_data)
            return {"name": race_name, "data": race_data}
    except Exception:
        pass

    return {"name": f"レースID: {race_id}", "data": "情報取得不可"}

def get_race_date(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        res.encoding = "euc-jp"
        soup = BeautifulSoup(res.text, "html.parser")
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", soup.title.text)
        if match: return f"{match.group(1)}/{match.group(2).zfill(2)}/{match.group(3).zfill(2)}"
    except Exception: pass
    return pd.Timestamp.now().strftime("%Y/%m/%d")
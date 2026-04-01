import io
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
    inference_patterns = {
        "通過": r"^[\d\*]+(?:-[\d\*]+)+$",
        "ペース": r"^\d{2}\.\d(?:-\d{2}\.\d){1,2}$",
    }
    for target_col, pattern in inference_patterns.items():
        if target_col in df.columns:
            continue
        for col in df.columns:
            sample_data = df[col].dropna().astype(str).head(10)
            if not sample_data.empty and any(re.match(pattern, value.strip()) for value in sample_data):
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
        "馬場指数": ["馬場指数"],
        "タイム": ["タイム"],
        "着差": ["着差"],
        "通過": ["通過", "コーナー通過", "コーナー通過順", "コーナー"],
        "ペース": ["ペース", "ﾍﾟｰｽ"],
        "上り": ["上り", "上がり", "上り3F", "上がり3F"],
        "馬体重": ["馬体重", "馬体重(増減)"],
        "勝ち馬(2着馬)": ["勝ち馬(2着馬)", "勝ち馬"],
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
        return {
            "horse_id": horse_id,
            "name": "不明",
            "sire": "不明",
            "dam": "不明",
            "broodmare_sire": "不明",
        }


def scrape_race_results_dedicated(horse_id):
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"

    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = "euc-jp"
        soup = BeautifulSoup(response.text, "html.parser")

        try:
            raw_title = soup.title.text
            horse_name = raw_title.split("の競走成績")[0].split(" | ")[0].strip()
        except Exception:
            horse_name = "不明"

        dfs = pd.read_html(io.StringIO(response.text))
        result_df = None
        for candidate_df in dfs:
            if _looks_like_race_result_table(candidate_df):
                result_df = _prepare_race_result_df(candidate_df)
                break

        if result_df is None:
            print(f"  -> {horse_name} の戦績表が見つかりませんでした。")
            return pd.DataFrame()

        result_df.insert(0, "horse_id", horse_id)
        result_df.insert(1, "horse_name", horse_name)
        return result_df

    except Exception as e:
        print(f"  -> {horse_id} の取得中にエラーが発生しました: {e}")
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
                                horse_list.append(
                                    {"pop": popularity, "umaban": umaban, "id": match.group(1)}
                                )
                    except Exception:
                        pass
            if horse_list:
                horse_list.sort(key=lambda x: x["pop"])
                return [h["id"] for h in horse_list]

        url_future = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        time.sleep(1)
        res_future = session.get(url_future, headers=HEADERS, timeout=10)
        res_future.encoding = "euc-jp"
        soup_future = BeautifulSoup(res_future.text, "html.parser")

        horse_ids = []
        for link in soup_future.find_all("a", href=True):
            match = re.search(r"/horse/(\d{10})", link["href"])
            if match:
                horse_ids.append(match.group(1))
        unique_ids = list(dict.fromkeys(horse_ids))
        return unique_ids[:18]

    except Exception as e:
        print(f"データの取得中にエラーが発生しました: {e}")
        return []


def scrape_shutuba_table(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu"
    try:
        time.sleep(1)
        response = session.get(url, headers=HEADERS, timeout=10)
        response.encoding = "euc-jp"
        dfs = pd.read_html(io.StringIO(response.text))

        alias_map = {
            "枠": ["枠", "枠番"],
            "馬番": ["馬番"],
            "馬名": ["馬名"],
            "性齢": ["性齢"],
            "斤量": ["斤量"],
            "騎手": ["騎手"],
            "厩舎": ["厩舎"],
            "馬体重": ["馬体重"],
            "人気": ["人気"],
            "オッズ": ["オッズ"],
        }

        for df in dfs:
            df = _flatten_columns(df)
            df = _rename_columns(df, alias_map)
            if "馬名" in df.columns:
                df = df.dropna(subset=["馬名"])
                target_cols = ["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "馬体重", "人気", "オッズ"]
                keep_cols = [col for col in target_cols if col in df.columns]
                if keep_cols:
                    return df[keep_cols]
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
                    if ticket_type not in payouts:
                        payouts[ticket_type] = []
                    for combo, pay in zip(combos, pays):
                        pay_val = pay.replace(",", "").replace("円", "")
                        if pay_val.isdigit():
                            payouts[ticket_type].append({"combo": combo, "pay": int(pay_val)})
        return payouts
    except Exception:
        return {}


def get_race_date(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        res.encoding = "euc-jp"
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.text
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
        if match:
            return f"{match.group(1)}/{match.group(2).zfill(2)}/{match.group(3).zfill(2)}"
    except Exception:
        pass
    return pd.Timestamp.now().strftime("%Y/%m/%d")

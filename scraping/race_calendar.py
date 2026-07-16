"""指定期間の中央競馬レースIDを取得するカレンダー機能。

db.netkeiba.com のレース一覧ページ(1日単位)を巡回し、
その日に開催された race_id をすべて集める。
"""
import re
import time
from datetime import datetime, timedelta

from scraper import session, HEADERS, JRA_PLACE_CODES

RACE_LIST_URL = "https://db.netkeiba.com/race/list/{date_str}/"
RACE_ID_PATTERN = re.compile(r"race/(\d{12})")


def _is_jra_race_id(race_id):
    return len(race_id) >= 6 and race_id[4:6] in JRA_PLACE_CODES


def _to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "year") and not isinstance(value, str):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def get_race_id_list_for_date(target_date, sleep_sec=1.0):
    """指定日に開催されたレースIDの一覧を返す(開催がない日は空リスト)。"""
    date_str = target_date.strftime("%Y%m%d")
    url = RACE_LIST_URL.format(date_str=date_str)
    time.sleep(sleep_sec)
    response = session.get(url, headers=HEADERS, timeout=10)
    if response.status_code != 200:
        return []
    response.encoding = response.apparent_encoding or "euc-jp"
    all_ids = set(RACE_ID_PATTERN.findall(response.text))
    # db.netkeiba.com の日別一覧には地方競馬(NAR)のレースも混在するため、中央競馬(場id 01〜10)のみに絞る
    race_ids = sorted(rid for rid in all_ids if _is_jra_race_id(rid))
    return race_ids


def get_race_id_list(start_date, end_date, sleep_sec=1.0):
    """start_date〜end_date(両端含む)の中央競馬race_idを日付順に取得する。"""
    start = _to_date(start_date)
    end = _to_date(end_date)
    if start > end:
        start, end = end, start

    race_ids = []
    current = start
    while current <= end:
        race_ids.extend(get_race_id_list_for_date(current, sleep_sec=sleep_sec))
        current += timedelta(days=1)
    return race_ids

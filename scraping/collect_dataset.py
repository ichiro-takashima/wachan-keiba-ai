"""
最新N年分（既定5年）のJRAレースHTMLを netkeiba から取得し、
学習用(train)とテスト用(test)に分けて保存するスクリプト。

- 既に取得済みのレース（data/race_html/*.html など）は取得対象から除外する
- 分割は時系列で行う（直近 test_years 年をテスト、それより前を学習）。
  ランダム分割だと未来のレースで学習して過去を当てるリークになるため。
- 途中で止めても、再実行すれば保存済みファイルをスキップして続きから取得する

使い方:
    python -m scraping.collect_dataset                 # 最新5年分を取得（直近1年をテスト）
    python -m scraping.collect_dataset --list-only     # レースID一覧(manifest)の作成のみ
    python -m scraping.collect_dataset --years 5 --test-years 1 --sleep 1.0

保存先:
    data/train/race_html/{race_id}.html
    data/test/race_html/{race_id}.html
    data/dataset_manifest.csv   (race_id, date, split, status)

保存先ディレクトリは feature_importance.run_feature_importance(html_dir=...) に
そのまま渡せる形式（race_html と同じ構成）。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DATA_DIR = Path("data")
EXISTING_HTML_DIRS = [DATA_DIR / "race_html"]
TRAIN_HTML_DIR = DATA_DIR / "train" / "race_html"
TEST_HTML_DIR = DATA_DIR / "test" / "race_html"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"
MANIFEST_COLUMNS = ["race_id", "date", "split", "status"]

# JRAの開催場コード（01:札幌 〜 10:小倉）
JRA_VENUES = {f"{i:02d}" for i in range(1, 11)}

RACE_ID_PATTERN = re.compile(r"(?:/race/|race_id=)(\d{12})")
KAISAI_DATE_PATTERN = re.compile(r"kaisai_date=(\d{8})")


session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))


def _get(url, sleep_sec):
    time.sleep(sleep_sec)
    response = session.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response


def _progress(iterable, total, desc):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def iter_months(start_date, end_date):
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield year, month
        month += 1
        if month > 12:
            year, month = year + 1, 1


def fetch_kaisai_dates(year, month, sleep_sec):
    """race.netkeiba のカレンダーから、その月の開催日を取得する。"""
    url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
    try:
        response = _get(url, sleep_sec)
    except requests.RequestException:
        return []
    dates = set()
    for value in KAISAI_DATE_PATTERN.findall(response.text):
        try:
            d = datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            continue
        if d.year == year and d.month == month:
            dates.add(d)
    return sorted(dates)


def fetch_race_ids_on_date(target_date, sleep_sec):
    """開催日のレースID一覧を取得する。db.netkeiba → race.netkeiba の順に試す。"""
    ymd = target_date.strftime("%Y%m%d")
    urls = [
        f"https://db.netkeiba.com/race/list/{ymd}/",
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={ymd}",
    ]
    for url in urls:
        try:
            response = _get(url, sleep_sec)
        except requests.RequestException:
            continue
        race_ids = {
            rid for rid in RACE_ID_PATTERN.findall(response.text)
            if rid[4:6] in JRA_VENUES and rid[:4] == ymd[:4]
        }
        if race_ids:
            return sorted(race_ids)
    return []


def collect_race_calendar(start_date, end_date, sleep_sec):
    """期間内の {race_id: 開催日} を返す。"""
    race_dates = {}
    months = list(iter_months(start_date, end_date))
    for year, month in _progress(months, len(months), "開催日一覧"):
        kaisai_dates = fetch_kaisai_dates(year, month, sleep_sec)
        if not kaisai_dates:
            # カレンダーが取れない月は土日のみ総当たりで補完する
            d = max(date(year, month, 1), start_date)
            while d.month == month and d <= end_date:
                if d.weekday() >= 5:
                    kaisai_dates.append(d)
                d += timedelta(days=1)
        for kaisai_date in kaisai_dates:
            if not (start_date <= kaisai_date <= end_date):
                continue
            for race_id in fetch_race_ids_on_date(kaisai_date, sleep_sec):
                race_dates.setdefault(race_id, kaisai_date)
    return race_dates


def existing_race_ids(dirs):
    ids = set()
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            ids.update(p.stem for p in d.glob("*.html"))
    return ids


def load_manifest(path=MANIFEST_PATH):
    if not Path(path).is_file():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["race_id"]: row for row in csv.DictReader(f)}


def save_manifest(rows, path=MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for race_id in sorted(rows, key=lambda r: (rows[r]["date"], r)):
            writer.writerow({col: rows[race_id].get(col, "") for col in MANIFEST_COLUMNS})


def _years_before(d, years):
    """d の years 年前の翌日（うるう日は2/28扱い）。"""
    try:
        return d.replace(year=d.year - years) + timedelta(days=1)
    except ValueError:
        return d.replace(year=d.year - years, day=28) + timedelta(days=1)


def build_manifest(end_date, years, test_years, exclude_ids, sleep_sec, refresh=False):
    """取得対象レースの一覧（split付き）を作る。既存manifestがあれば再利用する。"""
    start_date = _years_before(end_date, years)
    test_start = _years_before(end_date, test_years)

    manifest = {} if refresh else load_manifest()
    if not manifest:
        print(f"期間: {start_date} 〜 {end_date}（テスト: {test_start} 以降）")
        race_dates = collect_race_calendar(start_date, end_date, sleep_sec)
        for race_id, race_date in race_dates.items():
            manifest[race_id] = {
                "race_id": race_id,
                "date": race_date.isoformat(),
                "split": "test" if race_date >= test_start else "train",
                "status": "pending",
            }

    excluded = [rid for rid in manifest if rid in exclude_ids]
    for race_id in excluded:
        del manifest[race_id]
    if excluded:
        print(f"取得済みのため除外: {len(excluded)}レース")

    save_manifest(manifest)
    return manifest


def download_races(manifest, sleep_sec):
    split_dirs = {"train": TRAIN_HTML_DIR, "test": TEST_HTML_DIR}
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    targets = sorted(manifest.values(), key=lambda r: (r["date"], r["race_id"]))
    saved = skipped = failed = 0
    for i, row in enumerate(_progress(targets, len(targets), "レースHTML"), start=1):
        output_path = split_dirs[row["split"]] / f"{row['race_id']}.html"
        if output_path.is_file():
            row["status"] = "saved"
            skipped += 1
            continue
        try:
            response = _get(f"https://db.netkeiba.com/race/{row['race_id']}/", sleep_sec)
            output_path.write_bytes(response.content)
            row["status"] = "saved"
            saved += 1
        except requests.RequestException:
            row["status"] = "failed"
            failed += 1
        # 中断に備えて定期的に進捗を書き出す
        if i % 100 == 0:
            save_manifest(manifest)

    save_manifest(manifest)
    return saved, skipped, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description="最新N年分のレースHTMLを学習用/テスト用に分けて取得します。")
    parser.add_argument("--years", type=int, default=5, help="取得する年数（既定: 5）")
    parser.add_argument("--test-years", type=int, default=1, help="テスト用にする直近の年数（既定: 1）")
    parser.add_argument("--end-date", default=None, help="取得期間の最終日 YYYY-MM-DD（既定: 昨日）")
    parser.add_argument("--sleep", type=float, default=1.0, help="リクエスト間隔（秒）")
    parser.add_argument("--list-only", action="store_true", help="レースID一覧の作成のみ行う")
    parser.add_argument("--refresh-list", action="store_true", help="既存のmanifestを使わずレースID一覧を作り直す")
    parser.add_argument(
        "--exclude-dir", action="append", default=[],
        help="取得済みとして除外するHTMLディレクトリ（複数指定可。data/race_html は常に除外）",
    )
    args = parser.parse_args(argv)

    if not 0 < args.test_years < args.years:
        parser.error("--test-years は 1 以上かつ --years 未満にしてください。")

    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d").date()
        if args.end_date else date.today() - timedelta(days=1)
    )
    exclude_ids = existing_race_ids(EXISTING_HTML_DIRS + [Path(d) for d in args.exclude_dir])

    manifest = build_manifest(end_date, args.years, args.test_years, exclude_ids, args.sleep, args.refresh_list)
    n_train = sum(1 for r in manifest.values() if r["split"] == "train")
    n_test = len(manifest) - n_train
    print(f"取得対象: {len(manifest)}レース（学習 {n_train} / テスト {n_test}） → {MANIFEST_PATH}")
    if not manifest:
        print("レースIDを取得できませんでした。ネットワーク接続を確認してください。", file=sys.stderr)
        return 1
    if args.list_only:
        return 0

    saved, skipped, failed = download_races(manifest, args.sleep)
    print(f"完了: 新規保存 {saved} / 保存済みスキップ {skipped} / 失敗 {failed}")
    print(f"学習用: {TRAIN_HTML_DIR}  テスト用: {TEST_HTML_DIR}")
    if failed:
        print("失敗分は再実行すると再取得します。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

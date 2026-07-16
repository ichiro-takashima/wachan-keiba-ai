"""race_idのリストから、出馬情報CSVと払戻情報CSVを構築する。

- HTMLはscraper.save_race_html_binary/load_race_html_textで data/race_html に
  キャッシュされ、既に保存済みのrace_idはHTTPリクエストを行わない。
- 出馬情報・払戻情報とも、CSVに既に含まれるrace_idはスキップする(レジューム可能)。
"""
import json
import re
from pathlib import Path

import pandas as pd

import scraper
from scraping.race_metadata import clean_html_comments, parse_race_metadata

RESULT_TABLE_REQUIRED_COLUMNS = {"着順", "馬番", "馬名"}

RACE_RESULT_COLUMNS = [
    "race_id", "馬", "騎手", "馬番", "走破時間", "オッズ", "通過順", "着順",
    "体重", "体重変化", "性", "齢", "斤量", "上がり", "人気", "レース名", "日付",
    "開催", "クラス", "芝・ダート", "距離", "回り", "馬場", "天気", "場id", "場名",
]

PAYOUT_TICKET_TYPES = ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]
PAYOUT_COLUMNS = ["race_id"] + PAYOUT_TICKET_TYPES


def _normalize_col(value):
    if isinstance(value, tuple):
        value = value[-1]
    return str(value).replace(" ", "").replace("　", "").replace("\n", "").strip()


def _is_premium_timeindex_cell(cell):
    classes = cell.get("class") or []
    return any("TimeIndex" in c for c in classes)


def _extract_result_rows(soup):
    """race_table_01を直接BeautifulSoupで解析し、行ごとの{見出し:値}辞書のリストを返す。

    netkeibaの「タイム指数」列(プレミアム機能)はヘッダー側のみstyle="display:none"が
    付いておりデータ側には付かない、という表示状態の不整合がある。
    pd.read_htmlに任せるとこの不整合でヘッダーとデータ列がずれ(通過/上り/単勝が
    ずれた列の値になる)ため、ヘッダー・データの両方からクラス名(TimeIndex*)で
    一貫して除外することで整合を取る。
    """
    table = soup.find("table", class_="race_table_01")
    if table is None:
        return None
    rows = table.find_all("tr")
    if not rows:
        return None

    header_cells = [c for c in rows[0].find_all("th") if not _is_premium_timeindex_cell(c)]
    headers = [_normalize_col(c.get_text(strip=True)) for c in header_cells]
    if not RESULT_TABLE_REQUIRED_COLUMNS.issubset(set(headers)):
        return None

    data_rows = []
    for tr in rows[1:]:
        cells = [c for c in tr.find_all("td") if not _is_premium_timeindex_cell(c)]
        if len(cells) != len(headers):
            # 中止・除外などでセル数が想定と異なる行は、誤った列に値を割り当てるより
            # 安全側に倒して読み飛ばす。
            continue
        values = [c.get_text(strip=True) or None for c in cells]
        data_rows.append(dict(zip(headers, values)))
    return data_rows


def _split_sex_age(value):
    text = "" if pd.isna(value) else str(value)
    match = re.match(r"(牡|牝|セ|せ|騙)(\d+)", text)
    if not match:
        return "", pd.NA
    return match.group(1), int(match.group(2))


def _split_body_weight(value):
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"(\d+)(?:\(([+-]?\d+)\))?", text)
    if not match:
        return pd.NA, pd.NA
    weight = int(match.group(1))
    diff = int(match.group(2)) if match.group(2) is not None else 0
    return weight, diff


def parse_race_html_to_rows(html_text, race_id):
    """1レース分のHTML文字列から、出馬情報の行リストと払戻情報の行を返す。"""
    cleaned_text = clean_html_comments(html_text)
    soup = scraper.BeautifulSoup(cleaned_text, "html.parser")

    result_rows = _extract_result_rows(soup)
    if not result_rows:
        return [], None, False

    meta = parse_race_metadata(html_text, race_id)

    rows = []
    for r in result_rows:
        sex, age = _split_sex_age(r.get("性齢"))
        weight, weight_diff = _split_body_weight(r.get("馬体重"))
        rows.append({
            "race_id": race_id,
            "馬": r.get("馬名"),
            "騎手": r.get("騎手"),
            "馬番": r.get("馬番"),
            "走破時間": r.get("タイム"),
            "オッズ": r.get("単勝"),
            "通過順": r.get("通過"),
            "着順": r.get("着順"),
            "体重": weight,
            "体重変化": weight_diff,
            "性": sex,
            "齢": age,
            "斤量": r.get("斤量"),
            "上がり": r.get("上り"),
            "人気": r.get("人気"),
            "レース名": meta["レース名"],
            "日付": meta["日付"],
            "開催": meta["開催"],
            "クラス": meta["クラス"],
            "芝・ダート": meta["芝・ダート"],
            "距離": meta["距離"],
            "回り": meta["回り"],
            "馬場": meta["馬場"],
            "天気": meta["天気"],
            "場id": meta["場id"],
            "場名": meta["場名"],
        })

    payouts = scraper._extract_payouts_from_soup(soup)
    payout_row = {"race_id": race_id}
    for ticket_type in PAYOUT_TICKET_TYPES:
        payout_row[ticket_type] = json.dumps(payouts.get(ticket_type, []), ensure_ascii=False)

    # ごく一部のページでは通過/上り/単勝/人気が本当に欠落している場合がある。
    # 既走馬(着順あり)の大半で通過順とオッズが同時に空なら、警告フラグを立てる。
    finished_rows = [row for row in rows if pd.notna(row["着順"])]
    layout_warning = False
    if finished_rows:
        both_missing = sum(1 for row in finished_rows if pd.isna(row["通過順"]) and pd.isna(row["オッズ"]))
        layout_warning = (both_missing / len(finished_rows)) > 0.5

    return rows, payout_row, layout_warning


def _existing_race_ids(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        return set()
    try:
        existing = pd.read_csv(path, usecols=["race_id"], dtype=str)
        return set(existing["race_id"].astype(str))
    except Exception:
        return set()


def _append_rows(csv_path, rows, columns):
    if not rows:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    write_header = not path.is_file()
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def build_race_and_payout_csv(
    race_id_list,
    results_csv="data/processed/race_results.csv",
    payouts_csv="data/processed/payouts.csv",
    base_dir="data",
    sleep_sec=1.0,
    show_progress=True,
):
    """race_idのリストを処理し、出馬情報CSVと払戻情報CSVに追記する。

    CSVに既に存在するrace_idはスキップするため、途中で中断しても再実行で再開できる。
    """
    done_results = _existing_race_ids(results_csv)
    done_payouts = _existing_race_ids(payouts_csv)
    target_ids = [rid for rid in race_id_list if rid not in done_results or rid not in done_payouts]

    iterator = target_ids
    if show_progress and scraper.tqdm is not None:
        iterator = scraper.tqdm(target_ids, desc="build_dataset")

    processed, failed, layout_warnings = 0, [], []
    for race_id in iterator:
        try:
            scraper.save_race_html_binary(race_id, base_dir=base_dir)  # 新規分のみ内部でsleep_sec相当(1秒)待機
            html_text = scraper.load_race_html_text(race_id, base_dir=base_dir)
            if not html_text:
                failed.append(race_id)
                continue

            rows, payout_row, layout_warning = parse_race_html_to_rows(html_text, race_id)
            if layout_warning:
                layout_warnings.append(race_id)
            if race_id not in done_results:
                _append_rows(results_csv, rows, RACE_RESULT_COLUMNS)
            if payout_row is not None and race_id not in done_payouts:
                _append_rows(payouts_csv, [payout_row], PAYOUT_COLUMNS)
            processed += 1
        except Exception:
            failed.append(race_id)

    return {
        "processed": processed,
        "skipped": len(race_id_list) - len(target_ids),
        "failed": failed,
        "layout_warnings": layout_warnings,
    }

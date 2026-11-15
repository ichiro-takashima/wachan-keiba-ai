"""収集済みレースHTMLに出てくる馬の血統(父・母・母父)を netkeiba から取得して保存する。

血統・戦績の相性モデル(model.affinity_model)の学習に使う。
- data/processed/horse_ids.csv : レースHTMLから抜き出した (race_id, 馬名, horse_id)
- data/processed/pedigree.csv  : horse_id ごとの父・母・母父(1頭取得するたびに追記)

途中で止めても、次回は未取得の馬だけを取りに行く。直近に走った馬から順に取得するので、
途中まででも予想に近い時期の馬ほど血統がそろう。

使い方:
    python -m scraping.collect_pedigree              # 未取得の馬をすべて取得(1頭あたり約1.5秒)
    python -m scraping.collect_pedigree --limit 3000 # 3000頭だけ取得
"""
import argparse
import csv
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HTML_DIRS = [Path("data/train/race_html"), Path("data/test/race_html")]
HORSE_IDS_PATH = Path("data/processed/horse_ids.csv")
PEDIGREE_PATH = Path("data/processed/pedigree.csv")
PEDIGREE_COLUMNS = ["horse_id", "name", "sire", "dam", "broodmare_sire"]

_ROW_RE = re.compile(r'<a href="[^"]*/horse/(\d{10})/"[^>]*>([^<]+)</a>')


def _horse_ids_in_file(path):
    """レース結果HTML1件の結果表から (race_id, 馬名, horse_id) を返す。"""
    content = Path(path).read_bytes()
    for encoding in ("utf-8", "euc-jp", "cp932"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = content.decode("utf-8", errors="ignore")
    start = text.find("race_table_01")
    if start < 0:
        return []
    end = text.find("</table>", start)
    table = text[start:end]
    race_id = Path(path).stem
    seen, rows = set(), []
    for horse_id, name in _ROW_RE.findall(table):
        if horse_id not in seen:
            seen.add(horse_id)
            rows.append((race_id, re.sub(r"\s+", "", name), horse_id))
    return rows


def build_horse_ids(rebuild=False):
    """全レースHTMLから (race_id, 馬名, horse_id) の表を作ってキャッシュする。"""
    if HORSE_IDS_PATH.exists() and not rebuild:
        return pd.read_csv(HORSE_IDS_PATH, encoding="utf-8-sig", dtype=str)
    paths = [p for d in HTML_DIRS for p in sorted(d.glob("*.html"))]
    rows = []
    with ProcessPoolExecutor() as pool:
        for file_rows in pool.map(_horse_ids_in_file, paths, chunksize=100):
            rows.extend(file_rows)
    df = pd.DataFrame(rows, columns=["race_id", "馬名", "horse_id"])
    HORSE_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HORSE_IDS_PATH, index=False, encoding="utf-8-sig")
    print(f"{len(paths)}レースから {df['horse_id'].nunique()}頭のhorse_idを抽出 → {HORSE_IDS_PATH}")
    return df


def load_pedigree():
    if not PEDIGREE_PATH.exists():
        return pd.DataFrame(columns=PEDIGREE_COLUMNS)
    return pd.read_csv(PEDIGREE_PATH, encoding="utf-8-sig", dtype=str)


def collect(limit=None):
    import scraper

    ids = build_horse_ids()
    done = set(load_pedigree()["horse_id"])
    # race_id の先頭は開催年なので、race_id が新しい(=直近に走った)馬から取得する
    latest = ids.groupby("horse_id")["race_id"].max().sort_values(ascending=False)
    todo = [h for h in latest.index if h not in done]
    if limit:
        todo = todo[:limit]
    print(f"取得済み {len(done)}頭 / 未取得 {len(todo)}頭を取得します")

    write_header = not PEDIGREE_PATH.exists()
    started = time.time()
    failed = 0
    with open(PEDIGREE_PATH, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PEDIGREE_COLUMNS)
        if write_header:
            writer.writeheader()
        for i, horse_id in enumerate(todo, start=1):
            ped = scraper.scrape_horse_ped(horse_id)  # 内部で1秒待つ
            if ped.get("sire", "不明") == "不明":
                failed += 1  # 書かずにおき、次回また取りに行く
            else:
                writer.writerow({k: ped.get(k, "") for k in PEDIGREE_COLUMNS})
                f.flush()
            if i % 200 == 0:
                rate = (time.time() - started) / i
                print(f"  {i}/{len(todo)}頭 (失敗{failed}) 残り約{(len(todo) - i) * rate / 3600:.1f}時間", flush=True)
    print(f"完了: {len(todo) - failed}頭取得 / 失敗{failed}頭 → {PEDIGREE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="収集済みレースの出走馬の血統を取得する")
    parser.add_argument("--limit", type=int, default=None, help="今回取得する最大頭数")
    parser.add_argument("--rebuild-ids", action="store_true", help="horse_ids.csv を作り直す")
    args = parser.parse_args()
    if args.rebuild_ids:
        build_horse_ids(rebuild=True)
    collect(limit=args.limit)


if __name__ == "__main__":
    main()

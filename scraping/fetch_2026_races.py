"""2026年度(2026-01-01〜2026-12-31)の中央競馬全レースを取得するスクリプト。

db.netkeiba.com のレース一覧を1日単位で巡回してrace_idを収集し、
出馬情報CSV・払戻情報CSVへ保存する。まだ開催されていない日は一覧が空になるため、
そのままスキップされる(実行日以降のレースは開催後に再実行すれば追記される)。

例:
    python -m scraping.fetch_2026_races
    python -m scraping.fetch_2026_races --end 2026-08-11
"""
import argparse
from datetime import date

from scraping.build_dataset import build_race_and_payout_csv
from scraping.race_calendar import get_race_id_list

YEAR = 2026
DEFAULT_START = date(YEAR, 1, 1)
DEFAULT_END = date(YEAR, 12, 31)


def main():
    parser = argparse.ArgumentParser(description=f"{YEAR}年度の中央競馬全レースを取得する")
    parser.add_argument("--start", default=DEFAULT_START.isoformat(), help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END.isoformat(), help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--results-csv", default="data/processed/race_results.csv")
    parser.add_argument("--payouts-csv", default="data/processed/payouts.csv")
    parser.add_argument("--sleep", type=float, default=1.0, help="新規ダウンロード1件あたりのsleep秒数")
    args = parser.parse_args()

    print(f"カレンダーを取得中: {args.start} 〜 {args.end}")
    race_ids = get_race_id_list(args.start, args.end, sleep_sec=args.sleep)
    print(f"対象race_id数: {len(race_ids)}")

    result = build_race_and_payout_csv(
        race_ids,
        results_csv=args.results_csv,
        payouts_csv=args.payouts_csv,
        sleep_sec=args.sleep,
    )
    print(f"処理完了: 新規処理={result['processed']} / スキップ={result['skipped']} / 失敗={len(result['failed'])}")
    if result["failed"]:
        print("失敗したrace_id:", result["failed"][:20])
    if result["layout_warnings"]:
        print(
            "[警告] 古い形式のページで通過順/オッズ等の列がずれて取得できていない可能性があるrace_id:",
            result["layout_warnings"],
        )


if __name__ == "__main__":
    main()

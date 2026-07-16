"""スクレイピング実行用CLI。

例:
    python -m scraping.run_scrape --start 2026-06-01 --end 2026-06-14
    python -m scraping.run_scrape --race-ids 202405020811,202405020812
    python -m scraping.run_scrape --from-cache   # data/race_html/*.html のみを対象にCSV再構築(ネットワークアクセス無し)
"""
import argparse
from pathlib import Path

from scraping.build_dataset import build_race_and_payout_csv
from scraping.race_calendar import get_race_id_list, _is_jra_race_id


def _race_ids_from_cache(base_dir="data"):
    race_html_dir = Path(base_dir) / "race_html"
    all_ids = (p.stem for p in race_html_dir.glob("*.html"))
    # data/race_html には検証用に取得した地方競馬(NAR)のHTMLも混在し得るため、中央競馬のみに絞る
    return sorted(rid for rid in all_ids if _is_jra_race_id(rid))


def main():
    parser = argparse.ArgumentParser(description="中央競馬のレース結果・払戻情報をCSVへ収集する")
    parser.add_argument("--start", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--race-ids", help="race_idをカンマ区切りで直接指定")
    parser.add_argument("--from-cache", action="store_true", help="data/race_html配下の既存HTMLのみを対象にする(ネットワークアクセス無し)")
    parser.add_argument("--results-csv", default="data/processed/race_results.csv")
    parser.add_argument("--payouts-csv", default="data/processed/payouts.csv")
    parser.add_argument("--sleep", type=float, default=1.0, help="新規ダウンロード1件あたりのsleep秒数")
    args = parser.parse_args()

    if args.from_cache:
        race_ids = _race_ids_from_cache()
    elif args.race_ids:
        race_ids = [rid.strip() for rid in args.race_ids.split(",") if rid.strip()]
    elif args.start and args.end:
        print(f"カレンダーを取得中: {args.start} 〜 {args.end}")
        race_ids = get_race_id_list(args.start, args.end, sleep_sec=args.sleep)
    else:
        parser.error("--from-cache か --race-ids か (--start と --end) のいずれかを指定してください")
        return

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

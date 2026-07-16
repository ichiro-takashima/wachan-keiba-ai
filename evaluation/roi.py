"""Step3: 払戻データを使った単勝・複勝の回収率計算。

予測確率が閾値以上の馬に単勝・複勝で一定額(既定100円)を賭けたと仮定し、
tickets.calculate_return (既存の的中判定・回収計算ロジック)を再利用して集計する。
"""
import json

import pandas as pd

from tickets import calculate_return

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


def load_payouts(payouts_csv="data/processed/payouts.csv"):
    """race_id -> {券種: [{"combo":..., "pay":...}, ...]} の辞書を返す。"""
    df = pd.read_csv(payouts_csv, encoding="utf-8-sig", dtype={"race_id": str})
    ticket_cols = [c for c in df.columns if c != "race_id"]

    payouts_by_race = {}
    for _, row in df.iterrows():
        payouts_by_race[row["race_id"]] = {
            col: json.loads(row[col]) if pd.notna(row[col]) and row[col] else []
            for col in ticket_cols
        }
    return payouts_by_race


def compute_roi_table(test_df, payouts_by_race, thresholds=DEFAULT_THRESHOLDS, stake=100):
    """test_df(race_id, 馬番, y_pred_proba列を含む)から、閾値ごとの単勝・複勝回収率表を作る。"""
    rows = []
    for threshold in thresholds:
        selected = test_df[test_df["y_pred_proba"] >= threshold]

        tan_bet_total = tan_return_total = 0
        fuku_bet_total = fuku_return_total = 0
        for race_id, group in selected.groupby("race_id"):
            payouts = payouts_by_race.get(race_id, {})
            horse_numbers = [str(int(round(h))) for h in group["馬番"] if pd.notna(h)]
            tan_bets = [{"type": "単勝", "combo": h, "amount": stake} for h in horse_numbers]
            fuku_bets = [{"type": "複勝", "combo": h, "amount": stake} for h in horse_numbers]

            t_bet, t_ret, _, _ = calculate_return(tan_bets, payouts)
            f_bet, f_ret, _, _ = calculate_return(fuku_bets, payouts)
            tan_bet_total += t_bet
            tan_return_total += t_ret
            fuku_bet_total += f_bet
            fuku_return_total += f_ret

        rows.append({
            "閾値": threshold,
            "Betting Horses": len(selected),
            "単勝購入額": tan_bet_total,
            "単勝回収額": tan_return_total,
            "単勝回収率(%)": round(tan_return_total / tan_bet_total * 100, 1) if tan_bet_total else float("nan"),
            "複勝購入額": fuku_bet_total,
            "複勝回収額": fuku_return_total,
            "複勝回収率(%)": round(fuku_return_total / fuku_bet_total * 100, 1) if fuku_bet_total else float("nan"),
        })

    return pd.DataFrame(rows)


def main():
    import argparse

    from model.common import train_and_evaluate

    parser = argparse.ArgumentParser(description="予測確率の閾値別に単勝・複勝回収率を計算する")
    parser.add_argument("--features-csv", default="data/processed/features.csv")
    parser.add_argument("--payouts-csv", default="data/processed/payouts.csv")
    parser.add_argument("--split", choices=["random", "time"], default="random")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--stake", type=int, default=100, help="1点あたりの購入額(円)")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, encoding="utf-8-sig", dtype={"race_id": str})
    result = train_and_evaluate(df, split=args.split, test_size=args.test_size)

    payouts_by_race = load_payouts(args.payouts_csv)
    roi_table = compute_roi_table(result.test_df, payouts_by_race, stake=args.stake)

    print(f"テスト件数: {len(result.test_df)} (split={args.split})")
    print(roi_table.to_string(index=False))


if __name__ == "__main__":
    main()

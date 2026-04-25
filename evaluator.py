from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Literal

import pandas as pd

BetType = Literal["馬連", "ワイド", "三連複"]


@dataclass(frozen=True)
class EvaluationResult:
    """シミュレーション結果をまとめるためのコンテナ。"""

    strategy_name: str
    top_n: int
    total_races: int
    total_tickets: int
    total_bet: int
    total_return: int
    hit_tickets: int
    hit_races: int
    ticket_hit_rate: float
    race_hit_rate: float
    return_rate: float


class Evaluator:
    """
    予測スコア上位N頭のボックス買いを評価するクラス。

    Notes
    -----
    - 払戻金は100円あたりの金額を想定。
    - 1点あたり購入額(`stake_per_ticket`)が100円以外の場合は、
      `払戻金 * (stake_per_ticket / 100)`で換算する。
    """

    def __init__(
        self,
        top_n: int = 3,
        stake_per_ticket: int = 100,
        bet_types: Iterable[BetType] = ("馬連", "ワイド", "三連複"),
    ) -> None:
        if top_n < 2:
            raise ValueError("top_n は2以上で指定してください。")
        if stake_per_ticket <= 0:
            raise ValueError("stake_per_ticket は1以上で指定してください。")
        self.top_n = top_n
        self.stake_per_ticket = stake_per_ticket
        self.bet_types = tuple(bet_types)

    @staticmethod
    def _normalize_combo(combo: str, bet_type: str) -> str:
        """券種に応じて組番文字列を正規化する。"""
        text = str(combo).replace("ー", "-").replace("→", "-").replace(">", "-").replace(" ", "")
        parts = [p for p in text.split("-") if p]

        if bet_type in {"馬連", "ワイド", "三連複"}:
            try:
                return "-".join(sorted(parts, key=int))
            except ValueError:
                return "-".join(sorted(parts))
        return "-".join(parts)

    def _build_box_tickets(
        self,
        ranking_df: pd.DataFrame,
        race_id_col: str,
        horse_col: str,
        strategy_name: str,
    ) -> pd.DataFrame:
        """レースごとの上位N頭からボックス買い目を展開する。"""
        tickets: list[dict] = []
        combo_size_map = {"馬連": 2, "ワイド": 2, "三連複": 3}

        for race_id, group in ranking_df.groupby(race_id_col):
            horses = [str(v) for v in group[horse_col].tolist()]
            for bet_type in self.bet_types:
                r = combo_size_map.get(bet_type)
                if r is None or len(horses) < r:
                    continue
                for combo in combinations(horses, r):
                    combo_text = "-".join(combo)
                    tickets.append(
                        {
                            "race_id": race_id,
                            "券種": bet_type,
                            "組番": self._normalize_combo(combo_text, bet_type),
                            "strategy": strategy_name,
                            "購入額": self.stake_per_ticket,
                        }
                    )

        return pd.DataFrame(tickets)

    def _summarize(self, detail_df: pd.DataFrame, strategy_name: str) -> EvaluationResult:
        total_tickets = int(len(detail_df))
        total_races = int(detail_df["race_id"].nunique()) if not detail_df.empty else 0
        total_bet = int(detail_df["購入額"].sum()) if not detail_df.empty else 0
        total_return = int(detail_df["払戻額"].sum()) if not detail_df.empty else 0
        hit_tickets = int(detail_df["的中"].sum()) if not detail_df.empty else 0

        if detail_df.empty:
            hit_races = 0
        else:
            hit_races = int(
                detail_df.groupby("race_id")["的中"].max().sum()
            )

        ticket_hit_rate = (hit_tickets / total_tickets * 100) if total_tickets else 0.0
        race_hit_rate = (hit_races / total_races * 100) if total_races else 0.0
        return_rate = (total_return / total_bet * 100) if total_bet else 0.0

        return EvaluationResult(
            strategy_name=strategy_name,
            top_n=self.top_n,
            total_races=total_races,
            total_tickets=total_tickets,
            total_bet=total_bet,
            total_return=total_return,
            hit_tickets=hit_tickets,
            hit_races=hit_races,
            ticket_hit_rate=ticket_hit_rate,
            race_hit_rate=race_hit_rate,
            return_rate=return_rate,
        )

    def evaluate(
        self,
        prediction_df: pd.DataFrame,
        payout_df: pd.DataFrame,
        *,
        score_col: str = "pred_score",
        race_id_col: str = "race_id",
        horse_col: str = "horse_number",
        payout_race_id_col: str = "race_id",
        payout_type_col: str = "券種",
        payout_combo_col: str = "組番",
        payout_amount_col: str = "払戻金",
        strategy_name: str = "model",
    ) -> tuple[EvaluationResult, pd.DataFrame]:
        """予測スコア上位N頭ボックスのシミュレーションを実施する。"""
        required_pred_cols = {race_id_col, horse_col, score_col}
        required_pay_cols = {
            payout_race_id_col,
            payout_type_col,
            payout_combo_col,
            payout_amount_col,
        }
        missing_pred = required_pred_cols - set(prediction_df.columns)
        missing_pay = required_pay_cols - set(payout_df.columns)
        if missing_pred:
            raise ValueError(f"prediction_df に必要な列が不足しています: {sorted(missing_pred)}")
        if missing_pay:
            raise ValueError(f"payout_df に必要な列が不足しています: {sorted(missing_pay)}")

        ranked = (
            prediction_df.sort_values([race_id_col, score_col], ascending=[True, False])
            .groupby(race_id_col, as_index=False)
            .head(self.top_n)
            .copy()
        )

        tickets = self._build_box_tickets(ranked, race_id_col, horse_col, strategy_name=strategy_name)
        if tickets.empty:
            empty_detail = pd.DataFrame(
                columns=["race_id", "券種", "組番", "strategy", "購入額", "払戻金", "払戻額", "的中"]
            )
            return self._summarize(empty_detail, strategy_name), empty_detail

        payouts = payout_df.rename(
            columns={
                payout_race_id_col: "race_id",
                payout_type_col: "券種",
                payout_combo_col: "組番",
                payout_amount_col: "払戻金",
            }
        ).copy()
        payouts["組番"] = payouts.apply(
            lambda x: self._normalize_combo(x["組番"], x["券種"]), axis=1
        )
        payouts["払戻金"] = pd.to_numeric(payouts["払戻金"], errors="coerce").fillna(0)

        detail = tickets.merge(payouts[["race_id", "券種", "組番", "払戻金"]], on=["race_id", "券種", "組番"], how="left")
        detail["払戻金"] = detail["払戻金"].fillna(0)
        detail["払戻額"] = (detail["払戻金"] * (self.stake_per_ticket / 100)).round().astype(int)
        detail["的中"] = (detail["払戻額"] > 0).astype(int)

        return self._summarize(detail, strategy_name), detail

    def compare_model_vs_popularity(
        self,
        prediction_df: pd.DataFrame,
        payout_df: pd.DataFrame,
        *,
        score_col: str = "pred_score",
        popularity_col: str = "popularity",
        race_id_col: str = "race_id",
        horse_col: str = "horse_number",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """AI予測と人気順のシミュレーション結果を比較する。"""
        model_result, model_detail = self.evaluate(
            prediction_df,
            payout_df,
            score_col=score_col,
            race_id_col=race_id_col,
            horse_col=horse_col,
            strategy_name="model",
        )

        if popularity_col not in prediction_df.columns:
            raise ValueError(f"prediction_df に必要な列が不足しています: ['{popularity_col}']")

        pop_df = prediction_df.copy()
        pop_df["_pop_rank"] = pd.to_numeric(pop_df[popularity_col], errors="coerce")
        pop_df = pop_df.sort_values([race_id_col, "_pop_rank"], ascending=[True, True])
        # evaluate() は score_col の降順を使うため、人気上位を擬似スコア化して流用
        pop_df["_pop_score"] = -pop_df["_pop_rank"].fillna(9999)

        pop_result, pop_detail = self.evaluate(
            pop_df,
            payout_df,
            score_col="_pop_score",
            race_id_col=race_id_col,
            horse_col=horse_col,
            strategy_name="popularity",
        )

        summary_df = pd.DataFrame(
            [
                vars(model_result),
                vars(pop_result),
            ]
        )
        detail_df = pd.concat([model_detail, pop_detail], ignore_index=True)
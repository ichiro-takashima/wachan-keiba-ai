"""オッズ取得(発売中・発売前)、ガミチェック、候補プランの単体テスト。

実行: python -m unittest tests.test_gami -v
"""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import scraper
from gami_prompt import (
    allocate_with_floor,
    build_candidate_plans,
    build_gami_free_prompt,
    finish_distribution,
    format_plans,
)
from tickets import check_gamiri


def _pred_df(n=8):
    weights = [0.30, 0.20, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03][:n]
    return pd.DataFrame({
        "馬番": range(1, n + 1), "馬名": [f"馬{i}" for i in range(1, n + 1)],
        "推定勝率": weights, "3着内確率": [min(w * 2.5, 0.9) for w in weights],
    })


def _response(payload):
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    return response


class ScrapeRealtimeOddsTest(unittest.TestCase):
    def test_requests_live_odds_and_parses_them(self):
        payload = {"status": "middle", "data": {"odds": {"1": {"01": ["36.3", "0", "15"], "02": ["1,234.5", "0", "2"]}}}}
        with patch.object(scraper.session, "get", return_value=_response(payload)) as get:
            result = scraper.scrape_realtime_odds("202605050211", ["単勝"])
        self.assertEqual(get.call_args.kwargs["params"]["action"], "init")
        self.assertEqual(result["単勝"]["1"], {"odds": 36.3, "odds_max": 36.3})
        self.assertEqual(result["単勝"]["2"]["odds"], 1234.5)

    def test_unsold_race_returns_empty_without_raising(self):
        for payload in (
            {"status": "NG", "data": "", "reason": "empty free odds schedule"},
            {"status": "yoso", "data": {"odds": []}},
            {"status": "middle", "data": {"odds": {"1": []}}},
            [],
        ):
            with patch.object(scraper.session, "get", return_value=_response(payload)):
                self.assertEqual(scraper.scrape_realtime_odds("202605050311", ["単勝"]), {}, payload)

    def test_zero_odds_are_treated_as_unavailable(self):
        payload = {"data": {"odds": {"1": {"01": ["0.0", "0", "0"], "02": ["5.0", "0", "1"]}}}}
        with patch.object(scraper.session, "get", return_value=_response(payload)):
            result = scraper.scrape_realtime_odds("202605050211", ["単勝"])
        self.assertEqual(list(result["単勝"]), ["2"])

    def test_local_race_uses_nar_api_for_win_odds_only(self):
        payload = {"status": "OK", "ary_odds": {"01": {"Odds": "24.8"}, "02": {"Odds": "---.-"}}}
        with patch.object(scraper.session, "get", return_value=_response(payload)) as get:
            result = scraper.scrape_realtime_odds("202646110309", ["単勝", "馬連"])
        self.assertIn("nar.netkeiba.com", get.call_args.args[0])
        self.assertEqual(result, {"単勝": {"1": {"odds": 24.8, "odds_max": 24.8}}})


class CheckGamiriTest(unittest.TestCase):
    BETS = [{"type": "ワイド", "combo": "1-2", "amount": 800}, {"type": "ワイド", "combo": "1-3", "amount": 200}]

    def _statuses(self, odds, ratio=1.0):
        rows, _ = check_gamiri(self.BETS, {"ワイド": odds}, ratio)
        return [r["status"] for r in rows]

    def test_no_odds_holds_judgement_and_gives_required_odds(self):
        rows, total = check_gamiri(self.BETS, {}, 0.8)
        self.assertEqual(total, 1000)
        self.assertEqual([r["status"] for r in rows], ["オッズ未発表", "オッズ未発表"])
        self.assertEqual([r["required_odds"] for r in rows], [1.0, 4.0])

    def test_zero_odds_is_not_reported_as_gami(self):
        self.assertEqual(self._statuses({"1-2": {"odds": 0.0}, "1-3": {"odds": 5.0}})[0], "オッズ未発表")

    def test_status_depends_on_tolerance(self):
        odds = {"1-2": {"odds": 2.0}, "1-3": {"odds": 4.0}}  # 払戻 1600 / 800 に対し合計1000
        self.assertEqual(self._statuses(odds, 1.0), ["問題なし", "ガミ"])
        self.assertEqual(self._statuses(odds, 0.8), ["問題なし", "ガミ（許容内）"])
        self.assertEqual(self._statuses(odds, 0.9), ["問題なし", "ガミ"])


class AllocationTest(unittest.TestCase):
    def test_every_bet_meets_the_floor(self):
        odds = [2.4, 5.1, 13.3, 30.9]
        for ratio in (1.0, 0.8, 0.6):
            amounts = allocate_with_floor(odds, 2000, ratio)
            self.assertLessEqual(sum(amounts), 2000)
            self.assertTrue(all(a % 100 == 0 and a * o >= sum(amounts) * ratio for a, o in zip(amounts, odds)))

    def test_relaxed_floor_allows_low_odds_combinations(self):
        odds = [1.5, 2.0, 2.5]  # 合成オッズ約0.64倍: 厳密には組めないが最低回収率60%なら組める
        self.assertIsNone(allocate_with_floor(odds, 3000, 1.0))
        self.assertIsNotNone(allocate_with_floor(odds, 3000, 0.6))

    def test_budget_too_small_for_points(self):
        self.assertIsNone(allocate_with_floor([10.0, 10.0], 100, 0.8))


class CandidatePlansTest(unittest.TestCase):
    def test_distribution_sums_to_one(self):
        self.assertAlmostEqual(sum(finish_distribution(_pred_df()).values()), 1.0)

    def test_plans_without_odds_are_estimated_not_empty(self):
        plans = build_candidate_plans(_pred_df(), "標準", {}, 1000, 0.8)
        built = [p for p in plans if p["bets"]]
        self.assertTrue(built)
        self.assertTrue(all(p["estimated"] for p in built))
        self.assertIn("推定", format_plans(plans, 0.8))

    def test_multi_point_plans_come_first_then_by_hit_rate(self):
        plans = [p for p in build_candidate_plans(_pred_df(), "堅い", {}, 3000, 1.0) if p["bets"]]
        multi = [p for p in plans if len(p["bets"]) >= 3]
        few = [p for p in plans if len(p["bets"]) < 3]
        self.assertTrue(multi and few)
        self.assertEqual(plans, multi + few)
        for group in (multi, few):
            rates = [p["hit_prob"] for p in group]
            self.assertEqual(rates, sorted(rates, reverse=True))

    def test_plans_reduced_to_the_same_bets_are_merged(self):
        # 上位馬の複勝が低オッズだと「複勝 上位3頭」は1点まで削られ「複勝◎1点」と同じになる
        odds = {"複勝": {str(i): {"odds": 1.1} for i in range(1, 9)}}
        plans = build_candidate_plans(_pred_df(), "標準", odds, 1000, 1.0)
        place_plans = [p for p in plans if p["bets"] and p["bets"][0]["type"] == "複勝"]
        self.assertEqual(len(place_plans), 1)

    def test_real_odds_used_where_available_and_estimate_elsewhere(self):
        odds = {"単勝": {str(i): {"odds": 3.0 + i} for i in range(1, 9)}}
        plans = build_candidate_plans(_pred_df(), "標準", odds, 3000, 0.8)
        by_type = {}
        for plan in plans:
            if plan["bets"]:
                by_type[plan["bets"][0]["type"]] = plan["estimated"]
        self.assertFalse(by_type["単勝"])
        self.assertTrue(by_type["馬連"])

    def test_undetermined_umaban_yields_no_plans(self):
        df = _pred_df()
        df["馬番"] = float("nan")
        self.assertEqual(build_candidate_plans(df, "標準", {}, 1000), [])

    def test_prompt_builds_without_odds(self):
        df = _pred_df().assign(
            印="", 騎手="", オッズ=float("nan"), 人気=float("nan"), 期待値=float("nan"), 期待値ランク="-", 根拠=[[]] * 8,
        )
        prediction = {
            "predictions": df, "model_label": "m", "model_key": "global", "fallback_reason": None,
            "conditions": {"surface": "芝", "distance": 1600, "going": "良"},
        }
        prompt = build_gami_free_prompt("202605050311", {"name": "R"}, prediction, None, {}, 1000, model_metrics={"auc": float("nan")})
        self.assertIn("推定オッズ", prompt)
        self.assertIn("オッズ未発表は見送りの理由にしない", prompt)


if __name__ == "__main__":
    unittest.main()

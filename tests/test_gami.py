"""オッズ取得(発売中・発売前)、ガミチェック、候補プランの単体テスト。

実行: python -m unittest tests.test_gami -v
"""
import unittest
from itertools import combinations, permutations
from unittest.mock import MagicMock, patch

import pandas as pd

import scraper
from gami_prompt import (
    MAX_PLACE_HORSES,
    PAYOUT_RATE,
    _hit_probability,
    allocate_with_floor,
    build_candidate_plans,
    build_gami_free_prompt,
    finish_distribution,
    format_plans,
    merge_model_views,
    plan_race,
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

    def test_plans_are_sorted_by_score_and_all_buildable(self):
        plans = build_candidate_plans(_pred_df(), "堅い", {}, 3000, 1.0)
        self.assertTrue(plans)
        self.assertTrue(all(p["bets"] for p in plans))  # 組めないプランは出さない
        scores = [p["score"] for p in plans]
        self.assertEqual(scores, sorted(scores, reverse=True))

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


def _race(live_wins, aff_wins=None):
    """出馬表モデル(live)と相性モデル(affinity)の予測表。live は3着内確率の順に並べる。"""
    n = len(live_wins)
    live = pd.DataFrame({
        "馬番": range(1, n + 1), "馬名": [f"馬{i}" for i in range(1, n + 1)],
        "推定勝率": live_wins, "3着内確率": [min(w * 2.5, 0.9) for w in live_wins],
    }).sort_values("3着内確率", ascending=False, kind="stable").reset_index(drop=True)
    aff = None
    if aff_wins is not None:
        aff = pd.DataFrame({"馬番": range(1, n + 1), "馬名": [f"馬{i}" for i in range(1, n + 1)], "推定勝率": aff_wins})
    return live, aff


def _market_odds(live):
    """live の勝率を市場の評価とみなした、全券種の現在オッズ(払戻率込み)。"""
    dist = finish_distribution(live)
    horses = [int(h) for h in live["馬番"]]
    combos = {
        "単勝": [str(h) for h in horses], "複勝": [str(h) for h in horses],
        "馬連": [f"{a}-{b}" for a, b in combinations(sorted(horses), 2)],
        "ワイド": [f"{a}-{b}" for a, b in combinations(sorted(horses), 2)],
        "馬単": [f"{a}-{b}" for a, b in permutations(horses, 2)],
        "3連複": ["-".join(map(str, c)) for c in combinations(sorted(horses), 3)],
    }
    table = {}
    for t_type, keys in combos.items():
        table[t_type] = {}
        for key in keys:
            p = _hit_probability(dist, t_type, key)
            if p > 0:
                table[t_type][key] = {"odds": max(round(PAYOUT_RATE[t_type] / p, 1), 1.1)}
    return table


# 1位が抜けていない・混戦でもない(どのパターンにも当たらない)出馬表モデルの勝率
FLAT_WINS = [0.22, 0.19, 0.16, 0.13, 0.10, 0.08, 0.07, 0.05]
# 8番(出馬表モデル8位・単勝13倍前後)を、相性モデルだけが1位に評価している
LONGSHOT_AFF = [0.16, 0.13, 0.12, 0.10, 0.09, 0.05, 0.05, 0.30]
STRONG_WINS = [0.45, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
CLOSE_WINS = [0.15, 0.14, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11]


def _scenario(name):
    """(situation, plans)。A〜D とパターンなしのダミーレース。"""
    if name == "A":
        live, aff = _race([0.20, 0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.06], LONGSHOT_AFF)
        return plan_race(live, "標準", _market_odds(live), 3000, 1.0, aff)
    if name == "B":
        live, aff = _race(STRONG_WINS, STRONG_WINS)
        return plan_race(live, "標準", _market_odds(live), 3000, 1.0, aff)
    if name == "C":
        live, aff = _race(CLOSE_WINS, CLOSE_WINS)
        return plan_race(live, "標準", _market_odds(live), 3000, 1.0, aff)
    if name == "D":
        # 1強(B)にも当たるが、予算500円で馬連・ワイドが低オッズなので3点買いはトリガミ確定 → D が優先
        live, aff = _race(STRONG_WINS, STRONG_WINS)
        odds = {"単勝": {"1": {"odds": 2.5}, "2": {"odds": 4.0}, "3": {"odds": 6.0}},
                "馬連": {f"{a}-{b}": {"odds": 2.0} for a, b in combinations(range(1, 9), 2)},
                "ワイド": {f"{a}-{b}": {"odds": 1.8} for a, b in combinations(range(1, 9), 2)}}
        return plan_race(live, "標準", odds, 500, 1.0, aff)
    live, aff = _race(FLAT_WINS, FLAT_WINS)
    return plan_race(live, "標準", _market_odds(live), 3000, 1.0, aff)


SCENARIOS = ["A", "B", "C", "D", "none"]


class RaceSituationPlanTest(unittest.TestCase):
    def test_pattern_A_puts_longshot_bets_first(self):
        situation, plans = _scenario("A")
        self.assertEqual(situation["pattern"], "A")
        self.assertEqual(situation["horse"], 8)
        top = plans[0]
        self.assertEqual(top["type_tag"], "血統穴馬狙い")
        self.assertIn(top["bets"][0]["type"], {"単勝", "複勝", "ワイド"})
        self.assertTrue(all("8" in b["combo"].split("-") for b in top["bets"]))
        self.assertIn(8, situation["gaps"]["馬番"].tolist())

    def test_pattern_B_puts_axis_bets_first(self):
        situation, plans = _scenario("B")
        self.assertEqual(situation["pattern"], "B")
        self.assertEqual(plans[0]["type_tag"], "1強軸流し")
        self.assertIn(plans[0]["bets"][0]["type"], {"馬連", "馬単", "3連複"})
        self.assertTrue(all("1" in b["combo"].split("-") for b in plans[0]["bets"]))

    def test_pattern_C_puts_box_bets_first(self):
        situation, plans = _scenario("C")
        self.assertEqual(situation["pattern"], "C")
        self.assertEqual(plans[0]["type_tag"], "混戦BOX")
        self.assertIn(plans[0]["bets"][0]["type"], {"ワイド", "馬連"})
        self.assertIn("BOX", plans[0]["name"])

    def test_pattern_D_wins_over_B_and_puts_pinpoint_bets_first(self):
        situation, plans = _scenario("D")
        self.assertEqual(situation["pattern"], "D")
        self.assertIn("B", situation["matched"])  # B にも当たっているが D が優先
        self.assertEqual(plans[0]["type_tag"], "少額ピンポイント")
        self.assertIn(plans[0]["bets"][0]["type"], {"単勝", "ワイド"})
        self.assertLessEqual(len(plans[0]["bets"]), 2)
        self.assertLessEqual(plans[0]["total"], 500)

    def test_no_pattern_uses_existing_plans_only(self):
        situation, plans = _scenario("none")
        self.assertIsNone(situation["pattern"])
        self.assertTrue(plans)
        self.assertTrue(all(p["pattern"] is None and p["type_tag"] is None for p in plans))

    def test_never_more_than_two_place_horses(self):
        for name in SCENARIOS:
            for plan in _scenario(name)[1]:
                with self.subTest(scenario=name, plan=plan["name"]):
                    self.assertLessEqual(sum(b["type"] == "複勝" for b in plan["bets"]), MAX_PLACE_HORSES)

    def test_every_plan_returns_at_least_the_stake(self):
        live, aff = _race(FLAT_WINS, LONGSHOT_AFF)
        runs = [_scenario(name)[1] for name in SCENARIOS]
        # 画面で低い回収率を指定しても100%未満のプランは出さない
        runs.append(build_candidate_plans(live, "標準", _market_odds(live), 1000, 0.6, aff))
        runs.append(build_candidate_plans(live, "標準", {}, 1000, 0.6, aff))
        for plans in runs:
            self.assertTrue(plans)
            for plan in plans:
                self.assertGreaterEqual(plan["min_return_ratio"], 1.0)
                for b in plan["bets"]:
                    self.assertGreaterEqual(b["amount"] * b["odds"], plan["total"] - 1e-6, plan["name"])

    def test_top_plans_are_not_all_the_same_ticket_type(self):
        for name in SCENARIOS:
            plans = _scenario(name)[1]
            top_types = [p["bets"][0]["type"] for p in plans[:3]]
            with self.subTest(scenario=name, top=top_types):
                self.assertGreaterEqual(len(set(top_types)), 2)
                self.assertTrue(all(top_types.count(t) <= 2 for t in top_types))

    def test_scores_use_expected_value_with_odds_and_hit_rate_without(self):
        self.assertEqual(_scenario("B")[0]["score_basis"], "期待値")
        live, aff = _race(STRONG_WINS, STRONG_WINS)
        situation, plans = plan_race(live, "標準", {}, 3000, 1.0, aff)
        self.assertEqual(situation["score_basis"], "的中率")
        self.assertEqual(situation["pattern"], "B")
        self.assertEqual(plans[0]["type_tag"], "1強軸流し")

    def test_missing_odds_and_one_sided_horses_do_not_break(self):
        live, aff = _race(FLAT_WINS, LONGSHOT_AFF)
        live["オッズ"] = float("nan")
        aff = aff[aff["馬番"] != 3]  # 相性モデルにいない馬
        aff = pd.concat([aff, pd.DataFrame({"馬番": [99], "馬名": ["出馬表にいない馬"], "推定勝率": [0.01]})])
        situation, plans = plan_race(live, "標準", {}, 3000, 1.0, aff)
        self.assertTrue(plans)
        view = situation["view"]
        self.assertTrue(pd.isna(view.loc[view["馬番"] == 3, "相性勝率"]).all())
        self.assertAlmostEqual(float(view.loc[view["馬番"] == 3, "期待値用勝率"].iloc[0]), 0.16)
        # オッズがないので単勝10倍の条件は使えないが、順位差(8位→1位)で A と判定できる
        self.assertEqual(situation["pattern"], "A")

    def test_expected_value_uses_average_of_both_models(self):
        live, aff = _race(FLAT_WINS, LONGSHOT_AFF)
        view = merge_model_views(live, aff)
        row = view[view["馬番"] == 8].iloc[0]
        self.assertAlmostEqual(row["期待値用勝率"], (0.05 + 0.30) / 2)
        self.assertEqual(row["順位差"], 7)

    def test_prompt_contains_type_tag_evidence_and_gaps(self):
        live, aff = _race([0.20, 0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.06], LONGSHOT_AFF)
        df = live.assign(印="", 騎手="", オッズ=float("nan"), 人気=float("nan"), 期待値=float("nan"),
                         期待値ランク="-", 根拠=[[]] * 8)
        prediction = {"predictions": df, "model_label": "m", "model_key": "global", "fallback_reason": None,
                      "conditions": {"surface": "芝", "distance": 1600, "going": "良"}}
        prompt = build_gami_free_prompt("202605050311", {"name": "R"}, prediction, None, _market_odds(live), 3000,
                                        affinity={"predictions": aff})
        self.assertIn("TYPE: 血統穴馬狙い", prompt)
        self.assertIn("【2モデルの評価ギャップ】", prompt)
        self.assertIn("8|馬8|8位", prompt)
        self.assertIn("なぜこの券種を先頭にしたのか", prompt)
        self.assertIn("[TYPE: 血統穴馬狙い]", prompt)


if __name__ == "__main__":
    unittest.main()

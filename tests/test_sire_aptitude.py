import unittest

import numpy as np
import pandas as pd

from model import affinity_model as am
from model import sire_aptitude as sa


def _runs(rows):
    df = pd.DataFrame(rows, columns=["_horse", "_date", "_rank", "_field", "_dist", "_surf", "_place", "_going",
                                     "_margin", "_weight", "_wchange", "sire", "bms"])
    df["_date"] = pd.to_datetime(df["_date"])
    return am._finish_runs(df)


def _history(n=80):
    """父S1: 芝(0)はよく走り(3着内 60%)、ダート(1)は走らない(10%)。父S2: 芝もダートも 30%。すべて東京(5)・良。"""
    rows = []
    for i in range(n):
        rows.append([f"a{i}", "2024-01-01", 1 if i % 10 < 6 else 9, 10, 1600, 0, 5, 0, 0, 480, 0, "S1", "B1"])
        rows.append([f"b{i}", "2024-01-01", 1 if i % 10 < 1 else 9, 10, 1600, 1, 5, 0, 0, 480, 0, "S1", "B1"])
        rows.append([f"c{i}", "2024-01-01", 1 if i % 10 < 3 else 9, 10, 1600, 0, 5, 0, 0, 480, 0, "S2", "B2"])
        rows.append([f"d{i}", "2024-01-01", 1 if i % 10 < 3 else 9, 10, 1600, 1, 5, 0, 0, 480, 0, "S2", "B2"])
    return rows


CURRENT = [
    ["X", "2024-02-01", np.nan, 10, 1600, 0, 5, 0, np.nan, 480, 0, "S1", "B1"],  # S1 の芝
    ["Y", "2024-02-01", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, "S1", "B1"],  # S1 のダート
    ["Z", "2024-02-01", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, "S2", "B2"],  # S2 のダート
]


class AsofAptitudeTest(unittest.TestCase):
    def test_strong_and_weak_conditions_are_judged_against_own_level(self):
        runs = _runs(_history() + CURRENT)
        apt = sa.asof_aptitude(runs).iloc[-3:]
        x, y, z = apt.iloc[0], apt.iloc[1], apt.iloc[2]
        self.assertEqual(x["父_芝ダ_判定"], 1)
        self.assertEqual(y["父_芝ダ_判定"], -1)
        self.assertGreater(x["父_芝ダ_z値"], sa.STRONG_Z)
        self.assertLess(y["父_芝ダ_z値"], sa.WEAK_Z)
        self.assertEqual(z["父_芝ダ_判定"], 0)  # 芝もダートも同じくらい走る種牡馬に得意・苦手はない
        self.assertEqual(x["父_芝ダ_出走数"], 80)

    def test_only_runs_before_the_day_are_counted(self):
        # 同じ日の走は数えない → 履歴と同じ日付の行には何も付かない
        runs = _runs(_history())
        apt = sa.asof_aptitude(runs)
        self.assertTrue(apt["父_芝ダ_出走数"].eq(0).all())
        self.assertTrue(apt["父_芝ダ_z値"].isna().all())
        self.assertTrue(apt["父_芝ダ_判定"].eq(0).all())

    def test_few_runs_are_not_judged(self):
        runs = _runs(_history(n=sa.MIN_RUNS // 2) + CURRENT)
        apt = sa.asof_aptitude(runs).iloc[-3:]
        self.assertTrue(apt["父_芝ダ_判定"].eq(0).all())

    def test_fast_path_for_prediction_matches_full_computation(self):
        # 履歴を2日に分けて、日付ごとの累積と単純な合計が同じになることを確かめる
        hist = _history()
        for row in hist[::3]:
            row[1] = "2023-12-01"
        runs = _runs(hist + CURRENT)
        full = sa.asof_aptitude(runs).iloc[-3:]
        fast = sa.asof_aptitude(sa.add_expected(runs), targets=runs.index[-3:])
        pd.testing.assert_frame_equal(fast, full)
        # 予想時の組み立て: 期待値を付けた過去の走 + 何も付いていない今回の出走
        hist_only = sa.add_expected(runs.iloc[:-3])
        mixed = pd.concat([runs.iloc[-3:], hist_only])
        pd.testing.assert_frame_equal(sa.asof_aptitude(mixed, targets=runs.index[-3:]), full)
        with self.assertRaises(ValueError):
            sa.asof_aptitude(runs, targets=runs.index[:3])

    def test_notes_name_the_condition_like_the_summary_csv(self):
        runs = _runs(_history() + CURRENT)
        apt = sa.asof_aptitude(runs).iloc[-3:]
        notes = sa.aptitude_notes(apt, runs)
        self.assertTrue(any(n.startswith("父 得意: 芝(") for n in notes[0]), notes[0])
        self.assertTrue(any(n.startswith("父 苦手: ダート(") for n in notes[1]), notes[1])
        self.assertFalse(any("芝ダ" in n for n in notes[2]))


class AptitudeFeaturesTest(unittest.TestCase):
    def test_features_follow_the_judgement(self):
        runs = _runs(_history() + CURRENT)
        ped = am.pedigree_features(runs, style="weighted").iloc[-3:]
        self.assertGreater(ped.iloc[0]["血統_父_芝ダ_適性z"], 0)
        self.assertLess(ped.iloc[1]["血統_父_芝ダ_適性z"], 0)
        self.assertGreater(ped.iloc[0]["血統_適性判定"], 0)
        self.assertLess(ped.iloc[1]["血統_適性判定"], 0)

    def test_gated_style_drops_aptitude_for_experienced_horse_in_known_conditions(self):
        past = [["Y", f"2024-01-{d}", 5, 10, 1600, 1, 5, 0, 1.0, 480, 0, "S1", "B1"] for d in (10, 20, 30)]
        runs = _runs(_history() + past + [CURRENT[1]])
        cur = am.pedigree_features(runs, style="gated").iloc[-1]
        self.assertTrue(np.isnan(cur["血統_父_芝ダ_適性z"]))
        self.assertTrue(np.isnan(cur["血統_適性判定"]))

    def test_aptitude_columns_can_be_left_out(self):
        self.assertFalse(set(am.feature_columns("weighted", aptitude=False)) & set(am.APTITUDE_FEATURES))
        self.assertTrue(set(am.APTITUDE_FEATURES) <= set(am.feature_columns("gated")))


if __name__ == "__main__":
    unittest.main()

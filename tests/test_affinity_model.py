import unittest

import numpy as np
import pandas as pd

from model import affinity_model as am


def _runs(rows):
    df = pd.DataFrame(rows, columns=["_horse", "_date", "_rank", "_field", "_dist", "_surf", "_place", "_going",
                                     "_margin", "_weight", "_wchange", "sire", "bms"])
    df["_date"] = pd.to_datetime(df["_date"])
    return am._finish_runs(df)


class PriorSumsTest(unittest.TestCase):
    def test_only_runs_before_the_day_are_counted(self):
        runs = _runs([
            ["A", "2024-01-01", 1, 10, 1600, 0, 5, 0, -0.2, 480, 0, "S1", "B1"],
            ["B", "2024-01-01", 2, 10, 1600, 0, 5, 0, 0.2, 470, 0, "S1", "B1"],  # 同じ日の産駒は数えない
            ["C", "2024-02-01", np.nan, 12, 1600, 0, 5, 0, np.nan, 476, 2, "S1", "B2"],  # 今回の出走(結果なし)
        ])
        s = am._prior_sums(runs, ["sire", "_surf"], ["n", "win", "top3"])
        self.assertEqual(s.loc[0].tolist(), [0, 0, 0])
        self.assertEqual(s.loc[1].tolist(), [0, 0, 0])
        self.assertEqual(s.loc[2].tolist(), [2, 1, 2])

    def test_missing_key_gives_nan(self):
        runs = _runs([["A", "2024-01-01", 1, 10, 1600, 0, 5, 0, 0, 480, 0, np.nan, np.nan]])
        self.assertTrue(am._prior_sums(runs, ["sire", "_surf"], ["n"]).isna().all().all())

    def test_horse_and_weight_features_use_past_runs_of_same_horse(self):
        runs = _runs([
            ["A", "2024-01-01", 1, 10, 1600, 0, 5, 0, -0.2, 480, 0, None, None],
            ["A", "2024-03-01", 8, 10, 2400, 0, 5, 0, 1.5, 500, 20, None, None],
            ["A", "2024-05-01", np.nan, 10, 1600, 0, 5, 0, np.nan, 490, -10, None, None],
        ])
        horse = am.horse_affinity_features(runs)
        # 今回(1600m)と同じ距離帯の過去走は1/1の1着だけ → 縮約しても全体平均より高い
        self.assertGreater(horse.loc[2, "戦績_同距離帯_3着内率"], am.PRIOR["top3"])
        self.assertEqual(horse.loc[2, "戦績_同距離帯_出走数"], 1)
        self.assertEqual(horse.loc[2, "戦績_好走距離差"], 0)
        weight = am.weight_features(runs)
        self.assertEqual(weight.loc[2, "体重_対好走時"], 10)  # 好走時480kg → 今回490kg
        self.assertEqual(weight.loc[2, "体重_対平均"], 0)  # 平均490kg
        self.assertAlmostEqual(weight.loc[2, "体重変化率"], -10 / 500)


class RaceSoftmaxTest(unittest.TestCase):
    def test_sums_to_one_per_race(self):
        q = am.race_softmax([0.1, 0.3, 0.05, 0.5, 0.2], pd.Series(["r1", "r1", "r1", "r2", "r2"]), 0.9)
        self.assertAlmostEqual(q[:3].sum(), 1.0)
        self.assertAlmostEqual(q[3:].sum(), 1.0)
        self.assertGreater(q[1], q[0])


class JraOnlyTest(unittest.TestCase):
    def test_local_races_are_dropped(self):
        results = pd.DataFrame({"開催": ["5東京8", "盛岡", "2水沢3", "1中京2"], "着順": [1, 1, 2, 3]})
        out = am._jra_only([{"pedigree": {}, "results": results}])
        self.assertEqual(out[0]["results"]["開催"].tolist(), ["5東京8", "1中京2"])


if __name__ == "__main__":
    unittest.main()

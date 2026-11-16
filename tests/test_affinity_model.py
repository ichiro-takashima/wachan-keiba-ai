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


class PedigreeAptitudeTest(unittest.TestCase):
    def _sire_runs(self, current_horse_rows):
        # 父S1: 芝(0)は 10戦0好走、ダート(1)は 10戦10好走 → 全体50%、ダート適性はプラス・芝適性はマイナス
        rows = []
        for i in range(10):
            rows.append([f"t{i}", "2024-01-01", 9, 10, 1600, 0, 5, 0, 2.0, 480, 0, "S1", "B1"])
            rows.append([f"d{i}", "2024-01-01", 1, 10, 1600, 1, 5, 0, 0.0, 480, 0, "S1", "B1"])
        return _runs(rows + current_horse_rows)

    def test_aptitude_is_relative_to_sires_own_level(self):
        runs = self._sire_runs([
            ["X", "2024-02-01", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, "S1", "B1"],  # ダート
            ["Y", "2024-02-01", np.nan, 10, 1600, 0, 5, 0, np.nan, 480, 0, "S1", "B1"],  # 芝
        ])
        ped = am.pedigree_features(runs)
        dirt, turf = ped.iloc[-2], ped.iloc[-1]
        self.assertGreater(dirt["血統_父_芝ダ_適性"], 0)
        self.assertLess(turf["血統_父_芝ダ_適性"], 0)
        # 実力(全体成績)は条件によらず同じ
        self.assertAlmostEqual(dirt["血統_父_全体_3着内率"], turf["血統_父_全体_3着内率"])

    def _experienced_horse(self, current_row):
        # 馬X: ダート1600m・東京(5)・良を3走したあと、current_row を走る
        past = [["X", f"2024-01-{d}", 5, 10, 1600, 1, 5, 0, 1.0, 480, 0, "S1", "B1"] for d in (10, 20, 30)]
        return am.pedigree_features(self._sire_runs(past + [current_row])).iloc[-1]

    def test_experienced_horse_in_same_conditions_uses_no_pedigree(self):
        cur = self._experienced_horse(["X", "2024-02-10", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, "S1", "B1"])
        self.assertTrue(cur.isna().all(), cur[cur.notna()].to_dict())

    def test_experienced_horse_uses_only_untried_aptitudes(self):
        # 初めての距離帯(2400m)・初めての競馬場(阪神=9)。芝ダ・馬場状態は経験済み
        cur = self._experienced_horse(["X", "2024-02-10", np.nan, 10, 2400, 1, 9, 0, np.nan, 480, 0, "S1", "B1"])
        self.assertTrue(np.isnan(cur["血統_父_全体_3着内率"]))  # 実力はキャリアの浅い馬だけ
        self.assertTrue(np.isnan(cur["血統_父_芝ダ_適性"]))
        self.assertTrue(np.isnan(cur["血統_父_馬場状態_適性"]))
        self.assertFalse(np.isnan(cur["血統_父_距離帯_適性"]))
        self.assertFalse(np.isnan(cur["血統_父_競馬場_適性"]))

    def test_weighted_style_uses_pedigree_for_everyone_with_decaying_weight(self):
        past = [["X", f"2024-01-{d}", 5, 10, 1600, 1, 5, 0, 1.0, 480, 0, "S1", "B1"] for d in (10, 20, 30)]
        runs = self._sire_runs(past + [["X", "2024-02-10", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, "S1", "B1"]])
        cur = am.pedigree_features(runs, style="weighted").iloc[-1]
        self.assertEqual(set(cur.index), set(am.PEDIGREE_FEATURES_BY_STYLE["weighted"]) - {"血統_父_距離帯_適性_対平均"})
        self.assertFalse(np.isnan(cur["血統_父_全体_3着内率"]))  # 経験十分でも使う
        # ダートを3回走っている → 効きは 3/(3+3) = 0.5
        self.assertAlmostEqual(cur["血統_父_芝ダ_適性_未経験"], cur["血統_父_芝ダ_適性"] * 0.5)

    def test_none_style_has_no_pedigree_columns(self):
        runs = self._sire_runs([["X", "2024-02-01", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, "S1", "B1"]])
        self.assertEqual(am.pedigree_features(runs, style="none").shape[1], 0)
        self.assertFalse(set(am.feature_columns("none")) & set(am.PEDIGREE_FEATURES))
        self.assertEqual(set(am.feature_columns("gated")) & set(am.PEDIGREE_FEATURES),
                         set(am.PEDIGREE_FEATURES_BY_STYLE["gated"]))

    def test_experience_segment(self):
        runs = _runs([
            ["A", "2024-01-01", 1, 10, 1600, 0, 5, 0, 0, 480, 0, None, None],
            ["A", "2024-02-01", 1, 10, 1600, 0, 5, 0, 0, 480, 0, None, None],
            ["A", "2024-03-01", 1, 10, 1600, 0, 5, 0, 0, 480, 0, None, None],
            ["A", "2024-04-01", np.nan, 10, 1600, 1, 5, 0, np.nan, 480, 0, None, None],  # 初ダート
            ["A", "2024-04-01", np.nan, 10, 1600, 0, 5, 0, np.nan, 480, 0, None, None],  # 同条件4戦目
        ])
        seg = am.experience_segment(runs)
        self.assertEqual(seg.iloc[0], am.SEGMENTS[0])
        self.assertEqual(seg.iloc[3], am.SEGMENTS[1])
        self.assertEqual(seg.iloc[4], am.SEGMENTS[2])


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

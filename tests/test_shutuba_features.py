"""出馬表モデル(model.shutuba_model)の近走特徴量のテスト。

実行: python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np
import pandas as pd

from model.shutuba_model import (
    ALL_FEATURE_COLUMNS,
    FEATURE_GROUPS,
    add_race_relative_features,
    base_features,
    entry_history_features,
    training_features,
)

HISTORY_COLUMNS = [
    "前走着順", "近走平均着順", "近走3着内率", "前走間隔",
    *FEATURE_GROUPS["近走の人気・着順率"], *FEATURE_GROUPS["条件替わり"], *FEATURE_GROUPS["タイム差・位置取り"],
]

# (race_id, 日付, 芝・ダート, 距離, クラス, [(馬名, 着順, タイム, 通過, 上り, 人気, 単勝, 斤量)])
RACES = [
    ("r1", "2025/01/05", "芝", 1600.0, "3歳未勝利 (馬齢)", [
        ("A", "2", "1:35.4", "5-4", "34.8", "3", "6.5", "56"),
        ("B", "1", "1:35.1", "1-1", "35.2", "1", "2.1", "56"),
        ("C", "3", "1:36.0", "2-2", "36.1", "2", "3.9", "54"),
    ]),
    ("r2", "2025/02/09", "ダート", 1800.0, "3歳未勝利 (馬齢)", [
        ("A", "1", "1:53.0", "2-2-2-1", "37.5", "1", "1.8", "57"),
        ("C", "中", None, "3-3", None, "2", "4.0", "54"),
        ("D", "2", "1:53.6", "1-1-1-2", "38.3", "3", "9.9", "57"),
    ]),
    ("r3", "2025/03/16", "ダート", 1400.0, "3歳1勝クラス (馬齢)", [
        ("A", "3", "1:25.2", "4-4", "36.9", "2", "3.3", "55"),
        ("B", "1", "1:24.5", "1-1", "36.6", "1", "2.0", "57"),
        ("D", "1", "1:24.5", "2-2", "36.5", "3", "8.0", "57"),
    ]),
    ("r4", "2025/04/20", "芝", 2000.0, "3歳1勝クラス (馬齢)", [
        ("A", "1", "2:00.1", "3-3-2-2", "34.0", "1", "2.4", "57"),
        ("B", "2", "2:00.3", "1-1-1-1", "34.5", "2", "3.0", "57"),
        ("E", "3", "2:01.0", "2-2-3-3", "35.0", "3", "7.7", "55"),
    ]),
]


def _rows(races=RACES):
    rows = []
    for race_id, date, surface, distance, race_class, runners in races:
        for umaban, (horse, rank, time, passing, last3f, pop, odds, weight) in enumerate(runners, start=1):
            rows.append({
                "race_id": race_id, "日付": date, "馬名": horse, "騎手": f"騎手{umaban}", "枠番": umaban,
                "馬番": umaban, "性齢": "牡3", "斤量": weight, "馬体重": "480(+2)", "単勝": odds, "人気": pop,
                "着順": rank, "タイム": time, "通過": passing, "上り": last3f, "芝・ダート": surface,
                "距離": distance, "回り": "右", "馬場": "良", "天気": "晴", "場id": "05", "クラス": race_class,
                "単勝払戻": 0, "複勝払戻": 0,
            })
    return pd.DataFrame(rows)


def _features(rows):
    df, feats = training_features(rows)
    feats.index = pd.MultiIndex.from_frame(df[["race_id", "馬名"]])
    return feats


# 馬Aの戦績ページ相当(netkeibaの馬の成績表。着差は勝ち馬との秒差で、勝ち馬は2着との差が負で入る)
RESULTS_PAGE_A = pd.DataFrame({
    "日付": ["2025/04/20", "2025/03/16", "2025/02/09", "2025/01/05"],
    "レース名": ["3歳1勝クラス", "3歳1勝クラス", "3歳未勝利", "3歳未勝利"],
    "頭数": [3, 3, 3, 3],
    "オッズ": [2.4, 3.3, 1.8, 6.5],
    "人気": [1, 2, 1, 3],
    "着順": [1, 3, 1, 2],
    "斤量": [57, 55, 57, 56],
    "距離": ["芝2000", "ダ1400", "ダ1800", "芝1600"],
    "着差": [-0.2, 0.7, -0.6, 0.3],
    "通過": ["3-3-2-2", "4-4", "2-2-2-1", "5-4"],
    "上り": [34.0, 36.9, 37.5, 34.8],
})


class HorseHistoryFeatureTest(unittest.TestCase):
    def setUp(self):
        self.feats = _features(_rows())

    def test_all_candidate_columns_exist(self):
        self.assertFalse(set(ALL_FEATURE_COLUMNS) - set(self.feats.columns))

    def test_first_run_has_no_history(self):
        first = self.feats.loc[("r1", "A")]
        self.assertEqual(first["近走数"], 0)
        self.assertTrue(first[[c for c in HISTORY_COLUMNS if c != "近走数"]].isna().all())

    def test_values_come_from_previous_runs(self):
        a = self.feats.loc[("r4", "A")]  # 過去走: r1 2着(+0.3秒), r2 1着(-0.6秒), r3 3着(+0.7秒)
        self.assertEqual(a["前走着順"], 3)
        self.assertEqual(a["前走間隔"], 35)
        self.assertAlmostEqual(a["近走平均着順"], 2.0)
        self.assertAlmostEqual(a["近走3着内率"], 1.0)
        self.assertEqual(a["近走数"], 3)
        self.assertEqual(a["前走人気"], 2)
        self.assertAlmostEqual(a["前走着順率"], 1.0)
        self.assertAlmostEqual(a["近走人気超過"], ((3 - 2) + (1 - 1) + (2 - 3)) / 3)
        self.assertEqual(a["距離変化"], 600)
        self.assertEqual(a["芝ダ替わり"], 1)
        self.assertEqual(a["クラス変化"], 0)
        self.assertEqual(a["斤量変化"], 2)
        self.assertAlmostEqual(a["前走タイム差"], 0.7)
        self.assertAlmostEqual(a["近走平均タイム差"], (0.3 - 0.6 + 0.7) / 3)
        self.assertAlmostEqual(a["近走最小タイム差"], -0.6)
        self.assertAlmostEqual(a["前走上がり"], 36.9)
        self.assertAlmostEqual(a["前走4角位置率"], 4 / 3)
        self.assertAlmostEqual(a["近走平均4角位置率"], (4 / 3 + 1 / 3 + 4 / 3) / 3)

    def test_dead_heat_winners_and_non_finishers(self):
        self.assertEqual(self.feats.loc[("r4", "B"), "前走タイム差"], 0)  # r3は同着1着
        c = self.feats.loc[("r2", "C")]
        self.assertAlmostEqual(c["前走タイム差"], 0.9)
        # r2で競走中止した馬の次走: 前走着順は欠損、出走数には数える
        rows = pd.concat([_rows(), _rows([("r5", "2025/05/25", "芝", 1600.0, "3歳未勝利 (馬齢)", [
            ("C", "1", "1:34.0", "1-1", "34.0", "1", "2.0", "56"),
            ("F", "2", "1:34.5", "2-2", "34.4", "2", "3.0", "56"),
        ])])], ignore_index=True)
        c = _features(rows).loc[("r5", "C")]
        self.assertTrue(np.isnan(c["前走着順"]))
        self.assertTrue(np.isnan(c["前走タイム差"]))
        self.assertEqual(c["近走数"], 2)
        self.assertAlmostEqual(c["近走平均着順"], 3.0)

    def test_own_race_result_does_not_leak(self):
        """そのレース自身の着順・タイム・通過・上りを変えても、そのレースの行の特徴量は変わらない。"""
        changed = _rows()
        last = changed["race_id"] == "r4"
        changed.loc[last, "着順"] = ["3", "1", "2"]
        changed.loc[last, "タイム"] = ["2:05.0", "2:00.0", "2:03.0"]
        changed.loc[last, "通過"] = ["9-9-9-9", "1-1-1-1", "5-5-5-5"]
        changed.loc[last, "上り"] = ["39.9", "33.0", "36.0"]
        after = _features(changed)
        r4 = self.feats.index.get_level_values("race_id") == "r4"
        pd.testing.assert_frame_equal(after[r4][ALL_FEATURE_COLUMNS], self.feats[r4][ALL_FEATURE_COLUMNS])

    def test_prediction_path_matches_training_path(self):
        """戦績ページから作る予想時の値が、収集データから作る学習時の値と一致する。"""
        entries = _rows([("r5", "2025/05/25", "芝", 1800.0, "3歳2勝クラス (定量)", [
            ("A", None, None, None, None, "1", "2.2", "56"),
            ("NEW", None, None, None, None, "2", "5.0", "54"),
        ])])
        expected = _features(pd.concat([_rows(), entries], ignore_index=True)).loc["r5"]

        feats = base_features(entries)
        feats["騎手3着内率"] = np.nan
        history = entry_history_features(feats, [RESULTS_PAGE_A, None], pd.Timestamp("2025-05-25"))
        actual = add_race_relative_features(feats.join(history), entries["race_id"])
        actual.index = entries["馬名"]

        relative = ["斤量_対平均", "近走平均着順率_対平均", "近走平均タイム差_対平均"]
        pd.testing.assert_frame_equal(
            actual[HISTORY_COLUMNS + relative].astype(float), expected.loc[["A", "NEW"], HISTORY_COLUMNS + relative].astype(float),
            check_names=False,
        )
        self.assertEqual(actual.loc["A", "クラス変化"], 1)
        self.assertEqual(actual.loc["NEW", "近走数"], 0)

    def test_future_rows_on_results_page_are_ignored(self):
        """戦績ページに予想日以降の走(バックテスト時など)が載っていても使わない。"""
        entries = _rows([("r4", "2025/04/20", "芝", 2000.0, "3歳1勝クラス (馬齢)", [
            ("A", None, None, None, None, "1", "2.4", "57"),
        ])])
        feats = base_features(entries)
        history = entry_history_features(feats, [RESULTS_PAGE_A], pd.Timestamp("2025-04-20"))
        expected = self.feats.loc[("r4", "A"), HISTORY_COLUMNS].astype(float)
        np.testing.assert_allclose(history.iloc[0][HISTORY_COLUMNS].astype(float), expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()

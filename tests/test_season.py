"""季節適性特徴量(preprocessing.season)と特徴量パイプラインの単体・回帰テスト。

実行: python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.build_features import build_feature_dataset
from preprocessing.season import (
    SEASON_FEATURE_COLUMNS,
    add_season_aptitude,
    build_month_lookup,
    month_to_season,
    season_year,
)


def _race(race_id, date, results, popularity=None):
    """results: [(馬, 着順)]。人気は省略時に馬番順。"""
    rows = []
    for i, (horse, rank) in enumerate(results):
        rows.append({
            "race_id": race_id, "馬": horse, "日付": date, "着順": rank,
            "人気": (popularity[i] if popularity else i + 1),
        })
    return rows


def _filler(prefix, n):
    return [(f"{prefix}{i}", str(i + 2)) for i in range(n)]


def _sample_df():
    rows = []
    # 8頭立て(複勝=3着まで)。Aは春に好走・夏に凡走。
    rows += _race("r1", "2025/04/01", [("A", "1")] + _filler("x", 7))
    rows += _race("r2", "2025/07/01", _filler("y", 7) + [("A", "8")])
    rows += _race("r3", "2025/12/20", _filler("z", 7) + [("A", "8")])
    rows += _race("r4", "2026/01/10", [("A", "2")] + _filler("w", 7))
    rows += _race("r5", "2026/04/05", [("A", "3")] + _filler("v", 7))
    return pd.DataFrame(rows)


def _row(df, race_id, horse="A"):
    return df[(df["race_id"] == race_id) & (df["馬"] == horse)].iloc[0]


class SeasonMappingTest(unittest.TestCase):
    def test_default_mapping(self):
        self.assertEqual(month_to_season(3), 0)
        self.assertEqual(month_to_season(8), 1)
        self.assertEqual(month_to_season(11), 2)
        self.assertEqual(month_to_season(12), 3)
        self.assertEqual(month_to_season(2), 3)
        self.assertTrue(np.isnan(month_to_season(np.nan)))
        self.assertTrue(np.isnan(month_to_season(13)))

    def test_custom_mapping(self):
        custom = {"暖": (4, 5, 6, 7, 8, 9, 10), "寒": (11, 12, 1, 2, 3)}
        self.assertEqual(month_to_season(3, custom), 1)
        self.assertEqual(month_to_season(10, custom), 0)

    def test_invalid_mapping_rejected(self):
        with self.assertRaises(ValueError):
            build_month_lookup({"a": (1, 2, 3), "b": (3, 4)})  # 重複・欠落
        with self.assertRaises(ValueError):
            build_month_lookup({"a": tuple(range(1, 12))})  # 12月欠落

    def test_winter_season_year_wraps(self):
        self.assertEqual(season_year(2025, 12), 2025)
        self.assertEqual(season_year(2026, 1), 2025)
        self.assertEqual(season_year(2026, 3), 2026)


class SeasonAptitudeTest(unittest.TestCase):
    def setUp(self):
        self.df = _sample_df()
        self.out = add_season_aptitude(self.df, prior_strength=3.0)

    def test_columns_and_order_preserved(self):
        for col in SEASON_FEATURE_COLUMNS:
            self.assertIn(col, self.out.columns)
        pd.testing.assert_frame_equal(self.out[self.df.columns], self.df)

    def test_first_run_has_no_history(self):
        r = _row(self.out, "r1")
        self.assertEqual(r["季節_出走数"], 0)
        self.assertTrue(np.isnan(r["季節_複勝率"]))
        self.assertEqual(r["季節_信頼度"], 0)
        self.assertEqual(r["季節_複勝超過_相対"], 0)
        self.assertEqual(r["季節_人気差_相対"], 0)

    def test_uses_only_prior_same_season(self):
        r = _row(self.out, "r5")  # 2026春: 前の春は r1(1着)のみ
        self.assertEqual(r["季節"], 0)
        self.assertEqual(r["季節_出走数"], 1)
        self.assertEqual(r["季節_勝率"], 1.0)
        self.assertEqual(r["季節_複勝率"], 1.0)
        self.assertEqual(r["季節_平均着順比"], 0.0)
        self.assertTrue(np.isnan(r["季節_着順比_標準偏差"]))  # 1走では計算しない
        self.assertEqual(r["季節_前年以前出走数"], 1)

    def test_winter_block_spans_new_year(self):
        r = _row(self.out, "r4")  # 2026/01 は 2025/12 と同じ冬 → 前年以前ではない
        self.assertEqual(r["季節_出走数"], 1)
        self.assertEqual(r["季節_前年以前出走数"], 0)
        self.assertEqual(r["季節_平均着順比"], 1.0)  # 2025/12の最下位

    def test_shrinkage_limits_extremes(self):
        r = _row(self.out, "r5")
        # 1走だけの好走: 生の複勝率=1.0でも、相対差は (1-3/8) より十分小さい
        self.assertGreater(r["季節_複勝超過_相対"], 0)
        self.assertLess(r["季節_複勝超過_相対"], 0.5 * (1 - 3 / 8))
        # 通年の凡走(夏・冬)より春の方が良い → 相対差は正
        self.assertGreater(r["季節_人気差_相対"], 0)

    def test_stronger_prior_shrinks_more(self):
        weak = add_season_aptitude(self.df, prior_strength=1.0)
        strong = add_season_aptitude(self.df, prior_strength=10.0)
        self.assertGreater(abs(_row(weak, "r5")["季節_複勝超過_相対"]),
                           abs(_row(strong, "r5")["季節_複勝超過_相対"]))

    def test_own_result_does_not_leak(self):
        changed = self.df.copy()
        changed.loc[(changed["race_id"] == "r5") & (changed["馬"] == "A"), "着順"] = "8"
        a = add_season_aptitude(changed)
        pd.testing.assert_series_equal(_row(a, "r5")[SEASON_FEATURE_COLUMNS],
                                       _row(self.out, "r5")[SEASON_FEATURE_COLUMNS])

    def test_future_results_do_not_leak(self):
        future = pd.DataFrame(_race("r9", "2027/04/01", [("A", "1")] + _filler("u", 7)))
        extended = add_season_aptitude(pd.concat([self.df, future], ignore_index=True))
        base_rows = extended[extended["race_id"] != "r9"].reset_index(drop=True)
        pd.testing.assert_frame_equal(base_rows[SEASON_FEATURE_COLUMNS].reset_index(drop=True),
                                      self.out[SEASON_FEATURE_COLUMNS].reset_index(drop=True))

    def test_small_field_place_rule(self):
        # 6頭立ては複勝2着まで: 3着は複勝圏外
        rows = _race("s1", "2025/04/01", [("B", "3")] + _filler("p", 5))
        rows += _race("s2", "2025/05/01", [("B", "1")] + _filler("q", 5))
        out = add_season_aptitude(pd.DataFrame(rows))
        self.assertEqual(_row(out, "s2", "B")["季節_複勝率"], 0.0)

    def test_scratched_and_dnf(self):
        rows = _race("t1", "2025/04/01", [("C", "取")] + _filler("p", 8))
        rows += _race("t2", "2025/04/08", [("C", "中")] + _filler("q", 8))
        rows += _race("t3", "2025/04/15", [("C", "1")] + _filler("r", 8))
        out = add_season_aptitude(pd.DataFrame(rows))
        self.assertEqual(_row(out, "t2", "C")["季節_出走数"], 0)  # 取消は出走数に数えない
        r = _row(out, "t3", "C")
        self.assertEqual(r["季節_出走数"], 1)  # 中止は出走扱い・最下位
        self.assertEqual(r["季節_平均着順比"], 1.0)

    def test_invalid_dates_are_nan_and_ignored(self):
        bad = self.df.copy()
        bad.loc[0, "日付"] = "不明"
        out = add_season_aptitude(bad)
        self.assertTrue(out.loc[0, SEASON_FEATURE_COLUMNS].isna().all())
        self.assertEqual(_row(out, "r5")["季節_出走数"], 0)  # 不正日付のr1は集計に使わない

    def test_deterministic(self):
        again = add_season_aptitude(self.df.sample(frac=1, random_state=0)).sort_index()
        pd.testing.assert_frame_equal(again[SEASON_FEATURE_COLUMNS], self.out[SEASON_FEATURE_COLUMNS])


class BuildFeaturesRegressionTest(unittest.TestCase):
    """季節特徴量の追加が既存の特徴量を変えないことを確認する。"""

    def test_existing_columns_unchanged(self):
        base = _sample_df()
        base["騎手"] = "J"
        for col, val in {"馬番": 1, "走破時間": "1:35.0", "オッズ": 3.0, "通過順": "1-1", "体重": 480,
                         "体重変化": 0, "性": "牡", "齢": 3, "斤量": 57, "上がり": 34.0,
                         "レース名": "R", "開催": "1回", "クラス": "1勝クラス", "芝・ダート": "芝",
                         "距離": 1600, "回り": "右", "馬場": "良", "天気": "晴", "場id": "05",
                         "場名": "東京"}.items():
            base[col] = val
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "race_results.csv"
            base.to_csv(path, index=False, encoding="utf-8-sig")
            without = build_feature_dataset(path, include_season=False)
            with_season = build_feature_dataset(path, include_season=True)
        self.assertEqual(set(with_season.columns) - set(without.columns), set(SEASON_FEATURE_COLUMNS))
        key = ["race_id", "馬"]
        a = without.sort_values(key).reset_index(drop=True)
        b = with_season.sort_values(key).reset_index(drop=True)[without.columns]
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()

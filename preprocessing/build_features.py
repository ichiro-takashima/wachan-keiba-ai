"""race_results.csvから、学習に使える特徴量データセット(features.csv)を作る。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing import encode
from preprocessing.history import add_avg_weight_carried, add_jockey_win_rate, add_last5_history
from preprocessing.season import DEFAULT_PRIOR_STRENGTH, add_season_aptitude

LABEL_ENCODE_COLUMNS = ["馬", "騎手", "レース名", "開催", "場名"]


def _apply_base_encoding(df):
    working = df.copy()

    working["クラス"] = working["クラス"].map(encode.encode_class)
    working["走破時間"] = working["走破時間"].map(encode.parse_time_to_seconds)
    working["通過順"] = working["通過順"].map(encode.parse_corner_passage_avg)
    working["性"] = working["性"].map(encode.encode_sex)
    working["芝・ダート"] = working["芝・ダート"].map(encode.encode_surface)
    working["距離区分"] = working["距離"].map(encode.encode_distance_category)
    working["回り"] = working["回り"].map(encode.encode_turn)
    working["馬場"] = working["馬場"].map(encode.encode_going)
    working["天気"] = working["天気"].map(encode.encode_weather)

    date_parts = working["日付"].map(encode.split_date)
    working["年"] = [p[0] for p in date_parts]
    working["月"] = [p[1] for p in date_parts]
    working["日"] = [p[2] for p in date_parts]

    return working


def _add_interaction_features(df):
    """芝・ダート/距離区分と既存特徴量の交互作用を明示的な列として追加する(アプローチ2)。

    LightGBMは決定木ベースで交互作用をある程度自動で拾うが、木の分岐に任せきりだと
    弱い交互作用は拾いきれないため、course_type×distance_categoryや
    脚質の代理指標(直近走の通過順平均)との交互作用を手動で用意する。
    """
    working = df.copy()
    working["芝ダート_距離区分"] = working["芝・ダート"] * 10 + working["距離区分"]
    working["通過順1_距離区分"] = working["通過順_1"] * working["距離区分"]
    working["通過順1_芝ダート"] = working["通過順_1"] * working["芝・ダート"]

    # 先行有利フラグ: 短距離(0)・長距離(3)は先行有利、マイル(1)・中距離(2)は先行有利度が弱まる
    # (通過順_1とtop3の相関が中距離で最も弱まりU字型に回復する傾向が実データで確認できたため)。
    working["距離帯_先行有利"] = working["距離区分"].isin([0, 3]).astype(float)
    working.loc[working["距離区分"].isna(), "距離帯_先行有利"] = np.nan
    working["通過順1_先行有利"] = working["通過順_1"] * working["距離帯_先行有利"]
    return working


def _label_encode(df, columns):
    working = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        working[f"{col}_id"] = le.fit_transform(working[col].fillna("不明").astype(str))
        encoders[col] = le
    return working, encoders


def build_feature_dataset(
    results_csv="data/processed/race_results.csv",
    include_season=True,
    season_months=None,
    season_prior_strength=DEFAULT_PRIOR_STRENGTH,
):
    """race_results.csvを読み込み、リーク防止済みの特徴量データセットを返す。

    include_season=Trueなら季節適性の特徴量(preprocessing.season)も追加する。
    season_monthsで季節の月割りを変更できる(既定は春3-5/夏6-8/秋9-11/冬12-2)。
    """
    df = pd.read_csv(results_csv, encoding="utf-8-sig", dtype={"race_id": str})

    # 季節適性は生の着順・人気・日付文字列から計算するため、エンコード前に追加する。
    if include_season:
        df = add_season_aptitude(df, season_months=season_months, prior_strength=season_prior_strength)

    df = _apply_base_encoding(df)
    df = add_last5_history(df)
    df = add_jockey_win_rate(df)
    df = add_avg_weight_carried(df)
    df = _add_interaction_features(df)
    df, _ = _label_encode(df, LABEL_ENCODE_COLUMNS)

    df["着順_数値"] = df["着順"].map(encode.parse_rank)
    df["top3"] = (df["着順_数値"] <= 3).astype("Int64")
    df.loc[df["着順_数値"].isna(), "top3"] = pd.NA

    # 穴馬モデル(Step5)用の目的変数: オッズ15倍以上 かつ 3着以内 = 1、それ以外(3着以内でも
    # 15倍未満、または4着以下)= 0。着順が数値化できない行(中止・除外等)はtop3と同様NaN。
    odds_numeric = pd.to_numeric(df["オッズ"], errors="coerce")
    df["top3_anauma"] = ((odds_numeric >= 15) & (df["top3"] == 1)).astype("Int64")
    df.loc[df["着順_数値"].isna(), "top3_anauma"] = pd.NA

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="race_results.csvから特徴量データセットを作成する")
    parser.add_argument("--results-csv", default="data/processed/race_results.csv")
    parser.add_argument("--output-csv", default="data/processed/features.csv")
    parser.add_argument("--no-season", action="store_true", help="季節適性の特徴量を追加しない")
    parser.add_argument("--season-prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH,
                        help="季節適性の平滑化の強さ(擬似出走数)")
    args = parser.parse_args()

    dataset = build_feature_dataset(
        args.results_csv,
        include_season=not args.no_season,
        season_prior_strength=args.season_prior_strength,
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"features.csv 生成完了: {dataset.shape[0]}行 / {dataset.shape[1]}列 -> {args.output_csv}")


if __name__ == "__main__":
    main()

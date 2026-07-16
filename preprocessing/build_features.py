"""race_results.csvから、学習に使える特徴量データセット(features.csv)を作る。"""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing import encode
from preprocessing.history import add_avg_weight_carried, add_jockey_win_rate, add_last5_history

LABEL_ENCODE_COLUMNS = ["馬", "騎手", "レース名", "開催", "場名"]


def _apply_base_encoding(df):
    working = df.copy()

    working["クラス"] = working["クラス"].map(encode.encode_class)
    working["走破時間"] = working["走破時間"].map(encode.parse_time_to_seconds)
    working["通過順"] = working["通過順"].map(encode.parse_corner_passage_avg)
    working["性"] = working["性"].map(encode.encode_sex)
    working["芝・ダート"] = working["芝・ダート"].map(encode.encode_surface)
    working["回り"] = working["回り"].map(encode.encode_turn)
    working["馬場"] = working["馬場"].map(encode.encode_going)
    working["天気"] = working["天気"].map(encode.encode_weather)

    date_parts = working["日付"].map(encode.split_date)
    working["年"] = [p[0] for p in date_parts]
    working["月"] = [p[1] for p in date_parts]
    working["日"] = [p[2] for p in date_parts]

    return working


def _label_encode(df, columns):
    working = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        working[f"{col}_id"] = le.fit_transform(working[col].fillna("不明").astype(str))
        encoders[col] = le
    return working, encoders


def build_feature_dataset(results_csv="data/processed/race_results.csv"):
    """race_results.csvを読み込み、リーク防止済みの特徴量データセットを返す。"""
    df = pd.read_csv(results_csv, encoding="utf-8-sig", dtype={"race_id": str})

    df = _apply_base_encoding(df)
    df = add_last5_history(df)
    df = add_jockey_win_rate(df)
    df = add_avg_weight_carried(df)
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
    args = parser.parse_args()

    dataset = build_feature_dataset(args.results_csv)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"features.csv 生成完了: {dataset.shape[0]}行 / {dataset.shape[1]}列 -> {args.output_csv}")


if __name__ == "__main__":
    main()

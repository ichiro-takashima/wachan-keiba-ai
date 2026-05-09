import pandas as pd
import streamlit as st


def find_matching_column(df, candidates):
    """DataFrameから候補リストに一致する最初の列名を見つける"""
    if df is None:
        return None

    for col in candidates:
        if col in df.columns:
            return col

    for col in candidates:
        clean_col = col.replace(" ", "").replace("　", "").replace("\n", "").lower()
        for actual_col in df.columns:
            col_str = str(actual_col[-1] if isinstance(actual_col, tuple) else actual_col)
            clean_actual = col_str.replace(" ", "").replace("　", "").replace("\n", "").lower()
            if clean_col == clean_actual:
                return actual_col

    for col in candidates:
        clean_col = col.replace(" ", "").replace("　", "").replace("\n", "").lower()
        for actual_col in df.columns:
            col_str = str(actual_col[-1] if isinstance(actual_col, tuple) else actual_col)
            clean_actual = col_str.replace(" ", "").replace("　", "").replace("\n", "").lower()
            if clean_col in clean_actual:
                return actual_col
    return None


def get_available_columns(df, candidate_map):
    """
    候補名の中から実在する列だけを重複なく返す。
    「Unnamed」列であっても、中身が重要そうな場合は予備として含める。
    """
    selected_cols = []
    for candidates in candidate_map:
        col = find_matching_column(df, candidates)
        if col and col not in selected_cols:
            selected_cols.append(col)

    for col in df.columns:
        col_str = str(col)
        if "Unnamed" in col_str and col not in selected_cols:
            if df[col].notna().any():
                selected_cols.append(col)

    return selected_cols


AI_DATA_FEATURES = [
    ("race_id", "レース基本: レースID"),
    ("race_name", "レース基本: レース名"),
    ("race_condition", "レース基本: コース・距離・天候"),
    ("past_trend_rows", "過去傾向: 過去5年の結果表"),
    ("past_trend_summary", "過去傾向: 勝ち馬人気・馬番傾向"),
    ("shutuba_umaban", "出馬表: 馬番"),
    ("shutuba_waku", "出馬表: 枠番"),
    ("shutuba_name", "出馬表: 馬名"),
    ("shutuba_sex_age", "出馬表: 性齢"),
    ("shutuba_jockey", "出馬表: 騎手"),
    ("shutuba_weight_carried", "出馬表: 斤量"),
    ("shutuba_odds", "出馬表: オッズ"),
    ("shutuba_popularity", "出馬表: 人気"),
    ("shutuba_stable", "出馬表: 厩舎"),
    ("shutuba_body_weight", "出馬表: 馬体重"),
    ("blood_sire", "血統: 父"),
    ("blood_dam", "血統: 母"),
    ("blood_broodmare_sire", "血統: 母父"),
    ("analysis_running_style", "分析: 脚質"),
    ("analysis_track_preference", "分析: 馬場適性"),
    ("analysis_latest_weight", "分析: 近走馬体重"),
    ("result_date", "過去戦績: 日付"),
    ("result_race_name", "過去戦績: レース名"),
    ("result_place", "過去戦績: 開催場"),
    ("result_rank", "過去戦績: 着順"),
    ("result_popularity", "過去戦績: 人気"),
    ("result_jockey", "過去戦績: 騎手"),
    ("result_weight_carried", "過去戦績: 斤量"),
    ("result_distance", "過去戦績: 距離"),
    ("result_track", "過去戦績: 馬場"),
    ("result_time", "過去戦績: タイム"),
    ("result_last3f", "過去戦績: 上り3F"),
    ("result_pace", "過去戦績: ペース"),
    ("result_corner", "過去戦績: 通過順"),
    ("result_body_weight", "過去戦績: 馬体重"),
]


def render_ai_data_feature_selector(key_prefix, caption):
    st.markdown("#### ⚙️ AIへ渡すデータ要素の選択 (過学習・ノイズ防止)")
    st.caption(caption)

    categories = {}
    for key, label in AI_DATA_FEATURES:
        parts = label.split(": ", 1)
        cat = parts[0] if len(parts) > 1 else "その他"
        name = parts[1] if len(parts) > 1 else label
        categories.setdefault(cat, []).append((key, name))

    selected = set()
    for cat, items in categories.items():
        with st.expander(f"📁 {cat}", expanded=False):
            cols = st.columns(3)
            for i, (key, name) in enumerate(items):
                with cols[i % 3]:
                    if st.checkbox(name, value=True, key=f"{key_prefix}_feature_{key}"):
                        selected.add(key)

    st.caption(f"{len(selected)} / {len(AI_DATA_FEATURES)} 項目を送信")
    return selected


def feature_enabled(selected_features, *keys):
    return any(key in selected_features for key in keys)


def get_selected_shutuba_df(shutuba_df, selected_features):
    if shutuba_df is None or shutuba_df.empty:
        return pd.DataFrame()

    feature_columns = [
        ("shutuba_umaban", ["馬番"]),
        ("shutuba_waku", ["枠", "枠番"]),
        ("shutuba_name", ["馬名"]),
        ("shutuba_sex_age", ["性齢", "性令"]),
        ("shutuba_jockey", ["騎手"]),
        ("shutuba_weight_carried", ["斤量"]),
        ("shutuba_odds", ["オッズ", "単勝"]),
        ("shutuba_popularity", ["人気"]),
        ("shutuba_stable", ["厩舎", "調教師"]),
        ("shutuba_body_weight", ["馬体重"]),
    ]
    candidate_map = [candidates for key, candidates in feature_columns if key in selected_features]
    cols = get_available_columns(shutuba_df, candidate_map)
    return shutuba_df[cols] if cols else pd.DataFrame()


def get_selected_result_columns(results_df, selected_features):
    feature_columns = [
        ("result_date", ["日付"]),
        ("result_race_name", ["レース名", "レース"]),
        ("result_place", ["開催"]),
        ("result_rank", ["着順", "着"]),
        ("result_popularity", ["人気"]),
        ("result_jockey", ["騎手"]),
        ("result_weight_carried", ["斤量"]),
        ("result_distance", ["距離"]),
        ("result_track", ["馬場", "馬場状態"]),
        ("result_time", ["タイム"]),
        ("result_last3f", ["上り", "上がり", "上り3F", "上がり3F"]),
        ("result_pace", ["ペース", "ﾍﾟｰｽ"]),
        ("result_corner", ["通過", "コーナー通過順", "コーナー"]),
        ("result_body_weight", ["馬体重"]),
    ]
    candidate_map = [candidates for key, candidates in feature_columns if key in selected_features]
    return get_available_columns(results_df, candidate_map)

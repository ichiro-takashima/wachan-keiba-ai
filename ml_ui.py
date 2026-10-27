"""カテゴリ別LightGBMモデルの予測結果を表示するStreamlit UI部品。

app.pyの単一レース予想画面で、
  Step1: レース概況・適用モデルバッジ
  Step2: AI推奨馬ランキング(上位5頭のカード)
  Step3: 馬券戦略・買い目カード + 資金配分シミュレータ
  詳細分析: 出走表全データ / 特徴量重要度 / 過去走ペース比較
の順に表示する。
"""
import json
import re
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

from model.categories import CATEGORY_KEYS, CATEGORY_LABELS
from model.live_predict import CATEGORY_MODEL_DIR, GLOBAL_MODEL_PATH, load_category_comparison

MARK_LABELS = {"◎": "本命", "○": "対抗", "▲": "単穴", "△": "連下", "☆": "穴"}
VOLATILITY_COLORS = {"堅い": "green", "標準": "blue", "荒れ気味": "orange", "大荒れ警戒": "red"}
VOLATILITY_ICONS = {"堅い": "🛡️", "標準": "⚖️", "荒れ気味": "⚠️", "大荒れ警戒": "🌪️"}
GRADE_COLORS = {"S": "red", "A": "orange", "B": "blue", "C": "gray", "-": "gray"}
# 荒れ具合に応じて、軸(◎)から流す相手の頭数を増減する。
PARTNER_COUNT_BY_LEVEL = {"堅い": 3, "標準": 4, "荒れ気味": 5, "大荒れ警戒": 6}
ALLOCATION_METHODS = {
    "推奨度比例": "推定勝率が高い馬ほど厚く買う",
    "期待値重視(ケリー基準)": "期待値が1を超える馬だけに、優位性の大きさに応じて配分する",
    "払戻均一": "どの馬が勝っても払戻額がほぼ同じになるよう、オッズに反比例して配分する",
}


def _fmt_pct(value):
    return "-" if pd.isna(value) else f"{value * 100:.1f}%"


def _fmt_odds(value):
    return "-" if pd.isna(value) else f"{value:.1f}倍"


def _horse_label(row):
    number = "-" if pd.isna(row["馬番"]) else int(row["馬番"])
    return f"{number} {row['馬名']}"


# ---------------------------------------------------------------- Step1

def _model_metrics(model_key):
    path = CATEGORY_MODEL_DIR / f"{model_key}.metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_race_overview(race_info, prediction, volatility):
    """Step1: レース名・条件と、馬場/距離・使用モデル・波乱度のバッジを表示する。"""
    st.markdown("### 【Step 1】レース概況・適用モデル")
    if race_info:
        st.markdown(f"## 🏆 {race_info.get('name', '')}")
        meta = [race_info.get("data", "")]
        if race_info.get("entries"):
            meta.append(race_info["entries"])
        st.caption(" / ".join(m for m in meta if m))

    badge_cols = st.columns(3)
    if prediction:
        cond = prediction["conditions"]
        course = f"{cond['surface'] or '?'}{int(cond['distance']) if pd.notna(cond['distance']) else '?'}m"
        if cond.get("going"):
            course += f"（{cond['going']}）"
        badge_cols[0].badge(course, icon="🏟️", color="violet")
        badge_cols[1].badge(f"使用モデル: {prediction['model_key']} ⚡", color="primary")
    if volatility:
        level = volatility["level"]
        badge_cols[2].badge(
            f"波乱度: {level} {VOLATILITY_ICONS.get(level, '')}（{volatility['score']}/100）",
            color=VOLATILITY_COLORS.get(level, "gray"),
        )

    if not prediction:
        return
    st.caption(f"予測モデル: {prediction['model_label']}")
    if prediction.get("fallback_reason"):
        st.warning(f"カテゴリ別モデルを使えないため全体モデルで予測しました: {prediction['fallback_reason']}")

    metrics = _model_metrics(prediction["model_key"])
    if metrics and pd.notna(metrics.get("global_model_auc")):
        cat_auc, global_auc = metrics["auc"], metrics["global_model_auc"]
        note = "カテゴリ別モデルの方が良好" if cat_auc > global_auc else "全体モデルの方が良好"
        st.caption(
            f"検証データでの精度(AUC): カテゴリ別 {cat_auc:.3f} / 全体 {global_auc:.3f} → {note}"
            f"（学習{metrics['n_train']:,}行、評価{metrics['n_test']:,}行）"
        )
        if metrics.get("n_train", 0) < 1000:
            st.caption("⚠️ このカテゴリは学習データが少ないため、予測のぶれが大きくなりがちです。")


# ---------------------------------------------------------------- Step2

def render_top_horse_cards(pred_df, top_n=5, win_prob_method="blend"):
    """Step2: AIの推奨上位馬をカード形式で表示する。"""
    st.markdown("### 【Step 2】AI推奨馬ランキング")
    if win_prob_method == "blend":
        win_note = "推定勝率は3着内確率と単勝オッズを検証データで補正して組み合わせた値"
    else:
        win_note = "推定勝率は3着内確率を全頭合計で割った目安（オッズ未取得のため補正なし。人気薄を過大評価しやすい）"
    st.caption(
        f"順位・印はLightGBMの3着内確率の順。{win_note}、"
        "期待値は 推定勝率×単勝オッズ（1.0超で妙味あり）です。"
    )
    top = pred_df.head(top_n)
    cols = st.columns(len(top)) if len(top) else []
    for col, (_, row) in zip(cols, top.iterrows()):
        with col.container(border=True):
            mark = row["印"]
            st.markdown(f"#### {mark} {MARK_LABELS.get(mark, '')}")
            st.markdown(f"**{_horse_label(row)}**")
            st.caption(f"{row['騎手']} / {_fmt_odds(row['オッズ'])}"
                       + (f"（{int(row['人気'])}人気）" if pd.notna(row["人気"]) else ""))
            st.metric("3着内確率", _fmt_pct(row["3着内確率"]))
            st.markdown(f"推定勝率 **{_fmt_pct(row['推定勝率'])}**")
            grade = row["期待値ランク"]
            ev_text = "-" if pd.isna(row["期待値"]) else f"{row['期待値']:.2f}"
            st.badge(f"期待値 {grade}（{ev_text}）", color=GRADE_COLORS.get(grade, "gray"))
            if row["根拠"]:
                st.caption("評価ポイント: " + "・".join(row["根拠"]))
            if row.get("過去走数", 5) == 0:
                st.caption("⚠️ 過去走データなし")


# ---------------------------------------------------------------- Step3

def build_ticket_suggestions(pred_df, volatility_level=None):
    """AI順位と荒れ具合から、単勝・馬連・3連複の推奨買い目を作る。"""
    numbers = [int(n) for n in pred_df["馬番"] if pd.notna(n)]
    if not numbers:
        return []
    axis = numbers[0]
    partner_count = min(PARTNER_COUNT_BY_LEVEL.get(volatility_level, 4), len(numbers) - 1)
    partners = numbers[1:1 + partner_count]

    tansho = [str(axis)]
    second = pred_df.iloc[1] if len(pred_df) > 1 else None
    if second is not None and pd.notna(second["期待値"]) and second["期待値"] >= 1.0:
        tansho.append(str(int(second["馬番"])))

    suggestions = [
        {"券種": "単勝", "方式": "◎" + ("＋期待値の高い○" if len(tansho) > 1 else "のみ"), "買い目": tansho},
        {"券種": "馬連", "方式": f"◎軸 流し（相手{len(partners)}頭）",
         "買い目": [f"{min(axis, p)}-{max(axis, p)}" for p in partners]},
    ]
    if len(partners) >= 2:
        suggestions.append({
            "券種": "3連複", "方式": f"◎1頭軸 流し（相手{len(partners)}頭）",
            "買い目": ["-".join(str(n) for n in sorted([axis, a, b])) for a, b in combinations(partners, 2)],
        })
    return suggestions


def render_ticket_strategy(pred_df, volatility):
    """Step3: 券種ごとの推奨買い目をアコーディオンで表示する。"""
    st.markdown("### 【Step 3】馬券戦略・買い目")
    level = volatility["level"] if volatility else None
    if level:
        st.caption(f"波乱度「{level}」に合わせて、◎から流す相手を{PARTNER_COUNT_BY_LEVEL.get(level, 4)}頭にしています。")
    for i, sug in enumerate(build_ticket_suggestions(pred_df, level)):
        points = len(sug["買い目"])
        with st.expander(f"🎫 {sug['券種']}：{sug['方式']}　{points}点（各100円なら{points * 100:,}円）", expanded=(i == 0)):
            st.code("  ".join(sug["買い目"]), language="text")

    render_budget_simulator(pred_df)


def allocate_budget(pred_df, budget, method, top_k=5, unit=100):
    """単勝の資金配分を計算する。budgetを100円単位で上位top_k頭へ割り振る。"""
    df = pred_df.head(top_k).copy()
    df = df[df["オッズ"].notna() & (df["オッズ"] > 1.0)]
    if df.empty or budget < unit:
        return pd.DataFrame()

    p, odds = df["推定勝率"].fillna(0), df["オッズ"]
    if method == "期待値重視(ケリー基準)":
        weights = ((p * odds - 1) / (odds - 1)).clip(lower=0)
    elif method == "払戻均一":
        weights = 1 / odds
    else:
        weights = p
    if weights.sum() <= 0:
        return pd.DataFrame()

    units_total = int(budget // unit)
    raw = weights / weights.sum() * units_total
    units = np.floor(raw).astype(int)
    # 端数は小数部分の大きい馬から順に100円ずつ割り当てる(最大剰余法)。
    for idx in (raw - units).sort_values(ascending=False).index[: units_total - units.sum()]:
        units[idx] += 1

    df["購入額"] = units * unit
    df = df[df["購入額"] > 0].copy()
    df["的中時払戻"] = (df["購入額"] * df["オッズ"]).round(-1).astype(int)
    df["的中時収支"] = df["的中時払戻"] - int(df["購入額"].sum())
    df["期待払戻"] = (df["推定勝率"] * df["的中時払戻"]).round().astype(int)
    return df[["印", "馬番", "馬名", "オッズ", "推定勝率", "購入額", "的中時払戻", "的中時収支", "期待払戻"]]


def render_budget_simulator(pred_df):
    with st.container(border=True):
        st.markdown("#### 💴 資金配分シミュレータ（単勝）")
        c1, c2, c3 = st.columns([1, 1, 2])
        budget = c1.number_input("予算 (円)", min_value=100, step=100, value=3000, key="ml_alloc_budget")
        top_k = c2.slider("対象頭数（AI上位）", min_value=1, max_value=min(8, max(1, len(pred_df))),
                          value=min(5, max(1, len(pred_df))), key="ml_alloc_topk")
        method = c3.radio("配分方法", list(ALLOCATION_METHODS), key="ml_alloc_method", horizontal=True)
        st.caption(ALLOCATION_METHODS[method])

        alloc = allocate_budget(pred_df, budget, method, top_k=top_k)
        if alloc.empty:
            if method == "期待値重視(ケリー基準)":
                st.info("対象馬の中に期待値が1を超える馬がいないため、この方法では購入を見送る判断になります。")
            else:
                st.info("オッズが取得できていないため配分を計算できません。")
            return

        total = int(alloc["購入額"].sum())
        expected = int(alloc["期待払戻"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("合計購入額", f"{total:,}円")
        m2.metric("期待払戻(目安)", f"{expected:,}円")
        m3.metric("期待回収率(目安)", f"{expected / total * 100:.0f}%" if total else "-")
        display = alloc.copy()
        display["推定勝率"] = display["推定勝率"].map(_fmt_pct)
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption("推定勝率はモデル予測からの概算です。オッズは取得時点のもので、発走直前に変動します。")


# ---------------------------------------------------------------- 詳細分析タブ

def render_prediction_table(pred_df):
    display = pred_df.copy()
    display["3着内確率"] = display["3着内確率"].map(_fmt_pct)
    display["推定勝率"] = display["推定勝率"].map(_fmt_pct)
    display["期待値"] = display["期待値"].round(2)
    display["根拠"] = display["根拠"].map(lambda r: "・".join(r))
    st.dataframe(display, width="stretch", hide_index=True)


def _booster_importance(model_key, top_n=15):
    path = GLOBAL_MODEL_PATH if model_key == "global" else CATEGORY_MODEL_DIR / f"{model_key}.txt"
    if not path.exists():
        return pd.DataFrame()
    booster = lgb.Booster(model_file=str(path))
    gain = booster.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": booster.feature_name(), "importance": gain})
    df["割合(%)"] = df["importance"] / df["importance"].sum() * 100
    df = df.sort_values("importance", ascending=False)
    return df.head(top_n) if top_n else df


def render_feature_importance(model_key):
    imp = _booster_importance(model_key)
    if imp.empty:
        st.caption("モデルファイルが見つかりません。")
        return
    st.markdown(f"**使用モデル（{model_key}）の特徴量重要度 Top15**（gain の割合）")
    st.bar_chart(imp.set_index("feature")["割合(%)"], horizontal=True)

    shares = {}
    for key in CATEGORY_KEYS:
        cat_imp = _booster_importance(key, top_n=None)
        if not cat_imp.empty:
            shares[key] = cat_imp.set_index("feature")["割合(%)"]
    if shares:
        st.markdown("**カテゴリ別モデルの重み比較**（各特徴量の重要度割合 %、全カテゴリ平均の高い順）")
        table = pd.DataFrame(shares)
        table = table.loc[table.mean(axis=1).sort_values(ascending=False).index].head(15).round(1)
        st.dataframe(table.style.background_gradient(axis=None, cmap="Blues"), width="stretch")

    comparison = load_category_comparison()
    if not comparison.empty:
        st.markdown("**カテゴリ別モデル vs 全体モデル（同じ検証データでの比較）**")
        st.dataframe(comparison, width="stretch", hide_index=True)


def _pace_diff(pace):
    m = re.match(r"\s*(\d{2}\.\d)\s*-\s*(\d{2}\.\d)", str(pace or ""))
    return float(m.group(1)) - float(m.group(2)) if m else np.nan


def build_pace_comparison(pred_df, all_horse_data, n_races=3):
    """AI順位順に、各馬の近走のペース・通過順・上がりを横並びにする。"""
    results_by_name = {
        re.sub(r"\s+", "", str(h["pedigree"].get("name", ""))): h.get("results")
        for h in (all_horse_data or [])
    }
    rows = []
    for _, horse in pred_df.iterrows():
        row = {"印": horse["印"], "馬": _horse_label(horse)}
        results = results_by_name.get(re.sub(r"\s+", "", str(horse["馬名"])))
        diffs, ups = [], []
        if results is not None and not results.empty:
            recent = results.copy()
            recent["_date"] = pd.to_datetime(recent.get("日付"), errors="coerce")
            recent = recent.sort_values("_date", ascending=False).head(n_races)
            for i, (_, r) in enumerate(recent.iterrows(), start=1):
                label = "前走" if i == 1 else f"{i}走前"
                diff = _pace_diff(r.get("ペース"))
                up = pd.to_numeric(r.get("上り"), errors="coerce")
                row[f"{label} 距離"] = r.get("距離", "")
                row[f"{label} ペース"] = r.get("ペース", "")
                row[f"{label} 通過"] = r.get("通過", "")
                row[f"{label} 上り"] = r.get("上り", "")
                row[f"{label} 着順"] = r.get("着順", "")
                if pd.notna(diff):
                    diffs.append(diff)
                if pd.notna(up):
                    ups.append(up)
        row["平均 前半-後半(秒)"] = round(float(np.mean(diffs)), 2) if diffs else np.nan
        row["平均上り"] = round(float(np.mean(ups)), 2) if ups else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def render_pace_comparison(pred_df, all_horse_data):
    st.caption(
        "ペースは「前半3F-後半3F」。前半-後半がマイナスなら前傾（ハイペース）、プラスなら後傾（スロー）の"
        "レースを経験していることを表します。"
    )
    table = build_pace_comparison(pred_df, all_horse_data)
    if table.empty:
        st.caption("過去走データがありません。")
        return
    st.dataframe(table, width="stretch", hide_index=True)

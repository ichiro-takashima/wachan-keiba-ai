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
VOLATILITY_ICONS = {"堅い": ":material/shield:", "標準": ":material/balance:", "荒れ気味": ":material/warning:",
                    "大荒れ警戒": ":material/thunderstorm:"}
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


SURFACE_LABELS = {"芝": ("芝", "green"), "ダ": ("ダート", "orange"), "障": ("障害", "violet")}
# 距離帯の区切りは学習データの距離区分(preprocessing.encode)に合わせる
DISTANCE_BANDS = [(1400, "短距離", "red"), (1800, "マイル", "blue"), (2200, "中距離", "violet"), (9999, "長距離", "gray")]


def course_condition_label(race_info):
    """「芝・マイル」「ダート・長距離」のような色付きラベル(Markdown)。判定できなければ空文字。"""
    from race_analysis import parse_course

    surface, distance = parse_course(race_info)
    if surface not in SURFACE_LABELS:
        return ""
    surface_name, surface_color = SURFACE_LABELS[surface]
    band_name, band_color = next((name, color) for upper, name, color in DISTANCE_BANDS if distance <= upper)
    return f":{surface_color}-badge[{surface_name}・{band_name}] :{band_color}-badge[{distance}m]"


def render_race_overview(race_info, prediction, volatility):
    """Step1: レース名・馬場/距離・使用モデル・波乱度を1行のバッジに、条件と補足を1行の注記にまとめる。"""
    badges = []
    if race_info and race_info.get("name"):
        badges.append(f"**{race_info['name']}**")
    label = course_condition_label(race_info) if race_info else ""
    if label:
        badges.append(label)
    if prediction:
        going = prediction["conditions"].get("going")
        if going:
            badges.append(f":violet-badge[:material/landscape: {going}]")
        badges.append(f":blue-badge[:material/bolt: {prediction['model_key']}]")
    if volatility:
        level = volatility["level"]
        badges.append(
            f":{VOLATILITY_COLORS.get(level, 'gray')}-badge[{VOLATILITY_ICONS.get(level, '')} "
            f"波乱度 {level} {volatility['score']}/100]"
        )
    if badges:
        st.markdown(" ".join(badges))

    notes = []
    if race_info:
        notes += [race_info.get("data", ""), race_info.get("entries", "")]
    if prediction:
        metrics = _model_metrics(prediction["model_key"])
        if metrics and pd.notna(metrics.get("global_model_auc")):
            notes.append(f"AUC カテゴリ別 {metrics['auc']:.3f} / 全体 {metrics['global_model_auc']:.3f}")
            if metrics.get("n_train", 0) < 1000:
                notes.append("学習データが少なく予測がぶれやすい")
        if prediction.get("fallback_reason"):
            notes.append(f"全体モデルで予測（{prediction['fallback_reason']}）")
    if volatility:
        notes.append(f"方針: {volatility['summary']}")
    st.caption(" / ".join(n for n in notes if n))


# ---------------------------------------------------------------- Step2

def render_top_horse_cards(pred_df, top_n=5, win_prob_method="blend"):
    """Step2: AIの推奨上位馬をカード形式で表示する。説明は数値のツールチップに入れて縦幅を取らない。"""
    if win_prob_method == "blend":
        win_note = "推定勝率は3着内確率と単勝オッズを検証データで補正して組み合わせた値です。"
    else:
        win_note = "推定勝率は3着内確率を全頭合計で割った目安です（オッズ未取得のため補正なし。人気薄を過大評価しやすい）。"
    help_text = f"順位・印はLightGBMの3着内確率の順。{win_note}期待値は 推定勝率×単勝オッズ（1.0超で妙味あり）。"
    top = pred_df.head(top_n)
    cols = st.columns(len(top)) if len(top) else []
    for col, (_, row) in zip(cols, top.iterrows()):
        with col.container(border=True):
            mark = row["印"]
            st.markdown(f"**{mark} {MARK_LABELS.get(mark, '')}**　{_horse_label(row)}")
            st.caption(f"{row['騎手']} / {_fmt_odds(row['オッズ'])}"
                       + (f"（{int(row['人気'])}人気）" if pd.notna(row["人気"]) else ""))
            st.metric("3着内確率", _fmt_pct(row["3着内確率"]), help=help_text)
            grade = row["期待値ランク"]
            ev_text = "-" if pd.isna(row["期待値"]) else f"{row['期待値']:.2f}"
            st.markdown(
                f"推定勝率 **{_fmt_pct(row['推定勝率'])}** "
                f":{GRADE_COLORS.get(grade, 'gray')}-badge[期待値 {grade}（{ev_text}）]"
            )
            if row["根拠"]:
                st.caption("・".join(row["根拠"]))
            if row.get("過去走数", 5) == 0:
                st.caption(":material/warning: 過去走データなし")


AFFINITY_HELP = (
    "推定勝率は相性モデル（血統・戦績・馬体重から算出。オッズ・人気は使いません）の値で、全頭の合計が100%です。"
    "血統相性・戦績相性は出走馬の中での偏差値（50が平均）。馬体重影響は、馬体重の情報で勝率が何ポイント動いたか。"
    "期待値は 推定勝率×単勝オッズ（1.0超で妙味あり）。"
)


def _weight_effect_text(value):
    if pd.isna(value):
        return "-"
    return f"{value:+.1f}pt"


def render_affinity_cards(result, top_n=5):
    """相性モデルの上位馬をカードで表示する。"""
    pred = result["predictions"]
    if result.get("mode"):
        st.caption(f":material/tune: 予想モード: {result['mode']}")
    if result.get("weight_missing"):
        st.caption(":material/info: 馬体重が未発表のため、馬体重なしで学習したモデルで計算しています。")
    if result.get("pedigree_missing"):
        st.caption(f":material/info: 血統が取得できなかった馬が{result['pedigree_missing']}頭います（血統相性は平均扱い）。")
    top = pred.head(top_n)
    cols = st.columns(len(top)) if len(top) else []
    for col, (_, row) in zip(cols, top.iterrows()):
        with col.container(border=True):
            mark = row["印"]
            st.markdown(f"**{mark} {MARK_LABELS.get(mark, '')}**　{_horse_label(row)}")
            st.caption(f"{row['騎手']} / {_fmt_odds(row['オッズ'])}"
                       + (f"（{int(row['人気'])}人気）" if pd.notna(row["人気"]) else ""))
            st.metric("推定勝率", _fmt_pct(row["推定勝率"]), help=AFFINITY_HELP)
            grade = row["期待値ランク"]
            ev_text = "-" if pd.isna(row["期待値"]) else f"{row['期待値']:.2f}"
            st.markdown(
                f"血統 **{row['血統相性']:.0f}**・戦績 **{row['戦績相性']:.0f}** "
                f":{GRADE_COLORS.get(grade, 'gray')}-badge[期待値 {grade}（{ev_text}）]"
            )
            st.caption(f"馬体重 {row['馬体重']}（影響 {_weight_effect_text(row['馬体重影響'])}）")
            if row["根拠"]:
                st.caption("・".join(row["根拠"]))
            for note in row.get("血統の得意・苦手") or []:
                st.caption(f":{'green' if '得意' in note else 'red'}[{note}]")
            if row.get("過去走数", 1) == 0:
                st.caption(":material/warning: 中央での過去走なし（血統中心の評価）")


def render_affinity_table(result, height="auto"):
    """相性モデルの全頭の表。"""
    display = result["predictions"].copy()
    display["推定勝率"] = display["推定勝率"] * 100
    display["期待値"] = display["期待値"].round(2)
    display["馬体重影響"] = display["馬体重影響"].map(_weight_effect_text)
    display["根拠"] = display["根拠"].map(lambda r: "・".join(r))
    if "血統の得意・苦手" in display.columns:
        display["血統の得意・苦手"] = display["血統の得意・苦手"].map(lambda r: "・".join(r or []))
    win_max = max(float(display["推定勝率"].max()), 1.0)
    st.dataframe(
        display, width="stretch", height=height, hide_index=True,
        column_config={
            "推定勝率": probability_bar("推定勝率", max_value=win_max),
            "血統相性": st.column_config.ProgressColumn("血統相性", format="%d", min_value=20, max_value=80),
            "戦績相性": st.column_config.ProgressColumn("戦績相性", format="%d", min_value=20, max_value=80),
        },
    )


def render_affinity_model_info(metrics):
    """相性モデルの検証成績・要素別の影響・馬体重の増減と成績。"""
    if not metrics:
        st.caption("検証成績がありません。")
        return
    pick, pop = metrics.get("勝率1位馬", {}), metrics.get("1番人気(参考)", {})
    scores = metrics.get("テスト成績", {})
    if metrics.get("予想モード"):
        st.markdown(f"**予想モード: {metrics['予想モード']}**　{metrics.get('モードの説明', '')}")
    st.caption(
        f"学習 {metrics.get('学習レース数', '?')}レース（{' 〜 '.join(metrics.get('学習期間', []))}）、"
        f"別期間 {metrics.get('テストレース数', '?')}レース（{' 〜 '.join(metrics.get('テスト期間', []))}）で検証。"
        f"血統の取得率 {metrics.get('血統の取得率', 0):.0%}。"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("勝率1位馬の単勝的中率", _fmt_pct(pick.get("単勝的中率")),
              delta=f"{(pick.get('単勝的中率', 0) - pop.get('単勝的中率', 0)) * 100:+.1f}pt",
              help="矢印は1番人気を買い続けた場合との差（参考）")
    c2.metric("単勝回収率", f"{pick.get('単勝回収率', 0)}%",
              delta=f"{pick.get('単勝回収率', 0) - pop.get('単勝回収率', 0):+.1f}pt")
    c3.metric("AUC(1着)", f"{scores.get('AUC(1着)', 0):.3f}")

    shares = metrics.get("要素別の影響度(%)")
    if shares:
        st.caption("勝率の判断に効いている要素の割合(%)")
        st.bar_chart(pd.Series(shares), horizontal=True)
    ablation = metrics.get("要素を外したときの成績")
    if ablation:
        st.caption("要素を1つずつ外して学習し直したときのテスト成績（対数尤度が増える・AUCが下がるほど、その要素が効いている）")
        st.dataframe(pd.DataFrame(ablation), width="stretch", hide_index=True)
    weight = metrics.get("馬体重の増減と成績")
    if weight:
        st.caption("馬体重の増減とテスト期間の実際の成績。「モデルの馬体重寄与」は馬体重の情報が勝率(対数オッズ)をどちらへ動かしたかの平均。")
        st.dataframe(pd.DataFrame(weight), width="stretch", hide_index=True)
    calib = metrics.get("勝率の当てはまり")
    if calib:
        st.caption("予測勝率と実際の勝率（近いほど、勝率の数字をそのまま信用できる）")
        st.dataframe(pd.DataFrame(calib), width="stretch", hide_index=True)


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
    """Step3: 券種ごとの推奨買い目を1つの表にまとめる(資金配分シミュレータは呼び出し側で横に並べる)。"""
    level = volatility["level"] if volatility else None
    if level:
        st.caption(f"波乱度「{level}」に合わせて、◎から流す相手を{PARTNER_COUNT_BY_LEVEL.get(level, 4)}頭にしています。")
    rows = [
        {"券種": sug["券種"], "方式": sug["方式"], "点数": len(sug["買い目"]),
         "各100円": f"{len(sug['買い目']) * 100:,}円", "買い目": "  ".join(sug["買い目"])}
        for sug in build_ticket_suggestions(pred_df, level)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={"買い目": st.column_config.TextColumn("買い目", width="large")})


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
        c1, c2 = st.columns(2)
        budget = c1.number_input("単勝の予算 (円)", min_value=100, step=100, value=3000, key="ml_alloc_budget")
        top_k = c2.slider("対象頭数（AI上位）", min_value=1, max_value=min(8, max(1, len(pred_df))),
                          value=min(5, max(1, len(pred_df))), key="ml_alloc_topk")
        method = st.radio("配分方法", list(ALLOCATION_METHODS), key="ml_alloc_method", horizontal=True,
                          captions=list(ALLOCATION_METHODS.values()))

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


# ---------------------------------------------------------------- AIプロンプト用

def format_steps_for_prompt(prediction, volatility=None, budget=3000, top_k=5, method="推奨度比例",
                            metrics=None, include_allocation=True):
    """Step1〜3で画面に出している内容を、AI総合予想のプロンプト用テキストにする。

    include_allocation=False で、Step3の資金配分シミュレーション(ガミを考慮しない配分)を省く。
    """
    if not prediction:
        return ""
    pred_df = prediction["predictions"]
    cond = prediction["conditions"]
    lines = ["【Step 1】レース概況・適用モデル"]
    course = f"{cond['surface'] or '?'}{int(cond['distance']) if pd.notna(cond['distance']) else '?'}m"
    if cond.get("going"):
        course += f"（{cond['going']}）"
    lines.append(f"- コース: {course}")
    lines.append(f"- 使用モデル: {prediction['model_label']}（{prediction['model_key']}）")
    if prediction.get("fallback_reason"):
        lines.append(f"- カテゴリ別モデルを使えず全体モデルで予測: {prediction['fallback_reason']}")
    metrics = metrics or _model_metrics(prediction["model_key"])
    if metrics and pd.notna(metrics.get("auc")):
        lines.append(f"- 検証データでの精度(AUC): {metrics['auc']:.3f}（学習{metrics['n_train']:,}行）")
    if volatility:
        lines.append(f"- 波乱度: {volatility['level']}（{volatility['score']}/100）")

    lines += ["", "【Step 2】AI推奨馬ランキング（推定勝率順。推定勝率×単勝オッズ=期待値、1.0超で妙味あり）"]
    table = pd.DataFrame({
        "印": pred_df["印"],
        "馬番": pred_df["馬番"].map(lambda n: "-" if pd.isna(n) else int(n)),
        "馬名": pred_df["馬名"],
        "騎手": pred_df["騎手"],
        "オッズ": pred_df["オッズ"].map(_fmt_odds),
        "人気": pred_df["人気"].map(lambda n: "-" if pd.isna(n) else int(n)),
        "3着内確率": pred_df["3着内確率"].map(_fmt_pct),
        "推定勝率": pred_df["推定勝率"].map(_fmt_pct),
        "期待値": pred_df["期待値"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
        "期待値ランク": pred_df["期待値ランク"],
        "評価ポイント": pred_df["根拠"].map(lambda r: "・".join(r) if r else ""),
    })
    lines.append(table.to_csv(index=False, sep="|").strip())

    level = volatility["level"] if volatility else None
    lines += ["", f"【Step 3】AIモデルの推奨買い目（◎から流す相手{PARTNER_COUNT_BY_LEVEL.get(level, 4)}頭）"]
    for sug in build_ticket_suggestions(pred_df, level):
        lines.append(f"- {sug['券種']} {sug['方式']} {len(sug['買い目'])}点: {' / '.join(sug['買い目'])}")
    alloc = allocate_budget(pred_df, budget, method, top_k=top_k) if include_allocation else pd.DataFrame()
    if not alloc.empty:
        lines.append(f"- 単勝の資金配分シミュレーション（予算{budget:,}円・上位{top_k}頭・{method}）:")
        for _, row in alloc.iterrows():
            lines.append(f"  {row['印']}{_horse_label(row)} {row['購入額']}円（オッズ{_fmt_odds(row['オッズ'])}、的中時払戻{row['的中時払戻']:,}円）")
    lines.append("上記Step1〜3はアプリ内の機械学習モデルの評価です。血統・過去10戦などの他データと照らし合わせ、最終判断に活用してください。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 詳細分析タブ

def probability_bar(label, max_value=100):
    """確率(0〜100の%値)を横棒で表示する列設定。"""
    return st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0, max_value=max_value)


def render_prediction_table(pred_df, height="auto"):
    display = pred_df.copy()
    display["3着内確率"] = display["3着内確率"] * 100
    display["推定勝率"] = display["推定勝率"] * 100
    display["期待値"] = display["期待値"].round(2)
    display["根拠"] = display["根拠"].map(lambda r: "・".join(r))
    win_max = max(float(display["推定勝率"].max()), 1.0) if display["推定勝率"].notna().any() else 100
    st.dataframe(
        display, width="stretch", height=height, hide_index=True,
        column_config={
            "3着内確率": probability_bar("3着内確率"),
            # 勝率は値が小さいので、最大の馬を棒いっぱいにして差を見やすくする
            "推定勝率": probability_bar("推定勝率", max_value=win_max),
        },
    )


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
    st.caption(f"使用モデル（{model_key}）の特徴量重要度 Top15（gain の割合）")
    st.bar_chart(imp.set_index("feature")["割合(%)"], horizontal=True)

    shares = {}
    for key in CATEGORY_KEYS:
        cat_imp = _booster_importance(key, top_n=None)
        if not cat_imp.empty:
            shares[key] = cat_imp.set_index("feature")["割合(%)"]
    if shares:
        st.caption("カテゴリ別モデルの重み比較（各特徴量の重要度割合 %、全カテゴリ平均の高い順）")
        table = pd.DataFrame(shares)
        table = table.loc[table.mean(axis=1).sort_values(ascending=False).index].head(15).round(1)
        st.dataframe(table.style.background_gradient(axis=None, cmap="Blues"), width="stretch")

    comparison = load_category_comparison()
    if not comparison.empty:
        st.caption("カテゴリ別モデル vs 全体モデル（同じ検証データでの比較）")
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


# ---------------------------------------------------------------- 出馬表モデル(収集レースで学習)

def render_score_cards(n_races, hit_rate, return_rate, hit_label="的中率", return_label="回収率",
                       hit_delta=None, return_delta=None, races_label="対象レース数"):
    """回収率・的中率・対象レース数を st.metric のカードで並べる。rate は 0〜1、回収率は%。"""
    cols = st.columns(3)
    cols[0].metric(return_label, "-" if return_rate is None else f"{return_rate:.1f}%",
                   delta=return_delta, border=True)
    cols[1].metric(hit_label, _fmt_pct(hit_rate) if hit_rate is not None else "-",
                   delta=hit_delta, border=True)
    cols[2].metric(races_label, f"{int(n_races):,}" if n_races is not None else "-", border=True)


def render_model_score_cards(metrics):
    """検証期間で、モデルの1位予想馬を買い続けた成績(1番人気買いとの差を delta で表示)。"""
    m, b = metrics.get("モデル1位馬", {}), metrics.get("1番人気", {})
    st.caption(
        f"学習 {metrics.get('学習レース数', '?')}レース（{' 〜 '.join(metrics.get('学習期間', []))}）で作り、"
        f"別期間（{' 〜 '.join(metrics.get('テスト期間', []))}）で、1位予想馬の複勝を毎レース買った場合の成績。"
        "矢印は1番人気を買い続けた場合との差です。"
    )
    render_score_cards(
        m.get("レース数"), m.get("複勝的中率"), m.get("複勝回収率"),
        hit_label="複勝的中率", return_label="複勝回収率", races_label="検証レース数",
        hit_delta=f"{(m.get('複勝的中率', 0) - b.get('複勝的中率', 0)) * 100:+.1f}pt",
        return_delta=f"{m.get('複勝回収率', 0) - b.get('複勝回収率', 0):+.1f}pt",
    )

def render_shutuba_model(result, metrics, error=None, height="auto"):
    """収集レースで学習した出馬表モデルの3着以内確率。検証成績と重視する情報はポップオーバーに畳む。"""
    if result is None:
        st.caption(error or "採用済みのモデルがありません（python -m model.shutuba_model で学習・検証してください）。")
        return

    if result.get("odds_missing"):
        st.warning("オッズが未発表のため、オッズ・人気なしで計算しています（精度が下がります）。", icon=":material/warning:")
    if result.get("weight_missing"):
        st.warning("馬体重が未発表のまま「馬体重を使う」モデルで計算しています（参考値）。"
                   "サイドバーで馬体重を「使わない」にすると、馬体重なしで学習したモデルで予想します。",
                   icon=":material/warning:")
    if result.get("features"):
        st.caption(f"説明変数 {len(result['features'])}個"
                   + ("" if "体重" in result["features"] else "（馬体重なし）")
                   + "：" + "、".join(result["features"]))
    if metrics:
        with st.popover("検証成績・重視する情報", icon=":material/insights:"):
            render_model_score_cards(metrics)
            importance = metrics.get("特徴量重要度(gain%)")
            if importance:
                st.caption("「根拠」列は各馬の確率を押し上げた上位の情報。グラフはモデル全体で重視している情報の割合(%)。")
                st.bar_chart(pd.Series(importance).head(10), horizontal=True)

    display = result["predictions"].copy()
    display["3着内確率"] = display["3着内確率"] * 100
    display["単勝オッズ"] = display["単勝オッズ"].map(_fmt_odds)
    st.dataframe(display, width="stretch", height=height, hide_index=True,
                 column_config={"3着内確率": probability_bar("3着内確率")})


# ---------------------------------------------------------------- モデル説明ページ

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def render_model_explanation():
    """各予測モデルの仕組み・検証結果・根拠をまとめて表示するページ。"""
    from model.shutuba_model import CANDIDATE_PATH, METRICS_PATH
    from race_analysis import DIRT_STRONG_SIRE_MIN_DISTANCE, DIRT_STRONG_SIRE_POINTS, DIRT_STRONG_SIRES

    tab_shutuba, tab_category, tab_rule = st.tabs([
        ":material/table_chart: 出馬表モデル", ":material/category: カテゴリ別 / 全体モデル", ":material/rule: ルールベースの加点",
    ])
    with tab_shutuba:
        _render_shutuba_model_explanation(METRICS_PATH, CANDIDATE_PATH)
    with tab_category:
        st.caption("各馬の直近5走・騎手勝率・レース条件から3着以内確率を出すモデルです（オッズ・人気は使いません）。"
                   "芝・ダート×距離帯ごとに分けたカテゴリ別モデルと、全条件共通の全体モデルがあります。")
        comparison = load_category_comparison()
        if not comparison.empty:
            st.dataframe(comparison, width="stretch", hide_index=True)
        with st.popover("全体モデルの特徴量重要度", icon=":material/insights:"):
            render_feature_importance("global")
    with tab_rule:
        st.markdown(
            f"- ダート{DIRT_STRONG_SIRE_MIN_DISTANCE}m以上のレースで、父が {'・'.join(DIRT_STRONG_SIRES)} の馬は "
            f"+{DIRT_STRONG_SIRE_POINTS:g}点（期待値・ルール評価）\n"
            "- 出馬表モデルが採用されている場合は、その3着以内確率も総合評価に加えます（重み1.5）"
        )


def _render_shutuba_model_explanation(metrics_path, candidate_path):
    adopted = _load_json(metrics_path)
    metrics = adopted or _load_json(candidate_path.with_suffix(".metrics.json"))
    if metrics is None:
        st.info("まだ学習されていません（python -m model.shutuba_model で学習・検証します）。", icon=":material/info:")
        return
    if adopted:
        st.success("検証に合格し、予想画面で使われています。", icon=":material/check_circle:")
    else:
        st.warning(f"不採用のため予想画面では使っていません（{metrics.get('採用理由', '')}）。"
                   "python -m model.shutuba_model --force で採用できます。", icon=":material/warning:")
    st.caption(
        "出馬表に載る情報（オッズ・人気・枠・斤量・馬体重・騎手の過去成績・近走成績・レース条件）から、"
        "各馬が3着以内に入る確率を出すLightGBMモデルです。"
    )
    render_model_score_cards(metrics)
    auc_cols = st.columns(3)
    auc_cols[0].metric("AUC（モデル）", f"{metrics.get('AUC', 0):.3f}",
                       delta=f"{metrics.get('AUC', 0) - metrics.get('AUC(人気順)', 0):+.3f}", border=True,
                       help="3着以内の馬をどれだけ上位に並べられたか(0.5=でたらめ、1.0=完璧)。差は人気順との比較。")
    auc_cols[1].metric("AUC（人気順）", f"{metrics.get('AUC(人気順)', 0):.3f}", border=True)
    auc_cols[2].metric("学習レース数", f"{metrics.get('学習レース数', 0):,}", border=True)

    pop_cols = st.columns(2)
    with pop_cols[0].popover("使っている特徴量・使っていない情報", icon=":material/list:", use_container_width=True):
        st.markdown("**特徴量:** " + "、".join(metrics.get("特徴量", [])))
        st.markdown("**使っていない情報（レース後にしか分からないため）:** "
                    + "、".join(metrics.get("除外した結果系の列", [])))
    importance = metrics.get("特徴量重要度(gain%)")
    if importance:
        with pop_cols[1].popover("特徴量重要度", icon=":material/insights:", use_container_width=True):
            st.bar_chart(pd.Series(importance).head(15), horizontal=True)

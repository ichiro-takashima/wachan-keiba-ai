"""的中時の損失を出さない範囲で、レース状況に合った買い方をAIに組ませるためのプロンプト。

能力評価はアプリ内の機械学習モデルに任せ、AIには現在オッズとの突き合わせ・金額配分の説明だけを担当させる。
配分はこちらで計算して検算済みの候補として渡す。オッズが未発表の券種は、モデルの確率から推定したオッズで組む。

使うモデルは2つ:
- live_predict(出馬表モデル): オッズや全体傾向を含む推定勝率・3着内確率。候補馬の順位はこちら
- affinity_model(相性モデル): オッズを使わない、血統・馬体重・戦績による勝率
レース状況を4パターン(A: 血統穴馬 / B: 1強・2強 / C: 混戦 / D: 少額予算)に分け、該当パターンの券種を先頭に出す。
候補プランは「期待値(両モデルの推定勝率の平均 × オッズ)+ 優先券種への加点 − 同じ券種の連続への減点」の順に並べる。
オッズ未発表の券種があるときは、期待値の代わりに的中率を使う(推定オッズでは期待値がどのプランも払戻率と同じになるため)。
"""
import math
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from ml_ui import PARTNER_COUNT_BY_LEVEL, format_steps_for_prompt
from tickets import normalize_combo, normalize_ticket_type

# scraper.scrape_realtime_odds に渡す券種名(オッズAPIの名前)
GAMI_ODDS_TICKET_TYPES = ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複"]
UNIT = 100
# 的中時の最低回収率(払戻÷購入合計)の選択肢と既定値。100%で「ガミを一切許さない」
MIN_RETURN_RATIO_OPTIONS = [1.0, 0.9, 0.8, 0.7, 0.6]
DEFAULT_MIN_RETURN_RATIO = 1.0
# 候補プランはこの回収率で組み、満たさないプランは除外する(画面で低い値が指定されても、この値より下げない)
REQUIRED_MIN_RETURN_RATIO = 1.0
# 同一レースで複勝を買う頭数の上限(これを超える複勝プランは作らない・出さない)
MAX_PLACE_HORSES = 2
# 3点以上のプランの目安(少額予算パターンの判定に使う)
PREFERRED_MIN_POINTS = 3
# JRAの券種別払戻率。未発表オッズの推定に使う
PAYOUT_RATE = {"単勝": 0.80, "複勝": 0.80, "馬連": 0.775, "ワイド": 0.775, "馬単": 0.75, "3連複": 0.75}

# レース状況の判定と並べ替えの設定。しきい値はすべてここで調整する。
SITUATION_RULES = {
    # 期待値・的中率に使う勝率 = live_predict の推定勝率 ×(1−w)+ affinity_model の推定勝率 × w
    # 片方のモデルに出力がない馬は、ある方の値をそのまま使う
    "affinity_weight": 0.5,
    # A: 血統穴馬。affinity の上位 top_n 頭に、単勝 min_odds 倍以上 または live の順位より rank_gap 以上上の馬がいる
    "A": {"top_n": 3, "min_odds": 10.0, "rank_gap": 4, "partners": 3},
    # B: 1強・2強。live の1位馬の推定勝率が top1_win 以上、または上位2頭の3着内確率の合計が3・4位の合計の
    #    top2_place_ratio 倍以上
    "B": {"top1_win": 0.35, "top2_place_ratio": 1.6, "exacta_partners": 3, "trio_partners": 4},
    # C: 混戦。live の1〜top_n位の推定勝率がすべて win_min〜win_max
    "C": {"top_n": 4, "win_min": 0.10, "win_max": 0.18},
    # D: 少額予算。予算 max_budget 円以下・最低回収率 min_ratio 以上で、multi_point_types の券種では
    #    min_points 点以上のプランが100円単位の配分で組めない(トリガミが確定する)
    "D": {"max_budget": 1000, "min_ratio": 1.0, "multi_point_types": ["ワイド", "馬連"], "min_points": 3},
    # 評価ギャップの一覧に載せる基準(どちらかを満たす馬)
    "gap": {"rank_gap": 3, "win_gap": 0.05},
    # 並べ替え: 判定パターンのプランへの加点と、同じ券種が上位にすでにある数 × 減点
    "pattern_bonus": 1.0,
    "same_type_penalty": 0.15,
}
PATTERN_PRIORITY = ["D", "B", "A", "C"]  # 複数に該当したときの優先順位(D は資金面の制約なので最優先)
PATTERN_TAGS = {"A": "血統穴馬狙い", "B": "1強軸流し", "C": "混戦BOX", "D": "少額ピンポイント"}
PATTERN_NAMES = {"A": "血統穴馬", "B": "1強・2強", "C": "混戦", "D": "少額予算"}


def build_persona_prompt(min_return_ratio):
    floor = f"{min_return_ratio:.0%}"
    tolerance_rule = (
        "" if min_return_ratio >= 1.0
        else "\n   この範囲内のガミは許容します。ガミを完全に消すために点数を削らないでください。"
    )
    return f"""あなたは競馬の資金配分に特化したアシスタントです。
各馬の能力評価はすでにアプリ内の2つの機械学習モデルが行っています。出馬表モデル（オッズや全体傾向を含む）と、
相性モデル（オッズを使わない、血統・馬体重・戦績による適性）です。あなたの役割は、その評価と現在オッズ、
【レース状況の判定】を使い、的中時に損をしない範囲でレース状況に合った券種・組み合わせ・金額配分を組み立てることです。

# 守るべきルール
1. 採用するすべての買い目について「購入額 × オッズ（複勝・ワイドは下限） ≥ 購入合計 × {floor}」を満たすこと。{tolerance_rule}
2. 候補プランは「期待値（両モデルの推定勝率の平均 × オッズ）＋ 判定パターンの券種への加点 − 同じ券種が続くことへの減点」の
   スコア順に並べています（オッズ未発表のときは期待値の代わりに的中率）。原則として先頭のプランを採用すること。
3. 採用理由では、プランの「TYPE」タグと【レース状況の判定】の根拠の数値、【2モデルの評価ギャップ】を使い、
   なぜこの券種を先頭にしたのかを説明すること（例: 相性モデルだけが高く評価する人気薄がいる → 穴馬の単勝・ワイド）。
4. 複勝は同じレースで最大{MAX_PLACE_HORSES}頭まで。3頭以上の複勝は買わないこと。
5. 金額の計算は【計算済み候補プラン】の数字（検算済み）をそのまま使うのが安全。独自に組む場合は、ルール1を買い目ごとに検算すること。
6. 推定オッズで組まれたプランも買い目として提案してよい。オッズ未発表は見送りの理由にしないこと。
   その場合は、発売開始後に必要オッズを満たしているか確認するよう一言添えること。
7. 見送り（0円）は、すべての候補プランが組めない場合だけにすること。
8. 金額は100円単位。馬は馬番で指定する。断定的な表現（「絶対」「鉄板」など）は使わない。
"""


def _normalize_odds(odds_table):
    """{券種: {組み合わせ: {...}}} の券種名・組み合わせ表記をtickets.pyの正規形にそろえる。"""
    normalized = {}
    for t_type, combos in (odds_table or {}).items():
        n_type = normalize_ticket_type(t_type)
        normalized[n_type] = {normalize_combo(k, n_type): v for k, v in combos.items()}
    return normalized


def _odds(odds_table, t_type, combo):
    """下限オッズ(単勝・馬連・3連複は唯一のオッズ)。取得できなければNone。"""
    info = odds_table.get(t_type, {}).get(normalize_combo(combo, t_type))
    odds = info.get("odds") if info else None
    return odds if odds and odds > 0 else None


def candidate_horses(pred_df, n=6):
    """AI評価(3着内確率)の上位n頭の馬番。"""
    numbers = [int(x) for x in pred_df["馬番"] if pd.notna(x)]
    return numbers[:n]


def synthetic_odds(odds_list):
    return 1 / sum(1 / o for o in odds_list)


def finish_distribution(pred_df):
    """推定勝率から1〜3着の着順の確率を求める(Harville法)。戻り値: {(1着, 2着, 3着): 確率}"""
    win_prob = {
        int(h): float(p) for h, p in zip(pred_df["馬番"], pred_df["推定勝率"])
        if pd.notna(h) and pd.notna(p) and p > 0
    }
    total = sum(win_prob.values())
    if total <= 0:
        return {}
    win_prob = {h: p / total for h, p in win_prob.items()}
    distribution = {}
    for a, b, c in permutations(win_prob, 3):
        rest_after_a = 1 - win_prob[a]
        rest_after_ab = rest_after_a - win_prob[b]
        if rest_after_a > 0 and rest_after_ab > 0:
            distribution[(a, b, c)] = win_prob[a] * win_prob[b] / rest_after_a * win_prob[c] / rest_after_ab
    return distribution


def _is_hit(t_type, combo, order):
    """着順orderのとき、その買い目が的中するか。"""
    numbers = {int(n) for n in combo.split("-")}
    if t_type == "単勝":
        return numbers == {order[0]}
    if t_type == "馬単":
        return [int(n) for n in combo.split("-")] == list(order[:2])
    if t_type == "馬連":
        return numbers == set(order[:2])
    # 複勝・ワイド・3連複は、指定馬がすべて3着以内に入れば的中
    return numbers <= set(order)


def _hit_probability(distribution, t_type, combo):
    return sum(p for order, p in distribution.items() if _is_hit(t_type, combo, order))


def _estimated_odds(distribution, t_type, combo):
    """モデル確率と払戻率から推定したオッズ。的中確率が0なら推定できないのでNone。"""
    hit_prob = _hit_probability(distribution, t_type, combo)
    return max(round(PAYOUT_RATE[t_type] / hit_prob, 1), 1.0) if hit_prob > 0 else None


def allocate_with_floor(odds_list, budget, min_return_ratio=1.0, unit=UNIT):
    """どれが的中しても払戻≥購入合計×min_return_ratioになる100円単位の配分。組めなければNone。

    合計Tを予算から順に下げ、各買い目に「その払戻を確保できる最低額」(100円単位で切り上げ)を置いて
    合計がTに収まる最大のTを探す。余った分は払戻がそろうよう(オッズに反比例する理想額に近づくよう)
    100円ずつ足す。足しても払戻が増えるだけなので条件は崩れない。
    """
    if not odds_list or budget < unit * len(odds_list) or synthetic_odds(odds_list) <= min_return_ratio:
        return None
    share = synthetic_odds(odds_list)
    for total in range(budget // unit * unit, unit * len(odds_list) - 1, -unit):
        amounts = [math.ceil(total * min_return_ratio / (o * unit) - 1e-9) * unit for o in odds_list]
        slack = total - sum(amounts)
        if slack < 0:
            continue
        ideal = [total * share / o for o in odds_list]
        for _ in range(slack // unit):
            i = max(range(len(amounts)), key=lambda k: ideal[k] - amounts[k])
            amounts[i] += unit
        return amounts
    return None


def _plan_outlook(bets, total, distribution):
    """プラン全体の的中率と期待回収率の目安(複数同時に的中する場合も数える)。"""
    hit_prob, expected_return = 0.0, 0.0
    for order, p in distribution.items():
        payout = sum(b["return"] for b in bets if _is_hit(b["type"], b["combo"], order))
        if payout > 0:
            hit_prob += p
            expected_return += p * payout
    return hit_prob, expected_return / total


def _make_plan(name, t_type, combos, odds_table, budget, distribution, min_return_ratio=1.0, note=""):
    """組み合わせから、最低回収率を守って組める最大点数のプランを作る(的中しにくい方から削る)。

    その券種のオッズがまったく取得できないときは、モデル確率からの推定オッズで暫定の配分を組む。
    """
    combos = [normalize_combo(c, t_type) for c in combos]
    combos.sort(key=lambda c: _hit_probability(distribution, t_type, c), reverse=True)
    estimated = t_type not in odds_table
    if estimated:
        priced = [(c, _estimated_odds(distribution, t_type, c)) for c in combos]
    else:
        priced = [(c, _odds(odds_table, t_type, c)) for c in combos]
    missing = [c for c, o in priced if o is None]
    priced = [(c, o) for c, o in priced if o is not None]
    base = {"name": name, "note": note, "estimated": estimated, "missing": missing}
    for k in range(len(priced), 0, -1):
        use = priced[:k]
        amounts = allocate_with_floor([o for _, o in use], budget, min_return_ratio)
        if amounts is None:
            continue
        total = sum(amounts)
        bets = [
            {"type": t_type, "combo": c, "amount": a, "odds": o, "return": int(round(a * o)),
             "required_odds": math.ceil(total * min_return_ratio / a * 10 - 1e-9) / 10}
            for (c, o), a in zip(use, amounts)
        ]
        hit_prob, expected_rate = _plan_outlook(bets, total, distribution)
        min_return = min(b["return"] for b in bets)
        return {
            **base, "bets": bets, "total": total,
            "synthetic_odds": synthetic_odds([o for _, o in use]),
            "min_return": min_return, "min_return_ratio": min_return / total,
            "dropped": [c for c, _ in priced[k:]],
            "hit_prob": hit_prob, "expected_rate": expected_rate,
        }
    return {**base, "bets": [], "dropped": [c for c, _ in priced]}


# ---------------------------------------------------------------- 2モデルの突き合わせ

def _numeric_col(df, name):
    if df is None or name not in df.columns:
        return pd.Series(np.nan, index=getattr(df, "index", None), dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def merge_model_views(pred_df, affinity_df=None, odds_table=None, rules=SITUATION_RULES):
    """live_predict と affinity_model の評価を馬番でつないだ表(live の順位順)。

    列: 馬番, 馬名, live勝率, live3着内率, live順位, 相性勝率, 相性順位, オッズ, 期待値用勝率, 順位差, 勝率差
    - live順位: pred_df の並び順(3着内確率の順)。相性順位: affinity の推定勝率の順
    - オッズ: 現在オッズ(単勝)→ live の出馬表オッズ → affinity のオッズ の順に、取れたものを使う
    - 期待値用勝率: 両モデルの推定勝率の加重平均。片方しかない馬はある方の値
    - 順位差 = live順位 − 相性順位(プラスほど相性モデルの方が高く評価)。勝率差 = 相性勝率 − live勝率
    片方のモデルにしか出ていない馬も行に入れる(もう片方の列は欠損)。
    """
    live = pred_df[_numeric_col(pred_df, "馬番").notna()]
    view = pd.DataFrame({
        "馬番": _numeric_col(live, "馬番").astype(int).to_numpy(),
        "馬名": live["馬名"].astype(str).to_numpy() if "馬名" in live.columns else "",
        "live勝率": _numeric_col(live, "推定勝率").to_numpy(),
        "live3着内率": _numeric_col(live, "3着内確率").to_numpy(),
        "live順位": np.arange(1, len(live) + 1, dtype=float),
        "_live_odds": _numeric_col(live, "オッズ").to_numpy(),
    }).drop_duplicates("馬番")
    if affinity_df is not None and len(affinity_df):
        aff = affinity_df[_numeric_col(affinity_df, "馬番").notna()]
        aff = pd.DataFrame({
            "馬番": _numeric_col(aff, "馬番").astype(int).to_numpy(),
            "_aff_name": aff["馬名"].astype(str).to_numpy() if "馬名" in aff.columns else "",
            "相性勝率": _numeric_col(aff, "推定勝率").to_numpy(),
            "_aff_odds": _numeric_col(aff, "オッズ").to_numpy(),
        }).drop_duplicates("馬番")
        aff["相性順位"] = aff["相性勝率"].rank(ascending=False, method="first")
        view = view.merge(aff, on="馬番", how="outer")
        view["馬名"] = view["馬名"].where(view["馬名"].notna() & (view["馬名"] != ""), view["_aff_name"])
    else:
        view["相性勝率"] = np.nan
        view["相性順位"] = np.nan
        view["_aff_odds"] = np.nan
    win_odds = {int(k): v.get("odds") for k, v in (odds_table or {}).get("単勝", {}).items()
                if str(k).isdigit() and v and v.get("odds")}
    view["オッズ"] = view["馬番"].map(win_odds).astype(float)
    view["オッズ"] = view["オッズ"].fillna(view["_live_odds"]).fillna(view["_aff_odds"])
    view.loc[view["オッズ"] <= 0, "オッズ"] = np.nan
    w = rules["affinity_weight"]
    both = view["live勝率"].notna() & view["相性勝率"].notna()
    view["期待値用勝率"] = np.where(both, view["live勝率"] * (1 - w) + view["相性勝率"] * w,
                               view["live勝率"].fillna(view["相性勝率"]))
    view["順位差"] = view["live順位"] - view["相性順位"]
    view["勝率差"] = view["相性勝率"] - view["live勝率"]
    view = view.sort_values(["live順位", "相性順位"], na_position="last").reset_index(drop=True)
    return view.drop(columns=[c for c in view.columns if c.startswith("_")])


def model_gaps(view, rules=SITUATION_RULES):
    """2モデルの評価が大きく食い違う馬(順位差・勝率差が基準以上)。順位差の大きい順。"""
    g = rules["gap"]
    big = (view["順位差"].abs() >= g["rank_gap"]) | (view["勝率差"].abs() >= g["win_gap"])
    gaps = view[big & view["live勝率"].notna() & view["相性勝率"].notna()]
    return gaps.reindex(gaps["順位差"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------- レース状況の判定

def _pct(v):
    return "-" if pd.isna(v) else f"{v:.1%}"


def _odds_text(v):
    return "オッズ未取得" if pd.isna(v) else f"単勝{v:.1f}倍"


def classify_race(view, base_plans, budget, min_return_ratio, rules=SITUATION_RULES):
    """レース状況をパターン A〜D に分け、優先順位 D > B > A > C で1つ選ぶ。どれにも当たらなければ pattern=None。

    base_plans はパターンによらない候補プラン(少額予算の判定で、多点買いが組めるかを見るのに使う)。
    戻り値: {"pattern", "tag", "matched": {パターン: 根拠の文字列のリスト}, "horse"(A の穴馬), "gaps"}
    """
    matched, horse = {}, None
    live = view.dropna(subset=["live順位"]).sort_values("live順位")

    d = rules["D"]
    multi = [p for p in base_plans if p["bets"] and p["bets"][0]["type"] in d["multi_point_types"]
             and len(p["bets"]) >= d["min_points"]]
    if budget <= d["max_budget"] and min_return_ratio >= d["min_ratio"] and not multi:
        matched["D"] = [f"予算{budget:,}円（基準{d['max_budget']:,}円以下）・的中時の最低回収率{min_return_ratio:.0%}",
                        f"{'・'.join(d['multi_point_types'])}を{d['min_points']}点以上買うと、100円単位の配分では"
                        "どれかが的中してもトリガミになる"]

    b = rules["B"]
    if len(live):
        top = live.iloc[0]
        reasons = []
        if pd.notna(top["live勝率"]) and top["live勝率"] >= b["top1_win"]:
            reasons.append(f"出馬表モデル1位の{int(top['馬番'])}番 {top['馬名']}の推定勝率{_pct(top['live勝率'])}"
                           f"（基準{b['top1_win']:.0%}以上）")
        place = live["live3着内率"].head(4)
        if len(place) == 4 and place.notna().all() and place.iloc[2:].sum() > 0:
            ratio = place.iloc[:2].sum() / place.iloc[2:].sum()
            if ratio >= b["top2_place_ratio"]:
                reasons.append(f"上位2頭の3着内確率の合計{place.iloc[:2].sum():.2f}が3・4位の合計{place.iloc[2:].sum():.2f}の"
                               f"{ratio:.2f}倍（基準{b['top2_place_ratio']}倍以上）")
        if reasons:
            matched["B"] = reasons

    a = rules["A"]
    aff_top = view[view["相性順位"] <= a["top_n"]].sort_values("相性順位")
    for _, row in aff_top.iterrows():
        by_odds = pd.notna(row["オッズ"]) and row["オッズ"] >= a["min_odds"]
        by_rank = pd.notna(row["順位差"]) and row["順位差"] >= a["rank_gap"]
        if by_odds or by_rank:
            horse = int(row["馬番"])
            live_rank = "-" if pd.isna(row["live順位"]) else f"{int(row['live順位'])}位"
            text = (f"{horse}番 {row['馬名']}: 相性モデル{int(row['相性順位'])}位（勝率{_pct(row['相性勝率'])}）、"
                    f"出馬表モデル{live_rank}（勝率{_pct(row['live勝率'])}）、{_odds_text(row['オッズ'])}")
            why = []
            if by_odds:
                why.append(f"単勝{a['min_odds']:.0f}倍以上")
            if by_rank:
                why.append(f"出馬表モデルより{int(row['順位差'])}順位上（基準{a['rank_gap']}以上）")
            matched["A"] = [text, "該当: " + "・".join(why)]
            break

    c = rules["C"]
    top_c = live.head(c["top_n"])
    if len(top_c) == c["top_n"] and top_c["live勝率"].between(c["win_min"], c["win_max"]).all():
        matched["C"] = ["出馬表モデル1〜{}位の推定勝率が {}（すべて{:.0%}〜{:.0%}）".format(
            c["top_n"], " / ".join(f"{int(r['馬番'])}番{_pct(r['live勝率'])}" for _, r in top_c.iterrows()),
            c["win_min"], c["win_max"])]

    pattern = next((p for p in PATTERN_PRIORITY if p in matched), None)
    return {"pattern": pattern, "tag": PATTERN_TAGS.get(pattern), "matched": matched,
            "horse": horse if pattern == "A" else None, "gaps": model_gaps(view, rules)}


# ---------------------------------------------------------------- 候補プラン

def _base_plans(horses, volatility_level, plan):
    """パターンによらない候補(従来の候補。複勝は MAX_PLACE_HORSES 頭まで)。"""
    axis = horses[0]
    partners = horses[1:1 + PARTNER_COUNT_BY_LEVEL.get(volatility_level, 4)]

    def box(size, n):
        return ["-".join(str(h) for h in c) for c in combinations(horses[:n], size)]

    plans = [
        plan("単勝◎1点", "単勝", [str(axis)]),
        plan("複勝◎1点", "複勝", [str(axis)], "複勝オッズは下限で計算"),
        plan("単勝 上位3頭", "単勝", [str(h) for h in horses[:3]]),
        plan(f"複勝 上位{MAX_PLACE_HORSES}頭", "複勝", [str(h) for h in horses[:MAX_PLACE_HORSES]], "複勝オッズは下限で計算"),
        plan(f"馬連 ◎流し（相手最大{len(partners)}頭）", "馬連", [f"{axis}-{p}" for p in partners]),
        plan(f"ワイド ◎流し（相手最大{len(partners)}頭）", "ワイド", [f"{axis}-{p}" for p in partners],
             "ワイドオッズは下限で計算"),
        plan("馬連 BOX 上位4頭", "馬連", box(2, 4)),
        plan("ワイド BOX 上位4頭", "ワイド", box(2, 4), "ワイドオッズは下限で計算"),
        plan("3連複 BOX 上位5頭", "3連複", box(3, 5)),
    ]
    if len(partners) >= 2:
        plans.append(plan(f"3連複 ◎1頭軸流し（相手最大{len(partners)}頭）", "3連複",
                          [f"{axis}-{a}-{b}" for a, b in combinations(partners, 2)]))
    return plans


def _pattern_plans(situation, horses, plan, rules=SITUATION_RULES):
    """判定パターンの優先プラン。"""
    pattern = situation["pattern"]
    if pattern == "A" and situation.get("horse") is not None:
        h = situation["horse"]
        mates = [x for x in horses if x != h][:rules["A"]["partners"]]
        out = [plan(f"単勝 穴馬{h}番1点", "単勝", [str(h)]),
               plan(f"複勝 穴馬{h}番1点", "複勝", [str(h)], "複勝オッズは下限で計算")]
        if mates:
            out.append(plan(f"ワイド 穴馬{h}番軸流し（相手最大{len(mates)}頭）", "ワイド",
                            [f"{h}-{m}" for m in mates], "ワイドオッズは下限で計算"))
        return out
    if pattern == "B" and len(horses) >= 2:
        axis, rest = horses[0], horses[1:]
        b = rules["B"]
        exacta = rest[:b["exacta_partners"]]
        out = [plan(f"馬連 ◎{axis}番から上位2頭（1〜2点）", "馬連", [f"{axis}-{p}" for p in rest[:2]]),
               plan(f"馬単 ◎{axis}番1着固定流し（相手最大{len(exacta)}頭）", "馬単", [f"{axis}-{p}" for p in exacta])]
        trio = rest[:b["trio_partners"]]
        if len(trio) >= 2:
            out.append(plan(f"3連複 ◎{axis}番1頭軸流し（相手最大{len(trio)}頭）", "3連複",
                            [f"{axis}-{x}-{y}" for x, y in combinations(trio, 2)]))
        return out
    if pattern == "C" and len(horses) >= 3:
        pairs3 = ["-".join(map(str, c)) for c in combinations(horses[:3], 2)]
        return [plan("ワイド BOX 上位3頭", "ワイド", pairs3, "ワイドオッズは下限で計算"),
                plan("馬連 BOX 上位3頭", "馬連", pairs3),
                plan("馬連 BOX 上位4頭", "馬連", ["-".join(map(str, c)) for c in combinations(horses[:4], 2)])]
    if pattern == "D" and len(horses) >= 2:
        axis = horses[0]
        return [plan("単勝 上位2頭（1〜2点）", "単勝", [str(h) for h in horses[:2]]),
                plan(f"ワイド ◎{axis}番から上位2頭（1〜2点）", "ワイド", [f"{axis}-{p}" for p in horses[1:3]],
                     "ワイドオッズは下限で計算")]
    return []


def _plan_type(plan):
    return plan["bets"][0]["type"] if plan["bets"] else None


def _is_allowed(plan, min_return_ratio):
    """除外条件: 組めない / 複勝が MAX_PLACE_HORSES 頭を超える / 最低回収率を満たさない。"""
    if not plan["bets"]:
        return False
    if sum(b["type"] == "複勝" for b in plan["bets"]) > MAX_PLACE_HORSES:
        return False
    return plan["min_return_ratio"] >= min_return_ratio - 1e-9


def _rank_plans(plans, pattern, use_expected_value, rules=SITUATION_RULES):
    """スコアの高い順に1つずつ選ぶ。スコア = 期待回収率(または的中率)+ 判定パターンのプランへの加点
    − 同じ券種がすでに上位に選ばれている数 × 減点。選んだ時点のスコアを plan["score"] に入れる。"""
    remaining, ranked, counts = list(plans), [], {}

    def score(p):
        base = p["expected_rate"] if use_expected_value else p["hit_prob"]
        bonus = rules["pattern_bonus"] if pattern and p.get("pattern") == pattern else 0.0
        return base + bonus - rules["same_type_penalty"] * counts.get(_plan_type(p), 0)

    while remaining:
        best = max(remaining, key=score)
        best["score"] = round(score(best), 4)
        ranked.append(best)
        counts[_plan_type(best)] = counts.get(_plan_type(best), 0) + 1
        remaining.remove(best)
    return ranked


def _empty_situation():
    return {"pattern": None, "tag": None, "matched": {}, "horse": None, "gaps": pd.DataFrame(),
            "score_basis": None, "view": pd.DataFrame()}


def plan_race(pred_df, volatility_level, odds_table, budget, min_return_ratio=DEFAULT_MIN_RETURN_RATIO,
              affinity_df=None, rules=SITUATION_RULES):
    """レース状況の判定と、並べ替え済みの候補プラン。戻り値: (situation, plans)

    min_return_ratio は REQUIRED_MIN_RETURN_RATIO より下げない(下回るプランは作らない・出さない)。
    affinity_df(相性モデルの predictions)がなくても動く(期待値は live の勝率だけ、パターン A は判定しない)。
    """
    ratio = max(min_return_ratio, REQUIRED_MIN_RETURN_RATIO)
    odds_table = _normalize_odds(odds_table)
    horses = candidate_horses(pred_df)
    if not horses or budget < UNIT:
        return _empty_situation(), []
    view = merge_model_views(pred_df, affinity_df, odds_table, rules)
    distribution = finish_distribution(view[["馬番", "期待値用勝率"]].rename(columns={"期待値用勝率": "推定勝率"}))
    if not distribution:
        return _empty_situation(), []

    def plan(name, t_type, combos, note=""):
        return _make_plan(name, t_type, combos, odds_table, budget, distribution, ratio, note)

    base = _base_plans(horses, volatility_level, plan)
    situation = classify_race(view, base, budget, ratio, rules)
    extra = _pattern_plans(situation, horses, plan, rules)
    for p in extra:
        p["pattern"], p["type_tag"] = situation["pattern"], situation["tag"]
    for p in base:
        p["pattern"], p["type_tag"] = None, None
    # 判定パターンのプランを先に置き、削られた結果同じ買い目になったプランは1つにまとめる
    plans, seen = [], set()
    for candidate in extra + base:
        key = frozenset((b["type"], b["combo"]) for b in candidate["bets"])
        if not _is_allowed(candidate, ratio) or key in seen:
            continue
        seen.add(key)
        plans.append(candidate)
    use_ev = bool(plans) and not any(p["estimated"] for p in plans)
    situation["score_basis"] = "期待値" if use_ev else "的中率"
    situation["view"] = view
    return situation, _rank_plans(plans, situation["pattern"], use_ev, rules)


def build_candidate_plans(pred_df, volatility_level, odds_table, budget, min_return_ratio=DEFAULT_MIN_RETURN_RATIO,
                          affinity_df=None):
    """AI上位馬と現在オッズから、配分を計算済みの候補プランをスコア順に返す(plan_race のプランだけ)。

    最低回収率を満たさないプラン・組めないプラン・複勝3頭以上のプランは含まない。
    各プランには score・pattern・type_tag(判定パターンの優先プランならタグ名)が付く。
    """
    return plan_race(pred_df, volatility_level, odds_table, budget, min_return_ratio, affinity_df)[1]


def format_plans(plans, min_return_ratio=DEFAULT_MIN_RETURN_RATIO, score_basis=None):
    floor = f"{max(min_return_ratio, REQUIRED_MIN_RETURN_RATIO):.0%}"
    basis = score_basis or "期待値"
    lines = [f"【計算済み候補プラン】（スコア順: {basis} ＋ 判定パターンの券種への加点 − 同じ券種の連続への減点。"
             f"各買い目の「購入額×オッズ」は購入合計の{floor}以上であることを検算済み。複勝は最大{MAX_PLACE_HORSES}頭）"]
    if not plans:
        lines.append("馬番が未確定、または予算不足のため、的中時に損をしないプランを組めませんでした。"
                     "枠順確定後に再度依頼するか、予算を見直してください。")
        return "\n".join(lines)
    if any(plan.get("estimated") for plan in plans):
        lines.append("※「推定オッズ」はオッズ未発表の券種について、モデルの確率と払戻率から計算した目安です。"
                     "発売後は実際のオッズが「必要オッズ」以上かを確認してください。")
    for i, plan in enumerate(plans, 1):
        tag = f"[TYPE: {plan['type_tag']}] " if plan.get("type_tag") else ""
        head = f"プラン{i}: {tag}{plan['name']}"
        if not plan["bets"]:
            lines.append(f"{head} → 予算内では組めません（合成オッズが低すぎる、または点数に対して予算不足）")
            continue
        extra = [f"購入合計{plan['total']:,}円", f"{len(plan['bets'])}点", f"的中率目安{plan['hit_prob']:.0%}",
                 f"的中時の最低回収率{plan['min_return_ratio']:.0%}"]
        if plan["estimated"]:
            extra.append("推定オッズ")
        else:
            # 推定オッズの期待回収率は払戻率そのものになり、判断材料にならないので出さない
            extra.append(f"期待回収率目安{plan['expected_rate']:.0%}")
        if plan.get("score") is not None:
            extra.append(f"スコア{plan['score']:.2f}")
        lines.append(f"{head}（{' / '.join(extra)}）" + (f" ※{plan['note']}" if plan["note"] else ""))
        for b in plan["bets"]:
            odds_label = "推定" if plan["estimated"] else ""
            lines.append(f"  {b['type']} {b['combo']}: {b['amount']:,}円 × {odds_label}{b['odds']:.1f}倍 = {b['return']:,}円"
                         f"（必要オッズ{b['required_odds']:.1f}倍以上）")
        if plan["dropped"]:
            lines.append(f"  最低回収率を守るため外した組み合わせ: {' / '.join(plan['dropped'])}")
        if plan["missing"]:
            lines.append(f"  オッズ未取得: {' / '.join(plan['missing'])}")
    return "\n".join(lines)


def _fmt_rank(v):
    return "-" if pd.isna(v) else f"{int(v)}"


def format_situation(situation):
    """レース状況の判定(タイプタグ・根拠の数値)と、2モデルの評価ギャップ。"""
    lines = ["【レース状況の判定】"]
    pattern = situation.get("pattern")
    matched = situation.get("matched") or {}
    if pattern:
        lines.append(f"TYPE: {situation['tag']}（パターン{pattern}: {PATTERN_NAMES[pattern]}）")
        if len(matched) > 1:
            lines.append(f"該当したパターン: {'・'.join(matched)}（優先順位 {' > '.join(PATTERN_PRIORITY)} で{pattern}を採用）")
        for p in PATTERN_PRIORITY:
            for reason in matched.get(p, []):
                lines.append(f"- [{p}] {reason}")
        lines.append(f"→ TYPE: {situation['tag']} のプランに加点して先頭に出しています。")
    else:
        lines.append("どのパターン（血統穴馬・1強2強・混戦・少額予算）にも当てはまりません。従来の候補をスコア順に並べています。")
    if situation.get("score_basis") == "的中率":
        lines.append("※オッズ未発表の券種があるため、期待値の代わりに的中率でスコアを付けています。")

    lines += ["", "【2モデルの評価ギャップ】（出馬表モデル=オッズや全体傾向を含む / 相性モデル=オッズ不使用の血統・馬体重・戦績。"
              "順位差がプラスなら相性モデルの方が高評価）"]
    gaps = situation.get("gaps")
    view = situation.get("view")
    if gaps is None or gaps.empty:
        has_aff = view is not None and not view.empty and view["相性勝率"].notna().any()
        lines.append("大きな食い違いはありません。" if has_aff else "相性モデルの評価がないため比較できません。")
    else:
        lines.append("馬番|馬名|出馬表モデル 順位・勝率|相性モデル 順位・勝率|順位差|勝率差|単勝")
        for _, r in gaps.iterrows():
            odds = "-" if pd.isna(r["オッズ"]) else f"{r['オッズ']:.1f}"
            lines.append(f"{int(r['馬番'])}|{r['馬名']}|{_fmt_rank(r['live順位'])}位・{_pct(r['live勝率'])}|"
                         f"{_fmt_rank(r['相性順位'])}位・{_pct(r['相性勝率'])}|{r['順位差']:+.0f}|{r['勝率差'] * 100:+.1f}pt|{odds}")
    return "\n".join(lines)


def format_odds_table(pred_df, odds_table, horses):
    """AI上位馬の単勝・複勝と、上位馬同士の馬連・ワイド・3連複の現在オッズ。"""
    odds_table = _normalize_odds(odds_table)
    lines = ["【現在オッズ（AI上位馬）】"]
    if not odds_table:
        lines.append("未発表です（発売前の可能性があります）。候補プランの推定オッズを使ってください。")
        return "\n".join(lines)

    def fmt(t_type, combo):
        info = odds_table.get(t_type, {}).get(normalize_combo(combo, t_type))
        if not info:
            return "-"
        lo, hi = info.get("odds"), info.get("odds_max")
        return f"{lo:.1f}" if not hi or hi == lo else f"{lo:.1f}-{hi:.1f}"

    names = dict(zip(pred_df["馬番"], pred_df["馬名"]))
    lines.append("馬番|馬名|単勝|複勝(下限-上限)")
    for h in horses:
        lines.append(f"{h}|{names.get(h, '')}|{fmt('単勝', str(h))}|{fmt('複勝', str(h))}")
    lines.append("組み合わせ|馬連|ワイド(下限-上限)")
    for a, b in combinations(horses, 2):
        combo = f"{min(a, b)}-{max(a, b)}"
        lines.append(f"{combo}|{fmt('馬連', combo)}|{fmt('ワイド', combo)}")
    trio = [f"{'-'.join(str(n) for n in sorted(c))}" for c in combinations(horses[:5], 3)]
    lines.append("3連複: " + " / ".join(f"{c}={fmt('3連複', c)}" for c in trio))
    return "\n".join(lines)


def format_steps(prediction, volatility, model_metrics=None):
    # Step3の資金配分シミュレーションはガミを考慮していないので渡さない
    return format_steps_for_prompt(prediction, volatility, metrics=model_metrics, include_allocation=False)


def build_gami_free_prompt(race_id, race_info, prediction, volatility, odds_table, budget, use_full_budget=False,
                           model_metrics=None, min_return_ratio=DEFAULT_MIN_RETURN_RATIO, affinity=None) -> str:
    """affinity は相性モデルの予測結果(affinity_model.predict の戻り値)。なくても組める。"""
    pred_df = prediction["predictions"]
    level = volatility["level"] if volatility else None
    ratio = max(min_return_ratio, REQUIRED_MIN_RETURN_RATIO)
    affinity_df = (affinity or {}).get("predictions")
    situation, plans = plan_race(pred_df, level, odds_table, budget, ratio, affinity_df)
    horses = candidate_horses(pred_df)
    budget_rule = f"予算{budget}円を使い切ってください。" if use_full_budget else f"予算{budget}円は上限です。"
    sections = [
        build_persona_prompt(ratio),
        f"【対象レース】ID: {race_id} / {(race_info or {}).get('name', '')}",
        f"条件: {(race_info or {}).get('data', '')}",
        f"予算: {budget}円。{budget_rule}",
        f"的中時の最低回収率: {ratio:.0%}",
        "",
        format_steps(prediction, volatility, model_metrics),
        "",
        format_situation(situation),
        "",
        format_plans(plans, ratio, situation.get("score_basis")),
        "",
        format_odds_table(pred_df, odds_table, horses),
        "",
        "【出力要件】前置きは書かず、以下の形式で出力してください。",
        "■ 採用プランと理由（どの候補プランを採用したか。TYPEタグ・判定の根拠の数値・2モデルの評価ギャップを使って、"
        "なぜこの券種を先頭にしたのかを2〜4行で）",
        "■ 買い目と金額（券種・組み合わせ・金額）",
        "■ ガミ検算表（買い目ごとに「購入額×オッズ=払戻」、購入合計に対する回収率、必要オッズを並べる）",
        "■ 注意点（オッズ未発表の場合の確認事項、評価が割れている馬のリスクなど）",
        "最後に必ず集計用JSONをこの形式で出力: ```json\n{\"bets\":[{\"type\":\"ワイド\",\"combo\":\"1-2\",\"amount\":100}]}\n```",
    ]
    return "\n".join(sections)

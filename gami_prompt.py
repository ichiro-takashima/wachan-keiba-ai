"""的中率を優先しつつ、的中時の損失を許容範囲に抑える買い方をAIに組ませるためのプロンプト。

能力評価はアプリ内の機械学習モデル(Step1〜3)に任せ、AIには現在オッズとの突き合わせ・
金額配分だけを担当させる。配分はこちらで計算して検算済みの候補として渡す。
オッズが未発表の券種は、モデルの確率から推定したオッズで暫定の配分を組む。
"""
import math
from itertools import combinations, permutations

import pandas as pd

from ml_ui import PARTNER_COUNT_BY_LEVEL, format_steps_for_prompt
from tickets import normalize_combo, normalize_ticket_type

# scraper.scrape_realtime_odds に渡す券種名(オッズAPIの名前)
GAMI_ODDS_TICKET_TYPES = ["単勝", "複勝", "馬連", "ワイド", "三連複"]
UNIT = 100
# 的中時の最低回収率(払戻÷購入合計)の選択肢と既定値。100%で「ガミを一切許さない」
MIN_RETURN_RATIO_OPTIONS = [1.0, 0.9, 0.8, 0.7, 0.6]
DEFAULT_MIN_RETURN_RATIO = 1.0
# この点数以上のプランを優先して提示する(1〜2点への集中で的中率が下がるのを防ぐ)
PREFERRED_MIN_POINTS = 3
# JRAの券種別払戻率。未発表オッズの推定に使う
PAYOUT_RATE = {"単勝": 0.80, "複勝": 0.80, "馬連": 0.775, "ワイド": 0.775, "3連複": 0.75}


def build_persona_prompt(min_return_ratio):
    floor = f"{min_return_ratio:.0%}"
    tolerance_rule = (
        "" if min_return_ratio >= 1.0
        else "\n   この範囲内のガミは許容します。ガミを完全に消すために点数を削らないでください。"
    )
    return f"""あなたは競馬の資金配分に特化したアシスタントです。
各馬の能力評価はすでにアプリ内の機械学習モデル（Step1〜3）が行っています。あなたの役割は、その評価と現在オッズを使い、
的中率を優先しながら、的中時の損失を許容範囲に抑える券種・組み合わせ・金額配分を組み立てることです。

# 守るべきルール
1. 採用するすべての買い目について「購入額 × オッズ（複勝・ワイドは下限） ≥ 購入合計 × {floor}」を満たすこと。{tolerance_rule}
2. 候補プランは{PREFERRED_MIN_POINTS}点以上のものを先に、それぞれ的中率目安の高い順に並べています。原則として先頭のプランを採用すること。
   {PREFERRED_MIN_POINTS}点未満に絞るのは、{PREFERRED_MIN_POINTS}点以上のプランが組めない場合だけにすること。
3. 金額の計算は【計算済み候補プラン】の数字（検算済み）をそのまま使うのが安全。独自に組む場合は、ルール1を買い目ごとに検算すること。
4. 推定オッズで組まれたプランも買い目として提案してよい。オッズ未発表は見送りの理由にしないこと。
   その場合は、発売開始後に必要オッズを満たしているか確認するよう一言添えること。
5. 見送り（0円）は、すべての候補プランが組めない場合だけにすること。
6. 金額は100円単位。馬は馬番で指定する。断定的な表現（「絶対」「鉄板」など）は使わない。
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


def build_candidate_plans(pred_df, volatility_level, odds_table, budget, min_return_ratio=DEFAULT_MIN_RETURN_RATIO):
    """AI上位馬と現在オッズから、配分を計算済みの候補プランを作る。

    PREFERRED_MIN_POINTS点以上を先に、それぞれ的中率の高い順に返す。組めないプランは末尾。
    """
    odds_table = _normalize_odds(odds_table)
    horses = candidate_horses(pred_df)
    distribution = finish_distribution(pred_df)
    if not horses or not distribution or budget < UNIT:
        return []
    axis = horses[0]
    partners = horses[1:1 + PARTNER_COUNT_BY_LEVEL.get(volatility_level, 4)]

    def plan(name, t_type, combos, note=""):
        return _make_plan(name, t_type, combos, odds_table, budget, distribution, min_return_ratio, note)

    def box(size, n):
        return ["-".join(str(h) for h in c) for c in combinations(horses[:n], size)]

    plans = [
        plan("単勝◎1点", "単勝", [str(axis)]),
        plan("複勝◎1点", "複勝", [str(axis)], "複勝オッズは下限で計算"),
        plan("単勝 上位3頭", "単勝", [str(h) for h in horses[:3]]),
        plan("複勝 上位3頭", "複勝", [str(h) for h in horses[:3]], "複勝オッズは下限で計算"),
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
    # 削られた結果、同じ買い目になったプランは1つにまとめる(「複勝 上位3頭」が1点になった場合など)
    unique_plans, seen = [], set()
    for candidate in plans:
        key = frozenset((b["type"], b["combo"]) for b in candidate["bets"])
        if key and key in seen:
            continue
        seen.add(key)
        unique_plans.append(candidate)
    return sorted(
        unique_plans,
        key=lambda p: (len(p["bets"]) >= PREFERRED_MIN_POINTS, p.get("hit_prob", -1)),
        reverse=True,
    )


def format_plans(plans, min_return_ratio=DEFAULT_MIN_RETURN_RATIO):
    floor = f"{min_return_ratio:.0%}"
    lines = [f"【計算済み候補プラン】（{PREFERRED_MIN_POINTS}点以上を優先し、的中率目安の高い順。"
             f"各買い目の「購入額×オッズ」は購入合計の{floor}以上であることを検算済み）"]
    if not plans:
        lines.append("馬番が未確定、または予算不足のため計算できませんでした。枠順確定後に再度依頼してください。")
        return "\n".join(lines)
    if any(plan.get("estimated") for plan in plans):
        lines.append("※「推定オッズ」はオッズ未発表の券種について、モデルの確率と払戻率から計算した目安です。"
                     "発売後は実際のオッズが「必要オッズ」以上かを確認してください。")
    for i, plan in enumerate(plans, 1):
        head = f"プラン{i}: {plan['name']}"
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
                           model_metrics=None, min_return_ratio=DEFAULT_MIN_RETURN_RATIO) -> str:
    pred_df = prediction["predictions"]
    level = volatility["level"] if volatility else None
    plans = build_candidate_plans(pred_df, level, odds_table, budget, min_return_ratio)
    horses = candidate_horses(pred_df)
    budget_rule = f"予算{budget}円を使い切ってください。" if use_full_budget else f"予算{budget}円は上限です。"
    sections = [
        build_persona_prompt(min_return_ratio),
        f"【対象レース】ID: {race_id} / {(race_info or {}).get('name', '')}",
        f"条件: {(race_info or {}).get('data', '')}",
        f"予算: {budget}円。{budget_rule}",
        f"的中時の最低回収率: {min_return_ratio:.0%}",
        "",
        format_steps(prediction, volatility, model_metrics),
        "",
        format_plans(plans, min_return_ratio),
        "",
        format_odds_table(pred_df, odds_table, horses),
        "",
        "【出力要件】前置きは書かず、以下の形式で出力してください。",
        "■ 採用プランと理由（どの候補プランを採用したか、的中率目安と合わせて2〜3行で）",
        "■ 買い目と金額（券種・組み合わせ・金額）",
        "■ ガミ検算表（買い目ごとに「購入額×オッズ=払戻」、購入合計に対する回収率、必要オッズを並べる）",
        "■ 注意点（ガミになる買い目、オッズ未発表の場合の確認事項があれば）",
        "最後に必ず集計用JSONをこの形式で出力: ```json\n{\"bets\":[{\"type\":\"ワイド\",\"combo\":\"1-2\",\"amount\":100}]}\n```",
    ]
    return "\n".join(sections)

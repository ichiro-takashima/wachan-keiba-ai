"""「どの買い目が的中してもガミにならない」買い方をAIに組ませるためのプロンプト。

能力評価はアプリ内の機械学習モデル(Step1〜3)に任せ、AIには現在オッズとの突き合わせ・
金額配分だけを担当させる。ガミ回避の配分はこちらで計算して検算済みの候補として渡す。
"""
import math
from itertools import combinations

import pandas as pd

from ml_ui import PARTNER_COUNT_BY_LEVEL, format_steps_for_prompt
from tickets import normalize_combo, normalize_ticket_type

# scraper.scrape_realtime_odds に渡す券種名(オッズAPIの名前)
GAMI_ODDS_TICKET_TYPES = ["単勝", "複勝", "馬連", "ワイド", "三連複"]
UNIT = 100

GAMI_PERSONA_PROMPT = """あなたは競馬の資金配分に特化したアシスタントです。
各馬の能力評価はすでにアプリ内の機械学習モデル（Step1〜3）が行っています。あなたの役割は、その評価と現在オッズだけを使い、
「どの買い目が的中しても払戻が購入合計を下回らない（ガミが出ない）」券種・組み合わせ・金額配分を組み立てることです。

# 守るべきルール
1. 採用するすべての買い目について「購入額 × オッズ（複勝・ワイドは下限） ≥ 購入合計」を満たすこと。
2. 合成オッズ（1 / Σ(1/各オッズ)）が1.0倍以下になる組み合わせは、どう配分してもガミになるので組まないこと。
   点数を増やすほど合成オッズは下がるため、ガミを避けるには点数を絞るか、オッズの高い組み合わせに寄せること。
3. 金額の計算は【ガミ回避の計算済み候補プラン】の数字（検算済み）をそのまま使うのが安全。独自に組む場合は、上の条件を買い目ごとに検算すること。
4. 期待値（推定勝率×単勝オッズ）が1.0を大きく下回る人気馬だけに予算を集中させないこと。妙味がなければ見送り（0円）も選択肢とする。
5. 金額は100円単位。馬は馬番で指定する。断定的な表現（「絶対」「鉄板」など）は使わない。
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


def allocate_no_gami(odds_list, budget, unit=UNIT):
    """どれが的中しても払戻≥購入合計になる100円単位の配分。組めなければNone。

    合計Tを予算から順に下げ、各買い目に「T円を払い戻せる最低額」(100円単位で切り上げ)を置いて
    合計がTに収まる最大のTを探す。余った分は払戻がそろうよう(オッズに反比例する理想額に近づくよう)
    100円ずつ足す。足しても払戻が増えるだけなので条件は崩れない。
    """
    if not odds_list or budget < unit * len(odds_list) or synthetic_odds(odds_list) <= 1.0:
        return None
    share = synthetic_odds(odds_list)
    for total in range(budget // unit * unit, unit * len(odds_list) - 1, -unit):
        amounts = [math.ceil(total / (o * unit) - 1e-9) * unit for o in odds_list]
        slack = total - sum(amounts)
        if slack < 0:
            continue
        ideal = [total * share / o for o in odds_list]
        for _ in range(slack // unit):
            i = max(range(len(amounts)), key=lambda k: ideal[k] - amounts[k])
            amounts[i] += unit
        return amounts
    return None


def _make_plan(name, t_type, combos, odds_table, budget, note=""):
    """組み合わせの強い順リストから、ガミ回避で組める最大点数のプランを作る(弱い方から削る)。"""
    combos = [normalize_combo(c, t_type) for c in combos]
    priced = [(c, _odds(odds_table, t_type, c)) for c in combos]
    missing = [c for c, o in priced if o is None]
    priced = [(c, o) for c, o in priced if o is not None]
    for k in range(len(priced), 0, -1):
        use = priced[:k]
        amounts = allocate_no_gami([o for _, o in use], budget)
        if amounts is None:
            continue
        total = sum(amounts)
        bets = [
            {"type": t_type, "combo": c, "amount": a, "odds": o, "return": int(round(a * o))}
            for (c, o), a in zip(use, amounts)
        ]
        dropped = [c for c, _ in priced[k:]]
        return {
            "name": name, "note": note, "bets": bets, "total": total,
            "synthetic_odds": synthetic_odds([o for _, o in use]),
            "min_return": min(b["return"] for b in bets),
            "dropped": dropped, "missing": missing,
        }
    return {"name": name, "note": note, "bets": [], "dropped": [c for c, _ in priced], "missing": missing}


def build_candidate_plans(pred_df, volatility_level, odds_table, budget):
    """AI上位馬と現在オッズから、ガミ回避の配分を計算済みの候補プランを券種ごとに作る。"""
    odds_table = _normalize_odds(odds_table)
    horses = candidate_horses(pred_df)
    if not horses or not odds_table or budget < UNIT:
        return []
    axis = horses[0]
    partners = horses[1:1 + PARTNER_COUNT_BY_LEVEL.get(volatility_level, 4)]
    win_prob = dict(zip(pred_df["馬番"], pred_df["推定勝率"]))
    top3_prob = dict(zip(pred_df["馬番"], pred_df["3着内確率"]))

    plans = [
        _make_plan("単勝◎1点", "単勝", [str(axis)], odds_table, budget),
        _make_plan("複勝◎1点", "複勝", [str(axis)], odds_table, budget, "複勝オッズは下限で計算"),
        _make_plan("単勝 上位2頭", "単勝", [str(h) for h in horses[:2]], odds_table, budget),
        _make_plan("単勝 上位3頭", "単勝", [str(h) for h in horses[:3]], odds_table, budget),
        _make_plan(f"馬連 ◎流し（相手最大{len(partners)}頭）", "馬連",
                   [f"{axis}-{p}" for p in partners], odds_table, budget),
        _make_plan(f"ワイド ◎流し（相手最大{len(partners)}頭）", "ワイド",
                   [f"{axis}-{p}" for p in partners], odds_table, budget, "ワイドオッズは下限で計算"),
    ]
    if len(partners) >= 2:
        # 相手同士の組は、AI順位の合計が小さい(強い馬同士の)順に並べる
        pairs = sorted(combinations(range(len(partners)), 2), key=lambda ij: (sum(ij), ij))
        plans.append(_make_plan(
            f"3連複 ◎1頭軸流し（相手最大{len(partners)}頭）", "3連複",
            [f"{axis}-{partners[i]}-{partners[j]}" for i, j in pairs], odds_table, budget,
        ))

    for plan in plans:
        if not plan["bets"]:
            continue
        # 単勝・複勝はモデルの確率から的中率と期待回収率の目安を出せる
        if plan["bets"][0]["type"] in ("単勝", "複勝"):
            prob_map = win_prob if plan["bets"][0]["type"] == "単勝" else top3_prob
            probs = [prob_map.get(int(b["combo"])) for b in plan["bets"]]
            if all(p is not None and pd.notna(p) for p in probs):
                plan["hit_prob"] = min(sum(probs), 1.0) if plan["bets"][0]["type"] == "単勝" else probs[0]
                plan["expected_rate"] = sum(p * b["return"] for p, b in zip(probs, plan["bets"])) / plan["total"]
    return plans


def format_plans(plans):
    lines = ["【ガミ回避の計算済み候補プラン】（各買い目の「購入額×オッズ」は購入合計以上であることを検算済み）"]
    if not plans:
        lines.append("現在オッズが取得できないため計算できませんでした（発売前の可能性があります）。オッズが出てから再度依頼するか、見送りを検討してください。")
        return "\n".join(lines)
    for i, plan in enumerate(plans, 1):
        head = f"プラン{i}: {plan['name']}"
        if not plan["bets"]:
            lines.append(f"{head} → 予算内ではガミを避けて組めません（合成オッズが低すぎる、または点数に対して予算不足）")
            continue
        extra = [f"購入合計{plan['total']:,}円", f"{len(plan['bets'])}点", f"合成オッズ{plan['synthetic_odds']:.2f}倍",
                 f"最低払戻{plan['min_return']:,}円"]
        if "hit_prob" in plan:
            extra.append(f"モデル的中率目安{plan['hit_prob']:.0%}")
            extra.append(f"期待回収率目安{plan['expected_rate']:.0%}")
        lines.append(f"{head}（{' / '.join(extra)}）" + (f" ※{plan['note']}" if plan["note"] else ""))
        for b in plan["bets"]:
            lines.append(f"  {b['type']} {b['combo']}: {b['amount']:,}円 × {b['odds']:.1f}倍 = {b['return']:,}円 ≥ {plan['total']:,}円")
        if plan["dropped"]:
            lines.append(f"  ガミ回避のため外した組み合わせ: {' / '.join(plan['dropped'])}")
        if plan["missing"]:
            lines.append(f"  オッズ未取得: {' / '.join(plan['missing'])}")
    return "\n".join(lines)


def format_odds_table(pred_df, odds_table, horses):
    """AI上位馬の単勝・複勝と、上位馬同士の馬連・ワイド・3連複の現在オッズ。"""
    odds_table = _normalize_odds(odds_table)
    lines = ["【現在オッズ（AI上位馬）】"]
    if not odds_table:
        lines.append("取得できませんでした。")
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
                           model_metrics=None) -> str:
    pred_df = prediction["predictions"]
    level = volatility["level"] if volatility else None
    plans = build_candidate_plans(pred_df, level, odds_table, budget)
    horses = candidate_horses(pred_df)
    budget_rule = (
        f"予算{budget}円を使い切ってください（ガミ回避できる範囲で）。"
        if use_full_budget else f"予算{budget}円は上限です。ガミが避けられない・妙味がない場合は一部だけ、または0円でも構いません。"
    )
    sections = [
        GAMI_PERSONA_PROMPT,
        f"【対象レース】ID: {race_id} / {(race_info or {}).get('name', '')}",
        f"条件: {(race_info or {}).get('data', '')}",
        f"予算: {budget}円。{budget_rule}",
        "",
        format_steps(prediction, volatility, model_metrics),
        "",
        format_plans(plans),
        "",
        format_odds_table(pred_df, odds_table, horses),
        "",
        "【出力要件】前置きは書かず、以下の形式で出力してください。",
        "■ 採用プランと理由（どの候補プランを採用・組み合わせたか、点数を絞った理由を2〜3行で）",
        "■ 買い目と金額（券種・組み合わせ・金額）",
        "■ ガミ検算表（買い目ごとに「購入額×オッズ=払戻」と購入合計を並べ、全買い目で払戻≥購入合計であることを示す）",
        "■ 見送り判断（買う価値が低い場合はその理由）",
        "最後に必ず集計用JSONをこの形式で出力: ```json\n{\"bets\":[{\"type\":\"ワイド\",\"combo\":\"1-2\",\"amount\":100}]}\n```",
    ]
    return "\n".join(sections)

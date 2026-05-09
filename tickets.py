import json
import re


TICKET_TYPE_OPTIONS = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
COMBO_COUNT_REQUIRED_TYPES = {"枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"}
BETTING_METHOD_OPTIONS = ["通常", "フォーメーション", "ながし", "ボックス"]


def normalize_ticket_type(t):
    return str(t).replace("三連", "3連").strip()


def normalize_combo(combo, ticket_type):
    normalized = str(combo).replace("ー", "-").replace(" ", "")
    if ticket_type in ["馬連", "ワイド", "三連複"]:
        return "-".join(sorted(normalized.split("-")))
    normalized = str(combo).replace("ー", "-").replace(" ", "").replace("→", "-").replace(">", "-")
    t_type = normalize_ticket_type(ticket_type)
    if t_type in ["馬連", "ワイド", "3連複"]:
        parts = [p for p in normalized.split("-") if p]
        try:
            return "-".join(sorted(parts, key=int))
        except Exception:
            return "-".join(sorted(parts))
    return normalized


def create_ticket_request():
    return {
        "ticket_type": "ワイド",
        "budget": 500,
        "combo_count": 3,
        "betting_method": "通常",
        "axis_horse": "",
    }


def ticket_type_requires_combo_count(ticket_type):
    return ticket_type in COMBO_COUNT_REQUIRED_TYPES


def ticket_type_supports_betting_method(ticket_type):
    return ticket_type not in ["単勝", "複勝"]


def build_ticket_plan_text(plan_mode, ticket_requests, fallback_budget):
    if plan_mode != "カスタム":
        return (
            "波乱度の判定に基づき、必ず以下のルールで穴馬を含め買い目を厚く構築してください。\n"
            "・判定が「堅い」の場合: 上位2頭を軸にした3連複フォーメーションと、3連単数点を提案すること（相手は穴馬まで5頭以上）。\n"
            "・判定が「標準」の場合: 1頭軸から相手5〜6頭への3連複流し（15点等）と、3連単フォーメーションを厚めに提案すること。\n"
            "・判定が「荒れる」の場合: 上位評価馬と穴馬を絡めた3連複ボックス（6頭以上）や、穴馬からの3連単フォーメーションを提案すること。\n"
            "・合計金額は予算以内に収めること。"
        ), fallback_budget

    valid_requests = []
    for req in ticket_requests:
        ticket_type = req.get("ticket_type")
        request_budget = int(req.get("budget", 0) or 0)
        combo_count = int(req.get("combo_count", 0) or 0)
        betting_method = req.get("betting_method", "通常")
        if not ticket_type or request_budget <= 0:
            continue
        if betting_method not in BETTING_METHOD_OPTIONS:
            betting_method = "通常"
        valid_requests.append({
            "ticket_type": ticket_type,
            "budget": request_budget,
            "combo_count": combo_count,
            "betting_method": betting_method,
            "axis_horse": req.get("axis_horse", "").strip(),
        })

    total_budget = sum(req["budget"] for req in valid_requests)
    lines = ["ユーザーの買い方希望に従って、以下の条件を満たす買い目を提案してください。"]
    for idx, req in enumerate(valid_requests, start=1):
        line = f"・条件{idx}: {req['ticket_type']}を{req['budget']}円分"
        if ticket_type_supports_betting_method(req["ticket_type"]):
            line += f"、方式は{req['betting_method']}"
        if req.get("axis_horse"):
            line += f"、軸馬は{req['axis_horse']}番"
        if ticket_type_requires_combo_count(req["ticket_type"]):
            line += f"、組数は{req['combo_count']}点ちょうど"
        lines.append(line)
    lines.extend([
        "・上記で指定された券種以外は提案しないこと。",
        "・各条件の予算を超えないこと。",
        "・券種ごとに指定された方式（通常／フォーメーション／ながし／ボックス）を守ること。",
        "・組数指定がある券種は、指定された組数ちょうどで買い目を構成すること。",
        "・金額は1点あたり必ず100円の倍数（100円、200円、300円...）とし、150円や250円のような端数は絶対に出さないこと。",
    ])
    return "\n".join(lines), total_budget


def format_ticket_plan_for_context(plan_mode, ticket_requests, fallback_budget):
    if plan_mode != "カスタム":
        return f"買い方希望: なし（おまかせ）。全体予算{fallback_budget}円で、現在の予想モデルに基づいて提案。"

    valid_requests = []
    for req in ticket_requests:
        ticket_type = req.get("ticket_type")
        request_budget = int(req.get("budget", 0) or 0)
        combo_count = int(req.get("combo_count", 0) or 0)
        betting_method = req.get("betting_method", "通常")
        axis_horse = req.get("axis_horse", "").strip()
        if not ticket_type or request_budget <= 0:
            continue
        line = f"・{ticket_type}: {request_budget}円"
        if ticket_type_supports_betting_method(ticket_type):
            if betting_method not in BETTING_METHOD_OPTIONS:
                betting_method = "通常"
            line += f" / {betting_method}"
        if axis_horse:
            line += f" / 軸:{axis_horse}"
        if ticket_type_requires_combo_count(ticket_type):
            line += f" / {combo_count}点"
        valid_requests.append(line)

    total_budget = sum(int(req.get("budget", 0) or 0) for req in ticket_requests)
    return "買い方希望:\n" + "\n".join(valid_requests) + f"\n合計予算: {total_budget}円"


def parse_ids_input(raw_text):
    if not raw_text:
        return []
    tokens = re.split(r"[\s,]+", raw_text.strip())
    return [token for token in tokens if token]


def calculate_return(bets, payouts):
    total_bet, total_return, hits = 0, 0, 0
    ticket_stats = {}
    norm_payouts = {normalize_ticket_type(k): v for k, v in payouts.items()}

    for bet in bets:
        raw_type = bet.get("type", "")
        t_type = normalize_ticket_type(raw_type)
        bet_combo = str(bet.get("combo", ""))
        try:
            amount = int(bet.get("amount", 0))
        except Exception:
            continue

        if raw_type not in ticket_stats:
            ticket_stats[raw_type] = {"bet": 0, "return": 0, "hits": 0}
        ticket_stats[raw_type]["bet"] += amount
        total_bet += amount

        is_hit = False
        if t_type in norm_payouts:
            for payout in norm_payouts[t_type]:
                if normalize_combo(bet_combo, t_type) == normalize_combo(payout["combo"], t_type):
                    return_amount = int((amount / 100) * payout["pay"])
                    total_return += return_amount
                    ticket_stats[raw_type]["return"] += return_amount
                    is_hit = True
                    break

        if is_hit:
            hits += 1
            ticket_stats[raw_type]["hits"] += 1

    return total_bet, total_return, hits, ticket_stats


def extract_bets_from_text(ai_text):
    if not ai_text:
        return []
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", ai_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        payload = json.loads(match.group(1))
        bets = payload.get("bets", [])
        return bets if isinstance(bets, list) else []
    except Exception:
        return []


def build_reflection_text(predicted_bets, payouts):
    predicted_ticket_types = {bet.get("type", "") for bet in predicted_bets if bet.get("type")}
    payout_ticket_types = set(payouts.keys())
    missing_ticket_types = sorted(list(payout_ticket_types - predicted_ticket_types))

    hit_combos = set()
    predicted_combos_by_type = {}
    for bet in predicted_bets:
        ticket_type = bet.get("type", "")
        combo = bet.get("combo", "")
        if not ticket_type or not combo:
            continue
        predicted_combos_by_type.setdefault(ticket_type, set()).add(normalize_combo(combo, ticket_type))

    uncovered_examples = []
    for ticket_type, payout_list in payouts.items():
        predicted_set = predicted_combos_by_type.get(ticket_type, set())
        for payout in payout_list:
            payout_combo = normalize_combo(payout.get("combo", ""), ticket_type)
            if payout_combo in predicted_set:
                hit_combos.add(f"{ticket_type}:{payout_combo}")
            elif len(uncovered_examples) < 5:
                uncovered_examples.append(f"{ticket_type} {payout.get('combo', '')}")

    reflection_lines = []
    if missing_ticket_types:
        reflection_lines.append("・今回買っていないのに的中が出ていた券種: " + "、".join(missing_ticket_types))
    if uncovered_examples:
        reflection_lines.append("・買い目に入っていなかった主な的中組み合わせ: " + " / ".join(uncovered_examples))

    if not reflection_lines:
        reflection_lines.append("・券種と組み合わせの観点では、予想は結果に近い構成でした。")

    reflection_lines.append("・次回改善案: ◎○の軸は維持しつつ、ワイド/三連複の保険を1〜2点追加して取りこぼしを減らす。")
    return "\n".join(reflection_lines)

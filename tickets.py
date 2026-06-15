import json
import re


def normalize_ticket_type(ticket_type):
    return str(ticket_type or "").replace("三連", "3連").strip()


def normalize_combo(combo, ticket_type):
    normalized = (
        str(combo or "")
        .replace("－", "-")
        .replace("ー", "-")
        .replace("→", "-")
        .replace(">", "-")
        .replace(" ", "")
    )
    t_type = normalize_ticket_type(ticket_type)
    parts = [part for part in normalized.split("-") if part]
    if t_type in ["馬連", "ワイド", "3連複"]:
        try:
            return "-".join(sorted(parts, key=int))
        except Exception:
            return "-".join(sorted(parts))
    return normalized


def parse_ids_input(raw_text):
    if not raw_text:
        return []
    tokens = re.split(r"[\s,]+", raw_text.strip())
    return [token for token in tokens if token]


def normalize_bet_amounts(bets):
    normalized_bets = []
    for bet in bets or []:
        if not isinstance(bet, dict):
            continue
        normalized_bet = dict(bet)
        try:
            amount = int(float(str(normalized_bet.get("amount", 0)).replace(",", "")))
        except Exception:
            amount = 0
        if amount <= 0:
            normalized_bet["amount"] = 0
        else:
            normalized_bet["amount"] = max(100, ((amount + 50) // 100) * 100)
        normalized_bets.append(normalized_bet)
    return normalized_bets


def extract_bets_from_text(ai_text):
    if not ai_text:
        return []
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", ai_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        payload = json.loads(match.group(1))
        bets = payload.get("bets", [])
        return normalize_bet_amounts(bets) if isinstance(bets, list) else []
    except Exception:
        return []


def calculate_return(bets, payouts):
    total_bet, total_return, hits = 0, 0, 0
    ticket_stats = {}
    norm_payouts = {normalize_ticket_type(k): v for k, v in (payouts or {}).items()}

    for bet in normalize_bet_amounts(bets):
        raw_type = bet.get("type", "")
        ticket_type = normalize_ticket_type(raw_type)
        bet_combo = str(bet.get("combo", ""))
        try:
            amount = int(bet.get("amount", 0))
        except Exception:
            continue

        if amount <= 0:
            continue

        if raw_type not in ticket_stats:
            ticket_stats[raw_type] = {"bet": 0, "return": 0, "hits": 0}
        ticket_stats[raw_type]["bet"] += amount
        total_bet += amount

        is_hit = False
        if ticket_type in norm_payouts:
            for payout in norm_payouts[ticket_type]:
                if normalize_combo(bet_combo, ticket_type) == normalize_combo(payout.get("combo", ""), ticket_type):
                    return_amount = int((amount / 100) * int(payout.get("pay", 0)))
                    total_return += return_amount
                    ticket_stats[raw_type]["return"] += return_amount
                    is_hit = True
                    break

        if is_hit:
            hits += 1
            ticket_stats[raw_type]["hits"] += 1

    return total_bet, total_return, hits, ticket_stats


def build_reflection_text(predicted_bets, payouts):
    predicted_ticket_types = {bet.get("type", "") for bet in predicted_bets if bet.get("type")}
    payout_ticket_types = set((payouts or {}).keys())
    missing_ticket_types = sorted(list(payout_ticket_types - predicted_ticket_types))

    predicted_combos_by_type = {}
    for bet in predicted_bets:
        ticket_type = bet.get("type", "")
        combo = bet.get("combo", "")
        if ticket_type and combo:
            predicted_combos_by_type.setdefault(ticket_type, set()).add(normalize_combo(combo, ticket_type))

    uncovered_examples = []
    for ticket_type, payout_list in (payouts or {}).items():
        predicted_set = predicted_combos_by_type.get(ticket_type, set())
        for payout in payout_list:
            payout_combo = normalize_combo(payout.get("combo", ""), ticket_type)
            if payout_combo not in predicted_set and len(uncovered_examples) < 5:
                uncovered_examples.append(f"{ticket_type} {payout.get('combo', '')}")

    lines = []
    if missing_ticket_types:
        lines.append("買っていない券種で的中が出ていました: " + "、".join(missing_ticket_types))
    if uncovered_examples:
        lines.append("買い目に入っていなかった主な的中組み合わせ: " + " / ".join(uncovered_examples))
    if not lines:
        lines.append("券種と組み合わせの観点では、予想は結果に近い構成でした。")
    lines.append("次回改善案: 軸馬の評価は維持しつつ、相手候補や保険の点数をAI判断で再調整してください。")
    return "\n".join(lines)

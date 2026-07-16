"""netkeibaの馬券購入画面を参考にした「買い方セレクト」用のロジック。

ユーザーは馬番そのものではなく、券種・方式・目標点数・配分金額を指定する。
実際にどの馬番を対象にするかはAIがレース分析に基づいて決定する。
"""

TICKET_ARITY = {
    "単勝": 1,
    "複勝": 1,
    "枠連": 2,
    "馬連": 2,
    "ワイド": 2,
    "馬単": 2,
    "三連複": 3,
    "三連単": 3,
}

ORDERED_TICKETS = {"馬単", "三連単"}

METHODS_BY_TICKET = {
    "単勝": ["通常"],
    "複勝": ["通常"],
    "枠連": ["通常", "ボックス", "流し"],
    "馬連": ["通常", "ボックス", "流し", "フォーメーション"],
    "ワイド": ["通常", "ボックス", "流し", "フォーメーション"],
    "馬単": ["通常", "ボックス", "流し", "フォーメーション"],
    "三連複": ["通常", "ボックス", "流し", "フォーメーション"],
    "三連単": ["通常", "ボックス", "流し", "フォーメーション"],
}


def compute_amounts(point_count, amount_mode, amount_value):
    """点数と金額指定方法から、各点の購入金額（100円単位）のリストを返す。"""
    if point_count <= 0:
        return []
    try:
        value = int(amount_value)
    except Exception:
        value = 0
    if value <= 0:
        return [0] * point_count

    if amount_mode == "per_point":
        per = max(100, ((value + 50) // 100) * 100)
        return [per] * point_count

    # total: 合計金額を均等割り（100円単位、端数は前方の点に上乗せ）
    units = value // 100
    base, remainder = divmod(units, point_count)
    return [(base + (1 if i < remainder else 0)) * 100 for i in range(point_count)]


def block_total_amount(block):
    amounts = compute_amounts(block.get("points", 0), block.get("amount_mode"), block.get("amount_value"))
    return sum(amounts)


def describe_block(block):
    """AIへの指示文用に、買い方ブロックを1行の日本語で説明する。"""
    ticket_type = block.get("ticket_type", "")
    method = block.get("method", "")
    arity = TICKET_ARITY.get(ticket_type, 0)
    ordered = ticket_type in ORDERED_TICKETS

    detail = ""
    if method == "流し":
        if arity == 3:
            axis_count = block.get("axis_count", 1)
            detail = f"軸{axis_count}頭ながし"
            if axis_count == 1 and ordered:
                detail += f"（{block.get('axis_position', '1着')}固定）"
        elif ordered:
            detail = f"{block.get('axis_position', '1着')}ながし"

    points = block.get("points", 0)
    total = block_total_amount(block)
    amount_desc = (
        f"1点あたり{block.get('amount_value', 0)}円"
        if block.get("amount_mode") == "per_point"
        else f"合計予算{block.get('amount_value', 0)}円（均等割り）"
    )
    detail_part = f"（{detail}）" if detail else ""
    return f"{ticket_type} / {method}{detail_part} / 目標{points}点 / {amount_desc}（見積合計{total}円）"

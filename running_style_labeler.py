<<<<<<< ours
import pandas as pd
from running_style_labeler import label_styles_from_past5_results

results_df = pd.DataFrame([
    ["111", "サンプルA", "2026-03-01", "4-3-2-1", 16],
    ["111", "サンプルA", "2026-02-10", "8-8-5-3", 16],
    ["111", "サンプルA", "2026-01-20", "7-7-6-4", 16],
    ["111", "サンプルA", "2025-12-28", "6-5-4-3", 16],
    ["111", "サンプルA", "2025-11-30", "9-9-7-5", 18],
], columns=["horse_id", "horse_name", "日付", "通過", "頭数"])

labeled_df = label_styles_from_past5_results(results_df)
print(labeled_df[["horse_id", "horse_name", "脚質", "使用通過順"]])

"""過去5走のコーナー通過順から脚質を自動ラベルする。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Sequence
import re

import pandas as pd

_CORNER_PATTERN = re.compile(r"^\d+(?:-\d+){3}$")


@dataclass(frozen=True)
class RaceCorner:
    c1: int
    c2: int
    c3: int
    c4: int
    field_size: int


@dataclass(frozen=True)
class StyleResult:
    label: str
    avg_r1: float
    avg_r4: float
    avg_gain: float
    detail: str


def _parse_corner_order(raw: str) -> tuple[int, int, int, int]:
    value = str(raw).strip()
    if not _CORNER_PATTERN.match(value):
        raise ValueError(f"コーナー通過順の形式が不正です: {raw}")

    parts = [int(x) for x in value.split("-")]
    return tuple(parts)  # type: ignore[return-value]


def _infer_field_size(corners: tuple[int, int, int, int]) -> int:
    inferred = max(corners) + 2
    return max(8, min(inferred, 18))


def classify_running_style(
    corner_orders: Sequence[str],
    field_sizes: Sequence[int] | None = None,
) -> StyleResult:
    """過去5レースの通過順から脚質を1つ返す。"""

    if len(corner_orders) != 5:
        raise ValueError(f"5レース分を入力してください。入力件数: {len(corner_orders)}")

    sizes = list(field_sizes) if field_sizes is not None else [None] * len(corner_orders)
    if len(sizes) != len(corner_orders):
        raise ValueError("field_sizes の件数は corner_orders と同じにしてください。")

    races: list[RaceCorner] = []
    for raw, n in zip(corner_orders, sizes):
        c1, c2, c3, c4 = _parse_corner_order(raw)
        field_size = int(n) if n is not None else _infer_field_size((c1, c2, c3, c4))
        races.append(RaceCorner(c1, c2, c3, c4, field_size))

    r1 = [race.c1 / race.field_size for race in races]
    r4 = [race.c4 / race.field_size for race in races]
    gains = [(race.c1 - race.c4) / race.field_size for race in races]

    avg_r1 = mean(r1)
    avg_r4 = mean(r4)
    avg_gain = mean(gains)

    if avg_r1 <= 0.18 and avg_r4 <= 0.20 and avg_gain < 0.08:
        label = "逃げ"
        detail = "序盤・終盤とも前方固定"
    elif avg_r1 <= 0.35 and avg_r4 <= 0.38:
        label = "先行"
        detail = "道中は前目で運び、終盤も前残り"
    elif avg_r1 > 0.35 and avg_r4 <= 0.55 and avg_gain >= 0.08:
        label = "差し"
        detail = "道中は中団〜後方、4角までに押し上げ"
    else:
        label = "追込"
        detail = "道中後方待機が中心で、直線勝負型"

    return StyleResult(label=label, avg_r1=avg_r1, avg_r4=avg_r4, avg_gain=avg_gain, detail=detail)


def _collect_last5_corner_orders(
    horse_results: pd.DataFrame,
    corner_col: str = "通過",
    field_size_col: str = "頭数",
) -> tuple[list[str], list[int]]:
    """1頭分の戦績DataFrameから有効なコーナー通過順を最大5件抽出する。"""

    corner_orders: list[str] = []
    field_sizes: list[int] = []

    for _, row in horse_results.iterrows():
        corner_raw = str(row.get(corner_col, "")).strip()
        if not _CORNER_PATTERN.match(corner_raw):
            continue

        corners = _parse_corner_order(corner_raw)
        field_raw = row.get(field_size_col)
        if pd.isna(field_raw):
            n = _infer_field_size(corners)
        else:
            try:
                n = int(field_raw)
            except (TypeError, ValueError):
                n = _infer_field_size(corners)

        corner_orders.append(corner_raw)
        field_sizes.append(max(8, min(n, 18)))

        if len(corner_orders) == 5:
            break

    return corner_orders, field_sizes


def label_styles_from_past5_results(
    results_df: pd.DataFrame,
    horse_id_col: str = "horse_id",
    horse_name_col: str = "horse_name",
    date_col: str = "日付",
    corner_col: str = "通過",
    field_size_col: str = "頭数",
) -> pd.DataFrame:
    """全馬の戦績DataFrameから脚質を割り振る。"""

    required_cols = {horse_id_col, horse_name_col, date_col, corner_col}
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        raise ValueError(f"必要な列が不足しています: {missing}")

    working = results_df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.sort_values([horse_id_col, date_col], ascending=[True, False])

    rows: list[dict] = []
    for horse_id, horse_df in working.groupby(horse_id_col, sort=False):
        horse_name = str(horse_df.iloc[0][horse_name_col])
        corner_orders, field_sizes = _collect_last5_corner_orders(
            horse_df,
            corner_col=corner_col,
            field_size_col=field_size_col,
        )

        if len(corner_orders) < 5:
            rows.append(
                {
                    horse_id_col: horse_id,
                    horse_name_col: horse_name,
                    "脚質": "判定不可",
                    "理由": f"有効な通過順が{len(corner_orders)}件（5件必要）",
                    "使用通過順": corner_orders,
                }
            )
            continue

        result = classify_running_style(corner_orders, field_sizes)
        rows.append(
            {
                horse_id_col: horse_id,
                horse_name_col: horse_name,
                "脚質": result.label,
                "理由": result.detail,
                "使用通過順": corner_orders,
                **asdict(result),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        [
            ["111", "サンプルA", "2026-03-01", "4-3-2-1", 16],
            ["111", "サンプルA", "2026-02-10", "8-8-5-3", 16],
            ["111", "サンプルA", "2026-01-20", "7-7-6-4", 16],
            ["111", "サンプルA", "2025-12-28", "6-5-4-3", 16],
            ["111", "サンプルA", "2025-11-30", "9-9-7-5", 18],
            ["222", "サンプルB", "2026-03-02", "1-1-1-1", 16],
            ["222", "サンプルB", "2026-02-11", "2-2-2-2", 16],
            ["222", "サンプルB", "2026-01-21", "2-1-1-1", 16],
            ["222", "サンプルB", "2025-12-27", "1-1-2-2", 16],
            ["222", "サンプルB", "2025-11-28", "1-1-1-2", 16],
        ],
        columns=["horse_id", "horse_name", "日付", "通過", "頭数"],
    )

    labeled = label_styles_from_past5_results(sample_df)
    print(labeled[["horse_id", "horse_name", "脚質", "使用通過順"]])
=======
"""過去5走のコーナー通過順から脚質を自動ラベルする。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Sequence
import re

import pandas as pd

_CORNER_PATTERN = re.compile(r"^\d+(?:-\d+){3}$")


@dataclass(frozen=True)
class RaceCorner:
    c1: int
    c2: int
    c3: int
    c4: int
    field_size: int


@dataclass(frozen=True)
class StyleResult:
    label: str
    avg_r1: float
    avg_r4: float
    avg_gain: float
    detail: str


def _parse_corner_order(raw: str) -> tuple[int, int, int, int]:
    value = str(raw).strip()
    if not _CORNER_PATTERN.match(value):
        raise ValueError(f"コーナー通過順の形式が不正です: {raw}")

    parts = [int(x) for x in value.split("-")]
    return tuple(parts)  # type: ignore[return-value]


def _infer_field_size(corners: tuple[int, int, int, int]) -> int:
    inferred = max(corners) + 2
    return max(8, min(inferred, 18))


def classify_running_style(
    corner_orders: Sequence[str],
    field_sizes: Sequence[int] | None = None,
) -> StyleResult:
    """過去5レースの通過順から脚質を1つ返す。"""

    if len(corner_orders) != 5:
        raise ValueError(f"5レース分を入力してください。入力件数: {len(corner_orders)}")

    sizes = list(field_sizes) if field_sizes is not None else [None] * len(corner_orders)
    if len(sizes) != len(corner_orders):
        raise ValueError("field_sizes の件数は corner_orders と同じにしてください。")

    races: list[RaceCorner] = []
    for raw, n in zip(corner_orders, sizes):
        c1, c2, c3, c4 = _parse_corner_order(raw)
        field_size = int(n) if n is not None else _infer_field_size((c1, c2, c3, c4))
        races.append(RaceCorner(c1, c2, c3, c4, field_size))

    r1 = [race.c1 / race.field_size for race in races]
    r4 = [race.c4 / race.field_size for race in races]
    gains = [(race.c1 - race.c4) / race.field_size for race in races]

    avg_r1 = mean(r1)
    avg_r4 = mean(r4)
    avg_gain = mean(gains)

    if avg_r1 <= 0.18 and avg_r4 <= 0.20 and avg_gain < 0.08:
        label = "逃げ"
        detail = "序盤・終盤とも前方固定"
    elif avg_r1 <= 0.35 and avg_r4 <= 0.38:
        label = "先行"
        detail = "道中は前目で運び、終盤も前残り"
    elif avg_r1 > 0.35 and avg_r4 <= 0.55 and avg_gain >= 0.08:
        label = "差し"
        detail = "道中は中団〜後方、4角までに押し上げ"
    else:
        label = "追込"
        detail = "道中後方待機が中心で、直線勝負型"

    return StyleResult(label=label, avg_r1=avg_r1, avg_r4=avg_r4, avg_gain=avg_gain, detail=detail)


def _collect_last5_corner_orders(
    horse_results: pd.DataFrame,
    corner_col: str = "通過",
    field_size_col: str = "頭数",
) -> tuple[list[str], list[int]]:
    """1頭分の戦績DataFrameから有効なコーナー通過順を最大5件抽出する。"""

    corner_orders: list[str] = []
    field_sizes: list[int] = []

    for _, row in horse_results.iterrows():
        corner_raw = str(row.get(corner_col, "")).strip()
        if not _CORNER_PATTERN.match(corner_raw):
            continue

        corners = _parse_corner_order(corner_raw)
        field_raw = row.get(field_size_col)
        if pd.isna(field_raw):
            n = _infer_field_size(corners)
        else:
            try:
                n = int(field_raw)
            except (TypeError, ValueError):
                n = _infer_field_size(corners)

        corner_orders.append(corner_raw)
        field_sizes.append(max(8, min(n, 18)))

        if len(corner_orders) == 5:
            break

    return corner_orders, field_sizes


def label_styles_from_past5_results(
    results_df: pd.DataFrame,
    horse_id_col: str = "horse_id",
    horse_name_col: str = "horse_name",
    date_col: str = "日付",
    corner_col: str = "通過",
    field_size_col: str = "頭数",
) -> pd.DataFrame:
    """全馬の戦績DataFrameから脚質を割り振る。"""

    required_cols = {horse_id_col, horse_name_col, date_col, corner_col}
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        raise ValueError(f"必要な列が不足しています: {missing}")

    working = results_df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.sort_values([horse_id_col, date_col], ascending=[True, False])

    rows: list[dict] = []
    for horse_id, horse_df in working.groupby(horse_id_col, sort=False):
        horse_name = str(horse_df.iloc[0][horse_name_col])
        corner_orders, field_sizes = _collect_last5_corner_orders(
            horse_df,
            corner_col=corner_col,
            field_size_col=field_size_col,
        )

        if len(corner_orders) < 5:
            rows.append(
                {
                    horse_id_col: horse_id,
                    horse_name_col: horse_name,
                    "脚質": "判定不可",
                    "理由": f"有効な通過順が{len(corner_orders)}件（5件必要）",
                    "使用通過順": corner_orders,
                }
            )
            continue

        result = classify_running_style(corner_orders, field_sizes)
        rows.append(
            {
                horse_id_col: horse_id,
                horse_name_col: horse_name,
                "脚質": result.label,
                "理由": result.detail,
                "使用通過順": corner_orders,
                **asdict(result),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        [
            ["111", "サンプルA", "2026-03-01", "4-3-2-1", 16],
            ["111", "サンプルA", "2026-02-10", "8-8-5-3", 16],
            ["111", "サンプルA", "2026-01-20", "7-7-6-4", 16],
            ["111", "サンプルA", "2025-12-28", "6-5-4-3", 16],
            ["111", "サンプルA", "2025-11-30", "9-9-7-5", 18],
            ["222", "サンプルB", "2026-03-02", "1-1-1-1", 16],
            ["222", "サンプルB", "2026-02-11", "2-2-2-2", 16],
            ["222", "サンプルB", "2026-01-21", "2-1-1-1", 16],
            ["222", "サンプルB", "2025-12-27", "1-1-2-2", 16],
            ["222", "サンプルB", "2025-11-28", "1-1-1-2", 16],
        ],
        columns=["horse_id", "horse_name", "日付", "通過", "頭数"],
    )

    labeled = label_styles_from_past5_results(sample_df)
    print(labeled[["horse_id", "horse_name", "脚質", "使用通過順"]])
>>>>>>> theirs

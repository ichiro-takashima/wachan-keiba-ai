# netkeiba-scraper

1. netkeibaから各馬の血統と戦績をスクレイプするコードを作成します。←完了
2. 精度検証できる仕組みを作成します。

## 脚質自動ラベル（過去5走）
`running_style_labeler.py` に、各馬の過去5戦の「通過」列から脚質（`逃げ/先行/差し/追込`）を割り振るプログラムを追加しています。

### 入力データ形式（DataFrame）
最低限、次の列が必要です。
- `horse_id`
- `horse_name`
- `日付`
- `通過`（例: `4-3-2-1`）

任意で `頭数` 列があると判定が安定します（無い場合は推定）。

### 使い方（全馬まとめて判定）
```python
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
```

### 判定ロジック
1. 馬ごとに日付降順で戦績を並べる
2. `通過` 列から `N-N-N-N` 形式だけ抽出
3. 先頭から5件を使用
4. `avg_r1` / `avg_r4` / `avg_gain` を計算し、しきい値で脚質を決定

有効な通過順が5件未満の馬は `判定不可` を返します。

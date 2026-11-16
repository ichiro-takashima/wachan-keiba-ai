"""種牡馬ごとに「どんなレースで産駒が走るか」を集計する(父・母父の得意条件と苦手条件)。

相性モデルの学習時に保存される data/models/affinity/pedigree_runs.csv(血統が分かる全走)を使う。
条件(芝ダ・芝ダ×距離・芝ダ×馬場・芝ダ×競馬場)ごとに、産駒の3着内率を「同じ条件を走った全馬」と比べた全体比と、
それをさらに「その種牡馬の普段の水準」で割った自身比(=適性)、偶然ではなさそうか(z値: 二項分布での偏り)を出す。
強い種牡馬はどの条件でも全体比が高くなるので、得意・苦手の判定は自身比で行う。

使い方:
    python -m model.sire_aptitude                # 父で集計 → CSV保存 + 出走数上位の要約を表示
    python -m model.sire_aptitude --by bms       # 母父で集計
    python -m model.sire_aptitude --name キズナ   # 指定した種牡馬の全条件を表示
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PEDIGREE_RUNS_PATH = Path("data/models/affinity/pedigree_runs.csv")
OUT_DIR = Path("data/models/affinity")

SURFACES = {0: "芝", 1: "ダート"}
PLACES = {1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京", 6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"}
DIST_BINS = [0, 1200, 1500, 1800, 2200, 10000]
DIST_LABELS = ["1200m以下", "1300-1500m", "1600-1800m", "1900-2200m", "2300m以上"]

MIN_RUNS = 60  # 条件ごとの最低出走数(これ未満の条件は判定しない)
STRONG_LIFT, STRONG_Z = 1.15, 2.0  # 得意: 自身比1.15以上 かつ z値2以上
WEAK_LIFT, WEAK_Z = 0.85, -2.0  # 苦手: 自身比0.85以下 かつ z値-2以下


def load_runs():
    runs = pd.read_csv(PEDIGREE_RUNS_PATH, encoding="utf-8-sig", dtype={"sire": str, "bms": str})
    runs = runs[(runs["n"] > 0) & runs["_surf"].isin(SURFACES)].copy()
    surf = runs["_surf"].map(SURFACES)
    dist = pd.cut(runs["_dist"], DIST_BINS, labels=DIST_LABELS).astype(str)
    going = runs["_wet"].map({0.0: "良", 1.0: "道悪(稍重〜不良)"})
    place = runs["_place"].map(PLACES)
    runs["芝ダ"] = surf
    runs["距離"] = surf + " " + dist
    runs["馬場"] = (surf + " " + going).where(going.notna())
    runs["競馬場"] = (place + " " + surf).where(place.notna())
    return runs


def aptitude_table(runs, by="sire"):
    """種牡馬×条件ごとに、産駒の3着内率を (1) 同条件の全馬 (2) その種牡馬自身の普段の水準 と比べる。

    全体比: 産駒の3着内率 ÷ 同条件の全馬の3着内率(強い種牡馬はどの条件でも1を超える)
    自身比: 全体比 ÷ その種牡馬の普段の全体比。芝ダは全条件の、距離・馬場・競馬場は同じ芝ダの水準と比べる。
            1を超えるほど「この種牡馬にしては特にこの条件で走る」= 適性。判定(得意・苦手)はこちらで行う。
    """
    r = runs[runs[by].notna()].copy()
    rows = []
    for kind in ["芝ダ", "距離", "馬場", "競馬場"]:
        k = r[r[kind].notna()].copy()
        base = k.groupby(kind)[["n", "win", "top3"]].sum()
        k["exp"] = k[kind].map(base["top3"] / base["n"]) * k["n"]  # 全馬並みなら期待される3着内数
        k["exp_win"] = k[kind].map(base["win"] / base["n"]) * k["n"]
        level_keys = [by] if kind == "芝ダ" else [by, "芝ダ"]
        level = k.groupby(level_keys)[["top3", "exp"]].sum()
        k["level"] = k.set_index(level_keys).index.map(level["top3"] / level["exp"]).to_numpy()
        k["exp_self"] = k["exp"] * k["level"]  # この種牡馬の普段の水準なら期待される3着内数
        cell = k.groupby([by, kind]).agg(出走数=("n", "sum"), win=("win", "sum"), top3=("top3", "sum"),
                                          exp=("exp", "sum"), exp_win=("exp_win", "sum"),
                                          exp_self=("exp_self", "sum")).reset_index()
        cell["種類"] = kind
        cell["3着内率"] = cell["top3"] / cell["出走数"]
        cell["勝率"] = cell["win"] / cell["出走数"]
        cell["全体比"] = cell["top3"] / cell["exp"]
        cell["勝率の全体比"] = cell["win"] / cell["exp_win"]
        cell["自身比"] = cell["top3"] / cell["exp_self"]
        p = (cell["exp_self"] / cell["出走数"]).clip(1e-6, 1 - 1e-6)
        cell["z値"] = (cell["top3"] - cell["exp_self"]) / np.sqrt(cell["出走数"] * p * (1 - p))
        rows.append(cell.rename(columns={by: "種牡馬", kind: "条件"}))
    out = pd.concat(rows, ignore_index=True)
    total = r.groupby(by)[["n", "top3"]].sum()
    out = out.join(total["n"].rename("産駒の総出走数"), on="種牡馬")
    out = out.join((total["top3"] / total["n"]).rename("産駒全体の3着内率"), on="種牡馬")
    out["判定"] = ""
    enough = out["出走数"] >= MIN_RUNS
    out.loc[enough & (out["自身比"] >= STRONG_LIFT) & (out["z値"] >= STRONG_Z), "判定"] = "得意"
    out.loc[enough & (out["自身比"] <= WEAK_LIFT) & (out["z値"] <= WEAK_Z), "判定"] = "苦手"
    cols = ["種牡馬", "産駒の総出走数", "産駒全体の3着内率", "種類", "条件", "出走数", "3着内率", "勝率",
            "全体比", "勝率の全体比", "自身比", "z値", "判定"]
    return out[cols].sort_values(["産駒の総出走数", "種牡馬", "種類", "z値"], ascending=[False, True, True, False])


# ---------------------------------------------------------------- 予想用(その日より前の産駒成績だけで判定)

# 種類 → (条件のキー, 比べる元のキー)。aptitude_table と同じ区切り(距離は DIST_BINS、芝ダは障害を除く)
ASOF_KINDS = {
    "芝ダ": (["_asurf"], []),
    "距離": (["_asurf", "_dband"], ["_asurf"]),
    "馬場": (["_asurf", "_wet"], ["_asurf"]),
    "競馬場": (["_asurf", "_place"], ["_asurf"]),
}
ASOF_BY = {"父": "sire", "母父": "bms"}
BASE_PRIOR = 0.215  # 条件ごとの全馬の3着内率の寄せ先
BASE_SHRINK = 100.0  # 条件ごとの全馬の3着内率を、この件数ぶん BASE_PRIOR に寄せる
LEVEL_SHRINK = 10.0  # 種牡馬の普段の水準(全体比)を、期待3着内数でこの件数ぶん 1.0 に寄せる
Z_CLIP = 5.0
_EXP_COLS = [f"_exp_{kind}" for kind in ASOF_KINDS]


def _with_condition_keys(runs):
    """条件の区切り(_asurf: 芝・ダートだけの芝ダ、_dband: DIST_BINS の距離帯)を付けた写し。"""
    r = runs.copy()
    r["_asurf"] = r["_surf"].where(r["_surf"].isin(SURFACES))
    r["_dband"] = pd.cut(r["_dist"], DIST_BINS, labels=False).astype(float)
    return r


def add_expected(runs):
    """各走に、その走の日より前の「条件ごとの全馬の3着内率」から見た期待3着内数(_exp_<種類>)を付ける。

    過去の走の期待値は後から変わらないので、予想用の過去データには一度付けておけば使い回せる。
    runs は affinity_model の1走=1行の表(_date, sire, bms, _surf, _dist, _wet, _place, n, top3 が必要)。
    """
    from model.affinity_model import _prior_sums  # 循環 import を避ける

    r = _with_condition_keys(runs)
    for kind, (cond, _) in ASOF_KINDS.items():
        s = _prior_sums(r, cond, ["n", "top3"])
        r[f"_exp_{kind}"] = r["n"] * ((s["top3"] + BASE_PRIOR * BASE_SHRINK) / (s["n"] + BASE_SHRINK))
    return r


def _sums_before(hist, tgt, keys, values):
    """tgt の各行と同じキーの、hist(すべて tgt より前の日付)の行の values の合計。
    _prior_sums と同じく、キーが欠けている行は NaN、該当する走がなければ 0。"""
    total = hist.dropna(subset=keys).groupby(keys)[values].sum()
    out = tgt[keys].merge(total, left_on=keys, right_index=True, how="left")[values].fillna(0.0)
    out.index = tgt.index
    out.loc[tgt[keys].isna().any(axis=1)] = np.nan
    return out


def asof_aptitude(runs, targets=None):
    """各走について、その日より前の産駒成績だけで aptitude_table と同じ自身比・z値・判定を出す。

    runs は affinity_model の1走=1行の表(_date, sire, bms, _surf, _dist, _wet, _place, n, top3 が必要)。
    今回の出走(n=0)の行も入れておけば、その行に「今回の条件での適性」が付く。
    targets(行のラベル)を渡すとその行だけを返す。予想時のように targets がほかのどの行よりも後の日付なら、
    日付ごとの累積を取らずに単純な合計で済むので速い。
    列: {父,母父}_{芝ダ,距離,馬場,競馬場}_{出走数,自身比,z値,判定}  判定は 得意=1 / 苦手=-1 / それ以外=0
    """
    from model.affinity_model import _prior_sums  # 循環 import を避ける

    r = _with_condition_keys(runs) if set(_EXP_COLS) <= set(runs.columns) else add_expected(runs)
    if targets is None:
        index = r.index

        def sums(keys, values):
            return _prior_sums(r, keys, values)
    else:
        index = pd.Index(targets)
        r = r.assign(**{c: r[c].fillna(0.0) for c in _EXP_COLS})
        rest = r.drop(index=index)
        if len(rest) and rest["_date"].max() >= r.loc[index, "_date"].min():
            raise ValueError("targets の日付は、ほかのどの行よりも後でなければならない")

        tgt = r.loc[index]
        # 今回の出走馬の父・母父の産駒の走だけに絞ってから合計する(全種牡馬を集計しない)
        hist = {by: rest[rest[by].isin(tgt[by].dropna())] for by in ASOF_BY.values()}

        def sums(keys, values):
            return _sums_before(hist[keys[0]], tgt, keys, values)

    out = pd.DataFrame(index=index)
    for kind, (cond, parent) in ASOF_KINDS.items():
        exp = f"_exp_{kind}"
        for label, by in ASOF_BY.items():
            cell = sums([by] + cond, ["n", "top3", exp])
            lv = sums([by] + parent, ["top3", exp])
            level = (lv["top3"] + LEVEL_SHRINK) / (lv[exp] + LEVEL_SHRINK)
            exp_self = cell[exp] * level
            n = cell["n"].where(cell["n"] > 0)
            p = (exp_self / n).clip(1e-6, 1 - 1e-6)
            ratio = cell["top3"] / exp_self.where(exp_self > 0)
            z = ((cell["top3"] - exp_self) / np.sqrt(n * p * (1 - p))).clip(-Z_CLIP, Z_CLIP)
            enough = cell["n"] >= MIN_RUNS
            judge = (enough & (ratio >= STRONG_LIFT) & (z >= STRONG_Z)).astype(float)                 - (enough & (ratio <= WEAK_LIFT) & (z <= WEAK_Z)).astype(float)
            pre = f"{label}_{kind}"
            out[f"{pre}_出走数"] = cell["n"]
            out[f"{pre}_自身比"] = ratio
            out[f"{pre}_z値"] = z
            out[f"{pre}_判定"] = judge.where(cell["n"].notna())
    return out


def condition_labels(runs):
    """各走の条件名(aptitude_table の「条件」列と同じ書き方)。{種類: Series}"""
    surf = runs["_surf"].map(SURFACES)
    dist = pd.cut(runs["_dist"], DIST_BINS, labels=DIST_LABELS).astype(object)
    going = runs["_wet"].map({0.0: "良", 1.0: "道悪(稍重〜不良)"})
    place = runs["_place"].map(PLACES)
    return {"芝ダ": surf, "距離": surf + " " + dist, "馬場": surf + " " + going, "競馬場": place + " " + surf}


def aptitude_notes(apt, runs):
    """今回の条件で得意・苦手と判定された父・母父の条件を、行ごとに文字列のリストにする。
    例: ["父 得意: 芝 2300m以上(1.43)", "母父 苦手: 福島 ダート(0.70)"]"""
    labels = condition_labels(runs.loc[apt.index])
    notes = []
    for i in apt.index:
        row = []
        for who in ASOF_BY:
            for kind in ASOF_KINDS:
                j = apt.at[i, f"{who}_{kind}_判定"]
                cond = labels[kind].get(i)
                if j in (1, -1) and isinstance(cond, str):
                    row.append(f"{who} {'得意' if j > 0 else '苦手'}: {cond}({apt.at[i, f'{who}_{kind}_自身比']:.2f})")
        notes.append(row)
    return notes


def summarize(table, top_n=3):
    """種牡馬ごとに、得意・苦手な条件を z値の大きい順に並べた1行の要約。"""
    lines = []
    for sire, g in table.groupby("種牡馬", sort=False):
        good = g[g["判定"] == "得意"].sort_values("z値", ascending=False).head(top_n)
        bad = g[g["判定"] == "苦手"].sort_values("z値").head(top_n)
        surf = g[g["種類"] == "芝ダ"].set_index("条件")["出走数"]
        dist = g[(g["種類"] == "距離") & (g["出走数"] >= MIN_RUNS)].sort_values("全体比", ascending=False).head(2)
        lines.append({
            "種牡馬": sire,
            "産駒の総出走数": int(g["産駒の総出走数"].iloc[0]),
            "産駒全体の3着内率": round(float(g["産駒全体の3着内率"].iloc[0]), 3),
            "ダート出走の割合": round(float(surf.get("ダート", 0) / surf.sum()), 2) if surf.sum() else None,
            "全馬と比べて走る距離": " / ".join(f"{r.条件}({r.全体比:.2f}倍)" for r in dist.itertuples()),
            "得意": " / ".join(f"{r.条件}({r.自身比:.2f})" for r in good.itertuples()),
            "苦手": " / ".join(f"{r.条件}({r.自身比:.2f})" for r in bad.itertuples()),
        })
    return pd.DataFrame(lines)


def main():
    parser = argparse.ArgumentParser(description="種牡馬ごとの得意条件・苦手条件を集計する")
    parser.add_argument("--by", choices=["sire", "bms"], default="sire", help="sire=父 / bms=母父")
    parser.add_argument("--top", type=int, default=40, help="要約を表示する種牡馬の数(産駒の出走数順)")
    parser.add_argument("--name", help="この種牡馬の全条件を表示する")
    args = parser.parse_args()

    table = aptitude_table(load_runs(), by=args.by)
    summary = summarize(table)
    label = "父" if args.by == "sire" else "母父"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT_DIR / f"sire_aptitude_{args.by}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / f"sire_aptitude_{args.by}_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 500)
    if args.name:
        g = table[table["種牡馬"].str.contains(args.name, regex=False)]
        print(g[g["出走数"] >= 20].drop(columns=["産駒の総出走数", "産駒全体の3着内率"]).round(3).to_string(index=False))
        return
    print(f"{label}ごとの得意条件・苦手条件(条件ごとに{MIN_RUNS}走以上。数値は自身比=その種牡馬の普段の水準と比べた3着内率の比)")
    print(summary.head(args.top).to_string(index=False))
    print(f"保存: {OUT_DIR / f'sire_aptitude_{args.by}.csv'} / {OUT_DIR / f'sire_aptitude_{args.by}_summary.csv'}")


if __name__ == "__main__":
    main()

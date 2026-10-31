"""競馬ラウンジ×サイバーパンク風の演出UIコンポーネント集(Streamlit用・外部CSSファイル不要)。

    streamlit run dopamine_ui.py     # デモ画面

app.py からは関数を import して使える:
    inject_theme()                         # 画面全体を漆黒テーマにする(最初に1回)
    odds_card(...)                         # ① オッズ・期待値カード(シマー+カウントアップ)
    horse_status_card(...)                 # ② 出走馬ステータスカード(レーダーチャート+ネオン評価)
    celebrate(payout)                      # ③ 全画面の金色紙吹雪+桜吹雪(万馬券演出)

JavaScriptが必要な部品は st.components.v1.html(iframe)で描画する。
st.markdown は <script> を実行しないため。
"""
import html
import math
import random
import time

import streamlit as st
import streamlit.components.v1 as components

BASE = "#0B0F19"
GOLD = "#FFD700"
EMERALD = "#00FF87"
RED = "#FF3366"
CYAN = "#22D3EE"

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900'
    '&family=Noto+Sans+JP:wght@400;700;900&display=swap" rel="stylesheet">'
)

# iframe内で数字を 0 → 目標値 へ高速カウントアップする(easeOutExpo)
COUNT_UP_JS = """
<script>
function countUp(el) {
  const target = parseFloat(el.dataset.target);
  const decimals = parseInt(el.dataset.decimals || "0");
  const duration = parseInt(el.dataset.duration || "1600");
  const delay = parseInt(el.dataset.delay || "0");
  const fmt = (v) => v.toLocaleString("ja-JP", {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
  setTimeout(() => {
    const start = performance.now();
    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
      el.textContent = fmt(target * eased);
      if (p < 1) requestAnimationFrame(tick);
      else el.classList.add("landed");
    }
    requestAnimationFrame(tick);
  }, delay);
}
document.querySelectorAll("[data-target]").forEach(countUp);
document.querySelectorAll("[data-width]").forEach((el) => {
  setTimeout(() => { el.style.width = el.dataset.width + "%"; }, 150);
});
</script>
"""

FRAME_BASE_CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: transparent; font-family: 'Noto Sans JP', sans-serif; color: #E5E7EB; overflow: hidden; }}
.num-font {{ font-family: 'Orbitron', sans-serif; }}
"""


# ---------------------------------------------------------------- 全体テーマ

def inject_theme():
    """アプリ全体を漆黒×ネオンのテーマにする。st.markdown で1回だけ呼ぶ。"""
    st.markdown(FONTS + f"""
<style>
.stApp {{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(255,215,0,0.10), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(0,255,135,0.08), transparent 60%),
    radial-gradient(800px 600px at 50% 120%, rgba(255,51,102,0.08), transparent 60%),
    {BASE};
  color: #E5E7EB;
}}
.stApp::before {{  /* うっすら走査線 */
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background: repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 3px);
}}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: #070A12; border-right: 1px solid rgba(255,215,0,0.15); }}
h1, h2, h3 {{ font-family: 'Orbitron', 'Noto Sans JP', sans-serif !important; letter-spacing: .04em; }}
.stApp p, .stApp label, .stApp span {{ font-family: 'Noto Sans JP', sans-serif; }}
.stButton > button {{
  background: linear-gradient(135deg, {GOLD}, #FFB300 50%, {GOLD});
  background-size: 200% 200%; animation: btnShift 3s ease infinite;
  color: #111 !important; font-weight: 900; border: none; border-radius: 12px;
  box-shadow: 0 0 18px rgba(255,215,0,.45); transition: transform .15s, box-shadow .15s;
}}
.stButton > button:hover {{ transform: translateY(-2px) scale(1.02); box-shadow: 0 0 32px rgba(255,215,0,.8); }}
.stButton > button p {{ color: #111 !important; font-weight: 900; }}
@keyframes btnShift {{ 0%,100% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} }}
.neon-title {{
  font-family: 'Orbitron', 'Noto Sans JP', sans-serif; font-weight: 900; font-size: clamp(1.6rem, 4vw, 2.8rem);
  background: linear-gradient(90deg, {GOLD}, #FFF7AE, {GOLD}, {EMERALD});
  background-size: 300% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: titleFlow 6s linear infinite; filter: drop-shadow(0 0 12px rgba(255,215,0,.35));
}}
@keyframes titleFlow {{ to {{ background-position: 300% 0; }} }}
.section-label {{
  display: inline-block; margin: 1.4rem 0 .6rem; padding: .2rem .9rem; border-radius: 999px;
  font-weight: 900; letter-spacing: .12em; font-size: .85rem; color: {EMERALD};
  border: 1px solid rgba(0,255,135,.5); box-shadow: 0 0 14px rgba(0,255,135,.25) inset, 0 0 10px rgba(0,255,135,.2);
}}
</style>
""", unsafe_allow_html=True)


def neon_title(text, sub=None):
    sub_html = f'<div style="color:#9CA3AF;margin-top:.2rem">{html.escape(sub)}</div>' if sub else ""
    st.markdown(f'<div class="neon-title">{html.escape(text)}</div>{sub_html}', unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<span class="section-label">{html.escape(text)}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------- ① オッズ・期待値カード

def _tier(odds, ev):
    """カードの色: 大穴(高オッズ)=レッド / 期待値1超=エメラルド / それ以外(本命筋)=ゴールド。"""
    if odds >= 20:
        return "大穴", RED
    if ev >= 1.0:
        return "妙味", EMERALD
    return "本命", GOLD


def odds_card(mark, number, name, jockey, odds, top3_prob, ev, label=None, height=340):
    """単勝オッズ・期待値・3着内確率を、シマー背景＋カウントアップで見せるカード。

    ev は 推定勝率×単勝オッズ(1.0 で収支トントン)。top3_prob は 0〜1。
    """
    tier_name, color = _tier(odds, ev)
    label = label or tier_name
    ev_pct = ev * 100
    hot = "hot" if ev >= 1.0 or odds >= 20 else ""
    body = f"""
{FONTS}
<style>
{FRAME_BASE_CSS}
.card {{
  position: relative; height: {height - 12}px; margin: 4px; padding: 18px 18px 16px; border-radius: 18px;
  background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01)) , #0F1523;
  border: 1px solid {color}55; overflow: hidden;
  box-shadow: 0 0 0 1px {color}22 inset, 0 10px 40px -10px {color}55;
}}
.card::before {{  /* 外周を流れる光の縁取り */
  content: ""; position: absolute; inset: -2px; border-radius: 20px; padding: 2px;
  background: conic-gradient(from var(--a), transparent 0 70%, {color} 80%, #fff 82%, {color} 84%, transparent 95%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; animation: spin 4s linear infinite;
}}
@property --a {{ syntax: '<angle>'; initial-value: 0deg; inherits: false; }}
@keyframes spin {{ to {{ --a: 360deg; }} }}
.shimmer {{  /* 斜めに走る光 */
  position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(115deg, transparent 30%, {color}26 45%, rgba(255,255,255,.18) 50%, {color}26 55%, transparent 70%);
  background-size: 250% 100%; animation: sweep 3.2s ease-in-out infinite;
}}
@keyframes sweep {{ 0% {{ background-position: 150% 0; }} 100% {{ background-position: -100% 0; }} }}
.glow {{ position: absolute; width: 220px; height: 220px; right: -60px; top: -80px; border-radius: 50%;
  background: radial-gradient(circle, {color}40, transparent 70%); filter: blur(10px); animation: breathe 3s ease-in-out infinite; }}
@keyframes breathe {{ 50% {{ transform: scale(1.15); opacity: .7; }} }}
.top {{ display: flex; justify-content: space-between; align-items: center; position: relative; }}
.tag {{ font-weight: 900; font-size: 13px; letter-spacing: .1em; color: #0B0F19; background: {color};
  padding: 3px 10px; border-radius: 999px; box-shadow: 0 0 14px {color}; }}
.num {{ font-family: 'Orbitron'; font-weight: 900; font-size: 15px; color: {color};
  border: 1px solid {color}; border-radius: 8px; padding: 2px 9px; }}
.name {{ position: relative; margin-top: 12px; font-size: 22px; font-weight: 900; color: #fff;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.jockey {{ position: relative; font-size: 12px; color: #9CA3AF; }}
.odds {{ position: relative; margin-top: 10px; display: flex; align-items: baseline; gap: 6px; }}
.odds .v {{ font-family: 'Orbitron'; font-weight: 900; font-size: clamp(40px, 12vw, 58px); line-height: 1; color: {color};
  text-shadow: 0 0 18px {color}aa, 0 0 42px {color}55; }}
.odds .u {{ color: #D1D5DB; font-weight: 700; }}
.landed {{ animation: pop .45s ease-out; }}
@keyframes pop {{ 0% {{ transform: scale(1); }} 40% {{ transform: scale(1.12); }} 100% {{ transform: scale(1); }} }}
.stats {{ position: relative; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }}
.stat {{ background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 8px 10px; }}
.stat .k {{ font-size: 11px; color: #9CA3AF; letter-spacing: .05em; }}
.stat .val {{ font-family: 'Orbitron'; font-weight: 700; font-size: 22px; color: #fff; }}
.stat.ev .val {{ color: {EMERALD if ev >= 1 else '#fff'}; text-shadow: {'0 0 12px ' + EMERALD if ev >= 1 else 'none'}; }}
.bar {{ position: relative; height: 8px; margin-top: 14px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }}
.bar i {{ display: block; height: 100%; width: 0; border-radius: 99px; transition: width 1.6s cubic-bezier(.16,1,.3,1);
  background: linear-gradient(90deg, {color}, #fff); box-shadow: 0 0 12px {color}; }}
.bar-label {{ position: relative; display: flex; justify-content: space-between; font-size: 11px; color: #9CA3AF; margin-top: 4px; }}
.hot .tag {{ animation: blink 1.1s ease-in-out infinite; }}
@keyframes blink {{ 50% {{ box-shadow: 0 0 2px {color}; filter: brightness(1.4); }} }}
</style>
<div class="card {hot}">
  <div class="glow"></div><div class="shimmer"></div>
  <div class="top"><span class="tag">{html.escape(mark)} {html.escape(label)}</span><span class="num">{int(number)}番</span></div>
  <div class="name">{html.escape(name)}</div>
  <div class="jockey">騎手 {html.escape(jockey)}</div>
  <div class="odds"><span class="v num-font" data-target="{odds}" data-decimals="1">0.0</span><span class="u">倍</span></div>
  <div class="stats">
    <div class="stat ev"><div class="k">回収期待値</div><div class="val"><span data-target="{ev_pct:.1f}" data-decimals="0" data-delay="200">0</span>%</div></div>
    <div class="stat"><div class="k">期待値スコア</div><div class="val"><span data-target="{ev:.2f}" data-decimals="2" data-delay="200">0.00</span></div></div>
  </div>
  <div class="bar"><i data-width="{min(top3_prob * 100, 100):.1f}"></i></div>
  <div class="bar-label"><span>3着内確率</span><span class="num-font"><span data-target="{top3_prob * 100:.1f}" data-decimals="1" data-delay="300">0.0</span>%</span></div>
</div>
{COUNT_UP_JS}
"""
    components.html(body, height=height)


# ---------------------------------------------------------------- ② 出走馬ステータスカード

GRADE_COLORS = {"S": GOLD, "A": EMERALD, "B": CYAN, "C": "#A78BFA", "D": "#6B7280"}


def _radar_svg(stats, color, size=190):
    """stats: {軸名: 0〜100}。SVGのレーダーチャートを返す。"""
    labels = list(stats)
    n = len(labels)
    c, r = size / 2, size / 2 - 30

    def point(i, ratio):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return c + r * ratio * math.cos(ang), c + r * ratio * math.sin(ang)

    rings = "".join(
        f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, lv) for i in range(n)))}" '
        f'fill="none" stroke="rgba(255,255,255,{0.06 + lv * 0.06:.2f})" stroke-width="1"/>'
        for lv in (0.25, 0.5, 0.75, 1.0)
    )
    axes = "".join(
        f'<line x1="{c}" y1="{c}" x2="{point(i, 1)[0]:.1f}" y2="{point(i, 1)[1]:.1f}" stroke="rgba(255,255,255,.08)"/>'
        for i in range(n)
    )
    shape = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, max(0, min(v, 100)) / 100) for i, v in enumerate(stats.values())))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
        for x, y in (point(i, max(0, min(v, 100)) / 100) for i, v in enumerate(stats.values()))
    )
    texts = "".join(
        f'<text x="{point(i, 1.22)[0]:.1f}" y="{point(i, 1.22)[1] + 4:.1f}" text-anchor="middle" '
        f'font-size="10.5" fill="#9CA3AF">{html.escape(label)}</text>'
        for i, label in enumerate(labels)
    )
    return f"""
<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:{size}px;overflow:visible">
  {rings}{axes}
  <g class="radar-shape" style="transform-origin:{c}px {c}px">
    <polygon points="{shape}" fill="{color}33" stroke="{color}" stroke-width="2" style="filter:drop-shadow(0 0 6px {color})"/>
    {dots}
  </g>
  {texts}
</svg>"""


def horse_status_card(number, name, grade, stats, training_comment="", paddock_comment="", odds=None, height=470):
    """ホバーでネオンに光る出走馬カード。評価(S/A/B...)はネオンで明滅する。stats は {軸名: 0〜100}。"""
    color = GRADE_COLORS.get(grade, CYAN)
    odds_html = f'<span class="odds num-font">{odds:.1f}倍</span>' if odds is not None else ""
    bars = "".join(
        f'<div class="srow"><span>{html.escape(k)}</span><div class="sbar"><i data-width="{v}"></i></div>'
        f'<b class="num-font" data-target="{v}" data-decimals="0" data-duration="1200">0</b></div>'
        for k, v in stats.items()
    )
    body = f"""
{FONTS}
<style>
{FRAME_BASE_CSS}
.hcard {{
  position: relative; height: {height - 14}px; margin: 6px; padding: 16px; border-radius: 18px; cursor: default;
  background: linear-gradient(180deg, #111827, #0B0F19); border: 1px solid rgba(255,255,255,.08);
  transition: border-color .25s, box-shadow .25s, transform .25s; overflow: hidden;
}}
.hcard::after {{  /* グリッド地模様 */
  content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .35;
  background-image: linear-gradient(rgba(0,255,135,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,135,.06) 1px, transparent 1px);
  background-size: 18px 18px; mask-image: linear-gradient(180deg, #000, transparent 70%);
}}
.hcard:hover {{ border-color: {color}; transform: translateY(-3px);
  box-shadow: 0 0 0 1px {color} inset, 0 0 24px {color}88, 0 0 60px {color}33; }}
.hcard:hover .radar-shape polygon {{ fill: {color}55; }}
.head {{ display: flex; align-items: center; gap: 10px; position: relative; z-index: 1; }}
.no {{ font-family: 'Orbitron'; font-weight: 900; width: 34px; height: 34px; display: grid; place-items: center;
  border-radius: 10px; background: {color}; color: #0B0F19; box-shadow: 0 0 12px {color}; }}
.nm {{ flex: 1; font-weight: 900; font-size: 18px; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.odds {{ font-size: 12px; color: #D1D5DB; }}
.grade {{ position: absolute; right: 14px; top: 46px; z-index: 1; font-family: 'Orbitron'; font-weight: 900; font-size: 54px;
  color: {color}; animation: neon 1.6s ease-in-out infinite; }}
.grade small {{ display: block; font-size: 10px; letter-spacing: .2em; color: #9CA3AF; text-align: center; text-shadow: none; animation: none; }}
@keyframes neon {{
  0%, 100% {{ text-shadow: 0 0 4px #fff, 0 0 10px {color}, 0 0 22px {color}, 0 0 44px {color}; opacity: 1; }}
  45% {{ text-shadow: 0 0 2px #fff, 0 0 6px {color}, 0 0 12px {color}; opacity: .82; }}
  50% {{ opacity: .45; }} 55% {{ opacity: .95; }}
}}
.radar {{ position: relative; z-index: 1; display: flex; justify-content: center; margin-top: 4px; }}
.radar-shape {{ animation: grow 1.2s cubic-bezier(.16,1,.3,1) both; }}
@keyframes grow {{ from {{ transform: scale(0); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}
.srow {{ position: relative; z-index: 1; display: grid; grid-template-columns: 52px 1fr 28px; align-items: center; gap: 8px;
  font-size: 11px; color: #9CA3AF; margin-top: 5px; }}
.srow b {{ color: #fff; font-size: 12px; text-align: right; }}
.sbar {{ height: 6px; border-radius: 99px; background: rgba(255,255,255,.07); overflow: hidden; }}
.sbar i {{ display: block; height: 100%; width: 0; background: linear-gradient(90deg, {color}88, {color}); box-shadow: 0 0 8px {color};
  transition: width 1.4s cubic-bezier(.16,1,.3,1); }}
.cm {{ position: relative; z-index: 1; margin-top: 10px; font-size: 11.5px; line-height: 1.55; color: #D1D5DB;
  border-left: 2px solid {color}; padding-left: 8px; }}
.cm b {{ color: {color}; }}
</style>
<div class="hcard">
  <div class="head"><div class="no">{int(number)}</div><div class="nm">{html.escape(name)}</div>{odds_html}</div>
  <div class="grade">{html.escape(grade)}<small>期待度</small></div>
  <div class="radar">{_radar_svg(stats, color)}</div>
  {bars}
  <div class="cm"><b>調教</b> {html.escape(training_comment)}<br><b>パドック</b> {html.escape(paddock_comment)}</div>
</div>
{COUNT_UP_JS}
"""
    components.html(body, height=height)


# ---------------------------------------------------------------- ③ 全画面の祝祭演出

def celebrate(payout=None, title="万馬券的中!!", key=None):
    """画面全体に金色の紙吹雪と桜吹雪を降らせ、払戻額をカウントアップ表示する。

    iframe から親ページ(Streamlit本体)に canvas-confetti を読み込んで描画するので、
    コンポーネント自体は高さ0。呼ぶたびに1回だけ発火するよう、呼び出し側で1回だけ描画する。
    """
    nonce = key or f"{time.time():.6f}"
    payout_js = "null" if payout is None else str(int(payout))
    body = f"""
<script>
(function() {{
  const P = window.parent, doc = P.document;
  const GOLD = ["#FFD700", "#FFE873", "#FFB300", "#FFF7AE", "#ffffff"];
  const SAKURA = ["#FFC0CB", "#FFB7C5", "#FF8FAB", "#FFE4EC"];
  const nonce = "{nonce}";

  function overlay() {{
    const old = doc.getElementById("dopa-overlay"); if (old) old.remove();
    const st = doc.createElement("style");
    st.textContent = `
      #dopa-overlay {{ position: fixed; inset: 0; z-index: 999998; pointer-events: none; display: grid; place-items: center;
        background: radial-gradient(circle at center, rgba(255,215,0,.28), rgba(11,15,25,.0) 60%); animation: dopaFade 5.2s ease forwards; }}
      #dopa-overlay .flash {{ position: absolute; inset: 0; background: #fff; animation: dopaFlash .5s ease-out forwards; }}
      #dopa-overlay .box {{ text-align: center; animation: dopaZoom .9s cubic-bezier(.17,1.6,.4,1) both; }}
      #dopa-overlay .ttl {{ font-family: 'Orbitron','Noto Sans JP',sans-serif; font-weight: 900; font-size: clamp(42px, 9vw, 120px);
        background: linear-gradient(90deg,#FFD700,#FFF7AE,#FFB300,#FFD700); background-size: 300% 100%;
        -webkit-background-clip: text; background-clip: text; color: transparent; animation: dopaFlow 1.2s linear infinite;
        filter: drop-shadow(0 0 18px rgba(255,215,0,.9)) drop-shadow(0 0 48px rgba(255,51,102,.5)); }}
      #dopa-overlay .pay {{ margin-top: 8px; font-family: 'Orbitron',sans-serif; font-weight: 900; font-size: clamp(28px, 6vw, 72px);
        color: #00FF87; text-shadow: 0 0 14px #00FF87, 0 0 40px #00FF87; }}
      @keyframes dopaFlash {{ from {{ opacity: .85; }} to {{ opacity: 0; }} }}
      @keyframes dopaZoom {{ from {{ transform: scale(.2) rotate(-8deg); opacity: 0; }} to {{ transform: scale(1) rotate(0); opacity: 1; }} }}
      @keyframes dopaFlow {{ to {{ background-position: 300% 0; }} }}
      @keyframes dopaFade {{ 0%, 80% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}`;
    const wrap = doc.createElement("div"); wrap.id = "dopa-overlay";
    wrap.innerHTML = `<div class="flash"></div><div class="box"><div class="ttl">{html.escape(title)}</div><div class="pay" id="dopa-pay"></div></div>`;
    wrap.appendChild(st); doc.body.appendChild(wrap);
    const payEl = doc.getElementById("dopa-pay"), target = {payout_js};
    if (target !== null) {{
      const start = performance.now(), dur = 1800;
      (function tick(now) {{
        const p = Math.min((now - start) / dur, 1), e = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
        payEl.textContent = "¥" + Math.round(target * e).toLocaleString("ja-JP");
        if (p < 1) P.requestAnimationFrame(tick);
      }})(start);
    }}
    P.setTimeout(() => wrap.remove(), 5400);
  }}

  function fire() {{
    const confetti = P.confetti;
    if (!confetti) return;
    let petal = null;
    try {{ petal = confetti.shapeFromPath({{ path: "M10 0 C 16 4, 18 12, 10 20 C 2 12, 4 4, 10 0 Z M10 0 L 8 3 L 10 2 L 12 3 Z" }}); }} catch (e) {{}}
    // 中央から大爆発
    confetti({{ particleCount: 180, spread: 100, startVelocity: 55, origin: {{ y: .55 }}, colors: GOLD, scalar: 1.2, zIndex: 999999 }});
    // 左右から金色の連射
    const end = Date.now() + 3200;
    (function frame() {{
      confetti({{ particleCount: 6, angle: 60, spread: 60, origin: {{ x: 0, y: .75 }}, colors: GOLD, zIndex: 999999 }});
      confetti({{ particleCount: 6, angle: 120, spread: 60, origin: {{ x: 1, y: .75 }}, colors: GOLD, zIndex: 999999 }});
      if (Date.now() < end) P.requestAnimationFrame(frame);
    }})();
    // 上からゆっくり舞う桜吹雪
    let n = 0;
    const timer = P.setInterval(() => {{
      confetti({{ particleCount: 5, startVelocity: 0, ticks: 420, gravity: .35, drift: (Math.random() - .5) * 1.2,
        origin: {{ x: Math.random(), y: -0.05 }}, colors: SAKURA, shapes: petal ? [petal] : ["circle"],
        scalar: petal ? 1.6 : 1.0, zIndex: 999999 }});
      if (++n > 45) P.clearInterval(timer);
    }}, 80);
  }}

  if (P.__dopaLast === nonce) return;  // 同じ演出を再描画で二重に発火させない
  P.__dopaLast = nonce;
  overlay();
  if (P.confetti) fire();
  else {{
    const s = doc.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js";
    s.onload = fire; doc.head.appendChild(s);
  }}
}})();
</script>
"""
    components.html(body, height=0)


# ---------------------------------------------------------------- デモ画面

DEMO_HORSES = [
    # 馬番, 馬名, 騎手, 単勝オッズ, 3着内確率, 推定勝率, 評価, ステータス, 調教, パドック
    (7, "ミッキージャンプ", "ルメール", 2.4, 0.74, 0.38, "S",
     {"スピード": 92, "スタミナ": 78, "瞬発力": 88, "調子": 95, "適性": 84},
     "併せ馬で楽々先着。終い11.4秒の切れ", "毛ヅヤ抜群、気合い乗り十分"),
    (4, "ロングトールサリー", "川田将雅", 5.8, 0.55, 0.19, "A",
     {"スピード": 85, "スタミナ": 90, "瞬発力": 72, "調子": 80, "適性": 88},
     "坂路で自己ベスト更新", "落ち着き十分、踏み込み力強い"),
    (12, "クリムゾンバレット", "横山武史", 38.5, 0.21, 0.04, "B",
     {"スピード": 80, "スタミナ": 64, "瞬発力": 93, "調子": 76, "適性": 70},
     "時計は平凡も動きは軽快", "テンションやや高め、一発の雰囲気"),
]


def demo():
    st.set_page_config(page_title="DOPAMINE KEIBA", page_icon="🏇", layout="wide")
    inject_theme()
    neon_title("DOPAMINE KEIBA LOUNGE", "プレミアム競馬ラウンジ × サイバー解析 — 東京11R 天皇賞(秋) GI 芝2000m")

    # ① オッズ・期待値カード
    section_label("① ODDS × EXPECTED VALUE")
    marks = ["◎", "○", "☆"]
    labels = ["大本命", "妙味", "大穴"]
    cols = st.columns(3)
    for col, mark, label, h in zip(cols, marks, labels, DEMO_HORSES):
        number, name, jockey, odds, top3, win, *_ = h
        with col:
            odds_card(mark, number, name, jockey, odds, top3, win * odds, label=label)

    # ② 出走馬ステータス
    section_label("② PADDOCK STATUS")
    cols = st.columns(3)
    for col, h in zip(cols, DEMO_HORSES):
        number, name, _, odds, _, _, grade, stats, training, paddock = h
        with col:
            horse_status_card(number, name, grade, stats, training, paddock, odds=odds)

    # ③ 的中演出
    section_label("③ JACKPOT SIMULATION")
    st.caption("三連単を100円で買ったと仮定したシミュレーション。万馬券（払戻1万円以上）が出ると全画面で祝福します。")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("🎰 三連単シミュレーション", use_container_width=True):
        roll = random.random()
        if roll < 0.6:
            st.session_state.sim_result = ("miss", 0)
        elif roll < 0.85:
            st.session_state.sim_result = ("hit", random.randrange(1000, 9900, 10))
        else:
            st.session_state.sim_result = ("manba", random.randrange(10000, 480000, 10))
        st.session_state.sim_fire = True
    if c2.button("💥 万馬券を強制的に出す", use_container_width=True):
        st.session_state.sim_result = ("manba", random.randrange(50000, 980000, 10))
        st.session_state.sim_fire = True

    result = st.session_state.get("sim_result")
    if result:
        kind, payout = result
        if kind == "miss":
            st.markdown(f'<div style="font-size:1.4rem;color:#9CA3AF">不的中… 次こそ <span style="color:{RED}">一撃</span> を。</div>',
                        unsafe_allow_html=True)
        else:
            color = GOLD if kind == "manba" else EMERALD
            st.markdown(
                f'<div style="font-family:Orbitron;font-weight:900;font-size:2.2rem;color:{color};'
                f'text-shadow:0 0 16px {color}">的中！ 払戻 ¥{payout:,}</div>', unsafe_allow_html=True)
        if kind == "manba" and st.session_state.pop("sim_fire", False):
            celebrate(payout)
        st.session_state.pop("sim_fire", None)

    # 高確率予想が出たときの自動演出
    with st.sidebar:
        st.markdown("### 演出設定")
        auto = st.toggle("3着内確率70%以上の馬がいたら自動で祝福", value=False)
    if auto and any(h[4] >= 0.7 for h in DEMO_HORSES) and not st.session_state.get("auto_fired"):
        st.session_state.auto_fired = True
        best = max(DEMO_HORSES, key=lambda h: h[4])
        celebrate(title=f"鉄板候補 {best[0]}番 {best[1]}")
    if not auto:
        st.session_state.auto_fired = False


if __name__ == "__main__":
    demo()

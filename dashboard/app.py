"""Customer Churn Intelligence dashboard — world-class executive UI.

Run from the project root:  python -m streamlit run dashboard/app.py
Tip: place the supplied .streamlit/config.toml in your repo so Streamlit's
default widgets (slider fill, multi-select tags, checkboxes) use cobalt
instead of the default red primaryColor.
"""
"""Customer Churn Intelligence dashboard.

Run from the project root:  python -m streamlit run dashboard/app.py
"""
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import campaign_tab  # noqa: E402
from dashboard import charts  # noqa: E402
from dashboard import model_card_tab  # noqa: E402
from dashboard import upload_tab  # noqa: E402
from dashboard import whatif_tab  # noqa: E402
from dashboard.data import load_dashboard_data  # noqa: E402
from src.ai_recommendations import generate_recommendation, load_env  # noqa: E402
from src.explainability import describe_feature  # noqa: E402
from src.prediction import prepare_features  # noqa: E402


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_env()

# ============================================================
# Design system — light alabaster canvas, midnight slate chrome,
# cobalt primary, semantic risk palette (crimson / amber / teal)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
  --cobalt:#2563EB; --cobalt-d:#1D4ED8; --cobalt-soft:#EFF6FF; --cobalt-edge:#DBEAFE;
  --slate-900:#0F172A; --slate-800:#1E293B; --slate-700:#334155; --slate-600:#475569;
  --canvas:#DCE4F2; --surface:#FFFFFF; --surface-2:#F8FAFC; --surface-3:#EAF0FA;
  --border:#E2E8F0; --border-strong:#CBD5E1;
  --text:#0F172A; --text-2:#475569; --muted:#64748B; --muted-2:#94A3B8;
  --crimson:#DC2626; --crimson-d:#B91C1C;
  --amber:#D97706; --amber-d:#B45309;
  --teal:#0D9488; --teal-d:#0F766E; --emerald:#10B981;
  --sh-sm:0 1px 2px rgba(15,23,42,.04);
  --sh:0 1px 3px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.05);
  --sh-lg:0 12px 32px -12px rgba(15,23,42,.18);
  --sh-glow:0 8px 24px -8px rgba(37,99,235,.22);
  --r:14px; --r-sm:10px;
}

/* ---- Canvas + typography ----
   Visibly tinted cool slate-blue canvas (#DCE4F2) with a cobalt / indigo /
   teal radial-glow mesh. Applied to EVERY layer that can paint (body, stApp,
   stAppViewContainer, section[stMain]) with !important so no Streamlit version
   can leave a white surface — robust on deployed cloud. */
body, .stApp,
[data-testid="stAppViewContainer"], .stAppViewContainer,
section[data-testid="stMain"], [data-testid="stMain"], .main {
  background:
    radial-gradient(900px 540px at 8% -6%, rgba(37,99,235,.16), transparent 60%),
    radial-gradient(820px 500px at 105% 6%, rgba(99,102,241,.13), transparent 55%),
    radial-gradient(720px 470px at 50% 114%, rgba(13,148,136,.11), transparent 60%),
    var(--canvas) !important;
  background-attachment: fixed;
  color: var(--text);
}
header[data-testid="stHeader"] { background: var(--canvas) !important; border-bottom: none !important; }
.block-container { padding-top:5rem; padding-bottom:2rem; max-width:1440px; }
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
h1, h2, h3, .hero h1, .sec h3, .sb-head { font-family:'Plus Jakarta Sans','Inter',sans-serif; }
.kpi .v, .sb-badge b, .hero .pill b, .insight,
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:999px;
  border:2px solid transparent; background-clip:padding-box; }
::-webkit-scrollbar-thumb:hover { background:var(--cobalt); background-clip:padding-box; }

/* ============ SIDEBAR — deep midnight slate (#0F172A → #1E293B) ============ */
[data-testid="stSidebar"] {
  position:relative;
  background:linear-gradient(180deg,#090D16 0%, #111827 100%) !important;
  border-right:1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"]::before {
  content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(440px 260px at 80% 0%, rgba(37,99,235,.18), transparent 70%);
}
[data-testid="stSidebar"] * { color:#E2E8F0; }
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#94A3B8 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color:#CBD5E1 !important;
  font-weight:600; font-size:.82rem; }

/* dark slate input boxes (#1E293B, #F8FAFC text, #334155 border) */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] [data-baseweb="input"] input {
  background:#1E293B !important; color:#F8FAFC !important;
  border:1px solid #334155 !important; border-radius:9px; }
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder { color:#64748B !important; }

.sb-head { color:#94A3B8 !important; font-size:.72rem; font-weight:700; letter-spacing:.14em;
  text-transform:uppercase; margin:.1rem 0 .55rem; display:flex; align-items:center; gap:.55rem;
  font-family:'Plus Jakarta Sans','Inter',sans-serif; }
.sb-head::before { content:""; width:18px; height:2px; background:var(--cobalt); border-radius:2px; }

.sb-badge { display:inline-flex; align-items:center; gap:.4rem; background:var(--cobalt-soft);
  color:var(--cobalt-d) !important; border:1px solid var(--cobalt-edge); border-radius:999px;
  padding:.3rem .8rem; font-size:.76rem; font-weight:600; margin-top:.55rem; }
.sb-badge b { color:var(--cobalt-d) !important; font-size:.9rem; }

/* model status — emerald-tint pill, single line, no clip */
.sb-status {
  display:inline-flex; align-items:center; gap:.4rem; margin-top:.7rem;
  background:rgba(16,185,129,.14); border:1px solid rgba(52,211,153,.35);
  border-radius:999px; padding:.32rem .75rem; white-space:nowrap; max-width:100%;
}
.sb-dot { width:7px; height:7px; border-radius:50%; background:#34D399;
  box-shadow:0 0 0 3px rgba(52,211,153,.22); flex-shrink:0; }
.sb-status .t { color:#6EE7B7 !important; font-size:.76rem; font-weight:600; line-height:1.3; }
.sb-status .t b { color:#A7F3D0 !important; font-variant-numeric:tabular-nums; }

/* select / multiselect containers — dark translucent slate (#1E293B + #334155) */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="multiselect"] > div {
  background:#1E293B !important; border:1px solid #334155 !important; border-radius:9px; }
/* closed selectbox / multiselect value text — light on dark (bulletproof) */
[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="multiselect"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] [data-baseweb="multiselect"] [role="combobox"],
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="multiselect"] input { color:#F8FAFC !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg,
[data-testid="stSidebar"] [data-baseweb="multiselect"] svg { fill:#94A3B8 !important; color:#94A3B8 !important; }

/* open dropdown menu (popover) — dark slate with WHITE option text.
   The popover renders in a portal outside the sidebar; a dark menu (#1E293B)
   with white text matches the chrome and keeps options readable everywhere
   (incl. the What-if tab). The "*" + !important guarantees no baseweb class
   can paint option text black (which was making options invisible). */
[data-baseweb="menu"] { background:#1E293B !important; border:1px solid #334155 !important;
  border-radius:10px !important; box-shadow:0 8px 24px rgba(0,0,0,.35) !important; }
[data-baseweb="menu"] * { color:#FFFFFF !important; }
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] { background:rgba(37,99,235,.28) !important; }

/* ACTIVE filter chips — indigo pills (#1D4ED8 bg, #DBEAFE text). NEVER red. */
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background:#1D4ED8 !important; border:1px solid #1D4ED8 !important; border-radius:999px; }
[data-testid="stSidebar"] [data-baseweb="tag"] span,
[data-testid="stSidebar"] [data-baseweb="tag"] svg {
  color:#DBEAFE !important; fill:#DBEAFE !important; }

/* slider thumb — cobalt (fill colour comes from config.toml primaryColor) */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
  background:#fff; border:3px solid var(--cobalt); box-shadow:0 2px 8px rgba(37,99,235,.4); }

/* checkbox / switch accent — cobalt */
input[type="checkbox"] { accent-color:var(--cobalt); }
[role="switch"][aria-checked="true"] > div { background:var(--cobalt) !important; }

/* ============ HERO — compact, glassmorphic icon, export button ============ */
.hero {
  position:relative; overflow:hidden; display:flex; align-items:center;
  justify-content:space-between; gap:1.2rem;
  background:linear-gradient(120deg,#0F172A 0%, #1E293B 100%);
  color:#fff; padding:1.35rem 1.7rem; border-radius:18px; margin-bottom:1rem; box-shadow:var(--sh-lg);
}
.hero::after {
  content:""; position:absolute; right:-90px; top:-90px; width:300px; height:300px;
  background:radial-gradient(closest-side, rgba(37,99,235,.28), transparent 70%); pointer-events:none; }
.hero-main { display:flex; align-items:center; gap:1rem; position:relative; z-index:1; min-width:0; }
.hero-icon {
  width:46px; height:46px; flex-shrink:0; border-radius:13px;
  background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18);
  backdrop-filter:blur(8px); display:flex; align-items:center; justify-content:center; color:#fff;
}
.hero-icon svg { width:24px; height:24px; }
.hero-text { min-width:0; }
.hero h1 { margin:0; font-size:1.6rem; color:#fff; letter-spacing:-.02em; font-weight:700; }
.hero p { margin:.4rem 0 0; opacity:.82; font-size:.9rem; max-width:760px; line-height:1.45; }
.hero .pills { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.85rem; }
.hero .pill { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14);
  backdrop-filter:blur(6px); padding:.28rem .75rem; border-radius:999px;
  font-size:.74rem; font-weight:600; color:#CBD5E1; }
.hero .pill b { color:#fff; font-weight:700; }
.hero-export {
  flex-shrink:0; position:relative; z-index:1; display:inline-flex; align-items:center; gap:.45rem;
  background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); backdrop-filter:blur(8px);
  color:#fff !important; font-size:.82rem; font-weight:600; padding:.55rem .95rem; border-radius:10px;
  text-decoration:none; transition:all .18s ease; cursor:pointer;
}
.hero-export:hover { background:rgba(255,255,255,.18); transform:translateY(-1px); }
.hero-export svg { width:16px; height:16px; }

/* ============ TABS — clean text, white pill active, cobalt text/icon ============ */
.stTabs [data-baseweb="tab-list"] {
  gap:.3rem; border-bottom:1px solid var(--border); background:transparent;
  padding:.3rem 0; margin-bottom:.9rem; }
.stTabs [data-baseweb="tab"] {
  background:transparent; border-radius:8px; padding:.5rem .9rem;
  font-weight:600; font-size:.84rem; color:var(--muted); height:auto;
  transition:all .18s ease; white-space:nowrap; border:1px solid transparent; }
.stTabs [data-baseweb="tab"]:hover { background:rgba(37,99,235,.06); color:var(--cobalt); }
.stTabs [aria-selected="true"] {
  background:#FFFFFF !important; color:var(--cobalt) !important;
  border:1px solid var(--border) !important;
  box-shadow:0 1px 2px rgba(15,23,42,.06), 0 4px 12px -4px rgba(37,99,235,.16) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* ============ KPI cards — Nordic Slate: crisp white + soft top sheen ============
   Unified #FFFFFF surface with a subtle #F8FAFC top sheen and 1px #E2E8F0 border
   on ALL cards. The 5 Portfolio cards are differentiated ONLY by their icon
   badge + bottom insight pill (accent colors set per-card via --c / --c2).
   Text stays fully dark & legible: #0F172A numbers, #475569 labels, #64748B sub. */
.kpi {
  position:relative; background:linear-gradient(180deg,#F8FAFC 0%,#FFFFFF 34%);
  border:1px solid #E2E8F0; border-radius:var(--r); padding:.95rem 1rem .9rem;
  box-shadow:var(--sh); transition:transform .2s ease, box-shadow .2s ease; height:100%;
}
.kpi:hover { transform:translateY(-3px); box-shadow:var(--sh-lg); }
.kpi .ic { width:34px; height:34px; border-radius:9px; display:flex; align-items:center;
  justify-content:center; margin-bottom:.6rem; color:var(--c);
  background:color-mix(in srgb, var(--c) 13%, #FFFFFF); }
.kpi .ic svg { width:18px; height:18px; }
.kpi .l { font-size:.68rem; color:#475569; text-transform:uppercase;
  letter-spacing:.06em; font-weight:600; }
.kpi .v { font-size:1.7rem; font-weight:700; color:#0F172A; line-height:1.1;
  margin-top:.15rem; letter-spacing:-.01em; }
.kpi .s { font-size:.73rem; color:#64748B; margin-top:.2rem; }
.kpi .insight { display:inline-block; margin-top:.55rem; font-size:.7rem; font-weight:600;
  color:var(--c2); background:color-mix(in srgb, var(--c) 9%, #FFFFFF);
  border:1px solid color-mix(in srgb, var(--c) 22%, #E2E8F0);
  border-radius:999px; padding:.18rem .55rem; }

/* ============ Section header — single cobalt accent, sharp title ============ */
.sec { display:flex; align-items:center; gap:.6rem; margin:1.5rem 0 .55rem; }
.sec .bar { width:3px; height:18px; border-radius:3px; background:var(--cobalt); flex-shrink:0; }
.sec h3 { margin:0; font-size:16px; color:#0F172A; font-weight:600; line-height:1.25; letter-spacing:-.01em; }
.sec span { font-size:13px; color:#64748B; }

/* ============ Chart cards ============ */
[data-testid="stPlotlyChart"] { background:#F1F5F9; border:1px solid #CBD5E1;
  border-radius:var(--r); padding:.55rem .6rem .5rem; box-shadow:var(--sh-sm);
  transition:box-shadow .2s ease, transform .2s ease; }
[data-testid="stPlotlyChart"]:hover { box-shadow:0 4px 14px -6px rgba(15,23,42,.14); transform:translateY(-1px); }

/* ============ Dataframe ============ */
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:var(--r);
  overflow:hidden; box-shadow:var(--sh-sm); }
[data-testid="stDataFrame"] [data-testid="stTable"] thead th { background:var(--surface-3) !important; color:var(--text-2); }

/* ============ Badges / drivers ============ */
.badge { display:inline-block; padding:.26rem .8rem; border-radius:999px; color:#fff;
  font-weight:600; font-size:.78rem; vertical-align:middle; letter-spacing:.01em; }
.pill-chip { display:inline-block; padding:.2rem .65rem; border-radius:999px; font-size:.73rem;
  font-weight:600; background:var(--cobalt-soft); color:var(--cobalt-d); border:1px solid var(--cobalt-edge); }
.driver { display:flex; align-items:flex-start; gap:.5rem; padding:.5rem .75rem; border-radius:9px;
  background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--dc);
  margin-bottom:.35rem; font-size:.88rem; color:var(--text); }
.driver .m { color:var(--dc); display:flex; align-items:center; flex-shrink:0; }
.driver .m svg { width:15px; height:15px; }

/* ============ Buttons — cobalt ============ */
.stDownloadButton > button, .stButton > button {
  background:var(--cobalt) !important; color:#fff !important;
  border:1px solid var(--cobalt) !important; border-radius:9px !important; font-weight:600 !important;
  padding:.5rem 1.05rem !important; box-shadow:0 1px 2px rgba(37,99,235,.2) !important;
  transition:all .15s !important; }
.stDownloadButton > button:hover, .stButton > button:hover { background:var(--cobalt-d) !important; transform:translateY(-1px); }

/* ============ Alerts ============ */
[data-testid="stAlertContainer"] { border-radius:var(--r) !important;
  border-left:4px solid var(--cobalt) !important; }
[data-testid="stAlertContainerInfo"] { border-left-color:var(--cobalt) !important; }
[data-testid="stAlertContainerWarning"] { border-left-color:var(--amber) !important; }
[data-testid="stAlertContainerError"] { border-left-color:var(--crimson) !important; }
[data-testid="stAlertContainerSuccess"] { border-left-color:var(--teal) !important; }

/* ============ Captions / headings / dividers ============ */
[data-testid="stCaptionContainer"] { color:var(--muted); font-size:.78rem; }
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 { color:var(--text); }
hr, [data-testid="stDivider"] { border-color:var(--border) !important; }

/* ============ Footer ============ */
.foot { text-align:center; color:var(--muted); font-size:.76rem; margin-top:2rem;
  padding-top:1.1rem; border-top:1px solid var(--border); }
.foot b { color:var(--text-2); }

/* ============ Fade in ============ */
@keyframes _fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
.block-container > div { animation:_fadeIn .4s ease both; }

/* main-area slider (What-if) — cobalt thumb */
[data-baseweb="slider"] [role="slider"] { background:var(--cobalt); border:2px solid var(--surface); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- Lucide-style line icons (monochrome, inherit currentColor) ----------
_SVG = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
ICON_HERO = f'<svg viewBox="0 0 24 24" {_SVG}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
ICON_USERS = (f'<svg viewBox="0 0 24 24" {_SVG}><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>'
              f'<circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/>'
              f'<path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>')
ICON_LOGOUT = (f'<svg viewBox="0 0 24 24" {_SVG}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>'
               f'<polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>')
ICON_TREND_DOWN = (f'<svg viewBox="0 0 24 24" {_SVG}><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/>'
                   f'<polyline points="16 17 22 17 22 11"/></svg>')
ICON_FLAME = (f'<svg viewBox="0 0 24 24" {_SVG}><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3'
              f'-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0'
              f'c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>')
ICON_DOLLAR = (f'<svg viewBox="0 0 24 24" {_SVG}><line x1="12" y1="1" x2="12" y2="23"/>'
               f'<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>')
ICON_CARD = (f'<svg viewBox="0 0 24 24" {_SVG}><rect x="1" y="4" width="22" height="16" rx="2"/>'
             f'<line x1="1" y1="10" x2="23" y2="10"/></svg>')
ICON_EXPORT = (f'<svg viewBox="0 0 24 24" {_SVG}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
               f'<polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>')
ICON_ARROW_UP = (f'<svg viewBox="0 0 24 24" {_SVG}><line x1="12" y1="19" x2="12" y2="5"/>'
                 f'<polyline points="5 12 12 5 19 12"/></svg>')
ICON_ARROW_DOWN = (f'<svg viewBox="0 0 24 24" {_SVG}><line x1="12" y1="5" x2="12" y2="19"/>'
                   f'<polyline points="19 12 12 19 5 12"/></svg>')


# ---------- Small helpers ----------
def show(fig) -> None:
    """Plotly chart: Nordic Slate polish — transparent paper (Soft Cool Ice-Tint
    #F1F5F9 card shows), gentle cool plot canvas (#E6ECF5) on cartesian charts
    with clean light #F8FAFC gridlines, dark Inter text (#0F172A titles,
    #1E293B ticks/legend), crisp 2px white segment borders on donut/treemap.
    Version-robust rendering."""
    _CART = {"scatter", "scattergl", "bar", "histogram", "histogram2d",
             "histogram2dcontour", "box", "violin", "heatmap", "contour",
             "waterfall", "ohlc", "candlestick"}
    _is_cart = any(getattr(_tr, "type", None) in _CART for _tr in fig.data)
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#E6ECF5" if _is_cart else "rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#0F172A", size=12),
            margin=dict(l=10, r=16, t=50, b=10),
            title_font=dict(family="Plus Jakarta Sans, Inter, sans-serif",
                            color="#0F172A", size=14),
            legend=dict(font=dict(color="#1E293B", size=11),
                        bgcolor="rgba(0,0,0,0)", itemsizing="constant",
                        itemwidth=24, borderwidth=0),
        )
        if _is_cart:
            fig.update_xaxes(gridcolor="#F8FAFC", zerolinecolor="#F8FAFC",
                             linecolor="#CBD5E1",
                             tickfont=dict(color="#1E293B", size=11),
                             title_font=dict(color="#0F172A", size=12))
            fig.update_yaxes(gridcolor="#F8FAFC", zerolinecolor="#F8FAFC",
                             linecolor="#CBD5E1",
                             tickfont=dict(color="#1E293B", size=11),
                             title_font=dict(color="#0F172A", size=12))
        for _tr in fig.data:
            if getattr(_tr, "type", None) in ("treemap", "pie", "sunburst"):
                _tr.marker.line.width = 2
                _tr.marker.line.color = "#FFFFFF"
    except Exception:
        pass
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def kpi(column, label: str, value: str, sub: str, color: str, icon: str = "",
        color2: str = "", insight: str = "") -> None:
    color2 = color2 or color
    insight_html = f'<div class="insight">{insight}</div>' if insight else ""
    column.markdown(
        f'<div class="kpi" style="--c:{color};--c2:{color2}">'
        f'<div class="ic">{icon}</div>'
        f'<div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div>{insight_html}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "", icon: str = "") -> None:
    # icon kept for call compatibility but intentionally not rendered (clean header)
    st.markdown(
        f'<div class="sec"><div class="bar"></div>'
        f'<div><h3>{title}</h3><span>{subtitle}</span></div></div>',
        unsafe_allow_html=True)


def sb_head(text: str) -> None:
    st.sidebar.markdown(f'<div class="sb-head">{text}</div>', unsafe_allow_html=True)


def sb_badge(value: str, label: str) -> None:
    st.sidebar.markdown(
        f'<div class="sb-badge"><b>{value}</b> {label}</div>', unsafe_allow_html=True)


def sb_status(html_text: str) -> None:
    st.sidebar.markdown(
        f'<div class="sb-status"><span class="sb-dot"></span>'
        f'<span class="t">{html_text}</span></div>', unsafe_allow_html=True)


def model_abbr(name: str) -> str:
    """Compact model label for the status pill (e.g. Logistic Regression → LR)."""
    key = name.strip().lower()
    table = {"logistic regression": "LR", "logistic": "LR",
             "random forest": "RF", "gradient boosting": "GBM",
             "xgboost": "XGB", "lightgbm": "LGBM", "catboost": "CB"}
    if key in table:
        return table[key]
    return "".join(w[0] for w in key.replace("-", " ").split() if w)[:4].upper() or "ML"


@st.cache_data(show_spinner=False)
def build_features_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_features(frame).reset_index(drop=True)


try:
    data, shap_df, meta = load_dashboard_data()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

SHAP_COLUMNS = [c for c in shap_df.columns if c != "customer_id"]
MIN_IMPACT = 0.05
MEDIUM_T, HIGH_T = meta["risk_thresholds"]["medium"], meta["risk_thresholds"]["high"]
X_all = build_features_frame(data)

# ---------- Sidebar ----------
sb_head("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(data["contract"].unique()),
                                   default=sorted(data["contract"].unique()))
internet = st.sidebar.multiselect("Internet service", sorted(data["internet_service"].unique()),
                                  default=sorted(data["internet_service"].unique()))
tenure_max = int(data["tenure_months"].max())
tenure_range = st.sidebar.slider("Tenure (months)", 0, tenure_max, (0, tenure_max),
                                 key="tenure_filter_v2")

filtered = data[data["contract"].isin(contracts) & data["internet_service"].isin(internet)
                & data["tenure_months"].between(*tenure_range)]
if filtered.empty:
    st.warning("No customers match these filters.")
    st.stop()
active = filtered[filtered["churn"] == 0]
sb_badge(f"{len(filtered):,}", f"selected · of {len(data):,}")

st.sidebar.divider()
sb_head("Selected customer")
typed = st.sidebar.text_input("Customer ID", placeholder="e.g. 5150-ITWWB").strip()
ranked_ids = filtered.sort_values("churn_probability", ascending=False)["customer_id"].tolist()
chosen = typed if typed else st.sidebar.selectbox("...or pick (highest risk first)", ranked_ids)
if chosen not in set(data["customer_id"]):
    st.sidebar.error(f"Customer '{chosen}' not found.")
    st.stop()
sb_status(f"Model Ready · <b>{model_abbr(meta['model_name'])}</b> (θ = <b>{meta['threshold']:.2f}</b>)")

# ---------- Hero (with Export Report) ----------
_export_cols = ["customer_id", "churn_probability", "risk_level", "monthly_charges",
                "revenue_at_risk", "contract", "tenure_months"]
_export_df = filtered.sort_values("churn_probability", ascending=False)[_export_cols]
_export_href = "data:text/csv;charset=utf-8," + urllib.parse.quote(_export_df.to_csv(index=False))

st.markdown(
    f'<div class="hero">'
    f'<div class="hero-main">'
    f'<div class="hero-icon">{ICON_HERO}</div>'
    f'<div class="hero-text">'
    f'<h1>Customer Churn Intelligence</h1>'
    f'<p>Predict who may leave, understand why, and decide who to contact first. '
    f'Scores come from a statistical model and show associations, not causes.</p>'
    f'<div class="pills">'
    f'<span class="pill">Model <b>{meta["model_name"]}</b></span>'
    f'<span class="pill">Threshold <b>{meta["threshold"]:.2f}</b></span>'
    f'<span class="pill">Customers <b>{len(data):,}</b></span>'
    f'<span class="pill">Selected <b>{len(filtered):,}</b></span>'
    f'</div></div></div>'
    f'<a class="hero-export" href="{_export_href}" download="churn_intelligence_report.csv">'
    f'{ICON_EXPORT}<span>Export Report</span></a>'
    f'</div>', unsafe_allow_html=True)

tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g, tab_h, tab_i = st.tabs(
    ["Executive Overview", "Customer Analytics", "Churn Prediction",
     "Explainable AI", "AI Retention Advisor", "Upload & Score",
     "What-if", "Campaign", "Model Card"])


def customer_facts(customer_id: str) -> dict:
    """Facts for one customer from saved scores and SHAP (same shape as Phase 7)."""
    i = data.index[data["customer_id"] == customer_id][0]
    x, contrib, row = X_all.iloc[i], shap_df.iloc[i][SHAP_COLUMNS], data.iloc[i]

    def pick(series):
        return [{"feature": f, "description": describe_feature(f, x[f]), "impact": round(float(v), 3)}
                for f, v in series.items()]

    return {
        "customer_id": customer_id,
        "churn_probability": float(row["churn_probability"]),
        "risk_level": str(row["risk_level"]),
        "monthly_charges": float(row["monthly_charges"]),
        "revenue_at_risk": float(row["revenue_at_risk"]),
        "profile": {"tenure_months": int(x["tenure_months"]), "contract": str(x["contract"]),
                    "internet_service": str(x["internet_service"]),
                    "payment_method": str(x["payment_method"]),
                    "has_tech_support": bool(x["tech_support"]),
                    "has_online_security": bool(x["online_security"])},
        "drivers_up": pick(contrib[contrib >= MIN_IMPACT].sort_values(ascending=False).head(5)),
        "drivers_down": pick(contrib[contrib <= -MIN_IMPACT].sort_values().head(3)),
        "_contrib": contrib, "_x": x,
    }


def public(facts: dict) -> dict:
    return {k: v for k, v in facts.items() if not k.startswith("_")}


# ---------- A. Executive Overview ----------
with tab_a:
    section("Portfolio snapshot", "Active customers in the current filter")
    _churned = int(filtered["churn"].sum())
    _churn_rate = 100 * filtered["churn"].mean()
    _high_risk = int((active["risk_level"] == "High").sum())
    _rev_at_risk = float(active["revenue_at_risk"].sum())
    _pct_base = 100 * len(filtered) / len(data)
    _high_risk_share = 100 * _high_risk / max(len(active), 1)
    _avg_rev = _rev_at_risk / max(_high_risk, 1)
    k = st.columns(5)
    kpi(k[0], "Total customers", f"{len(filtered):,}", "in current filter",
        "#2563EB", ICON_USERS, "#1D4ED8", insight=f"{_pct_base:.0f}% of full base")
    kpi(k[1], "Churned", f"{_churned:,}", "already left",
        "#E11D48", ICON_LOGOUT, "#BE123C", insight=f"{_churn_rate:.1f}% of segment")
    kpi(k[2], "Churn rate", f"{_churn_rate:.1f}%", "churned / total",
        "#D97706", ICON_TREND_DOWN, "#B45309", insight=f"{_churned:,} customers lost")
    kpi(k[3], "High-risk active", f"{_high_risk:,}", "can still be retained",
        "#DC2626", ICON_FLAME, "#B91C1C", insight=f"{_high_risk_share:.1f}% of active base")
    kpi(k[4], "Revenue at risk", f"${_rev_at_risk:,.0f}", "monthly, probability x charge",
        "#EA580C", ICON_DOLLAR, "#C2410C", insight=f"Avg ${_avg_rev:.2f} / high-risk user")
    st.caption("High-risk and revenue at risk cover ACTIVE customers. Scores for customers the "
               "model was trained on are slightly optimistic.")
    section("Risk & revenue mix", "Where the exposure concentrates")
    left, right = st.columns([1, 2])
    with left:
        show(charts.risk_donut(active))
    with right:
        show(charts.revenue_treemap(active))
    section("Lift & segments", "Model ranking quality and contract effect")
    left, right = st.columns(2)
    with left:
        show(charts.gain_curve(filtered))
    with right:
        show(charts.churn_rate_bar(filtered, "contract", "Churn rate by contract"))
    st.caption("Gain curve is computed on all customers in the filter, so it is slightly optimistic "
               "(the honest test-set result is in reports/model_evaluation.md).")

# ---------- B. Customer Analytics ----------
with tab_b:
    section("Distribution & segments", "How churn spreads across the base")
    c1, c2 = st.columns(2)
    with c1:
        show(charts.contract_tenure_heatmap(filtered))
    with c2:
        show(charts.churn_distribution(filtered))
    show(charts.risk_scatter(active))
    section("Service & billing signals", "Churn by plan, payment and tenure")
    c3, c4 = st.columns(2)
    with c3:
        show(charts.churn_rate_bar(filtered, "internet_service", "Churn by internet service"))
    with c4:
        show(charts.churn_rate_bar(filtered, "payment_method", "Churn by payment method"))
    c5, c6 = st.columns(2)
    with c5:
        show(charts.churn_rate_bar(filtered, "tenure_band", "Churn by tenure band"))
    with c6:
        show(charts.tenure_curve(filtered))
    show(charts.revenue_by_contract(filtered))

# ---------- C. Churn Prediction ----------
facts = customer_facts(chosen)
risk_color = {"High": "#DC2626", "Medium": "#D97706", "Low": "#0D9488"}[facts["risk_level"]]
with tab_c:
    st.markdown(f"### Customer {chosen} "
                f'<span class="badge" style="background:{risk_color}">{facts["risk_level"]} risk</span>',
                unsafe_allow_html=True)
    st.caption("MODEL PREDICTION (statistical model)")
    left, right = st.columns([1, 1])
    with left:
        show(charts.probability_gauge(facts["churn_probability"], MEDIUM_T, HIGH_T))
    with right:
        m1, m2 = st.columns(2)
        kpi(m1, "Monthly charge", f"${facts['monthly_charges']:,.2f}", "current bill",
            "#2563EB", ICON_CARD, "#1D4ED8",
            insight=f"{facts['profile']['contract']} · {facts['profile']['tenure_months']}mo tenure")
        kpi(m2, "Revenue at risk", f"${facts['revenue_at_risk']:,.2f}", "per month, estimate",
            "#DC2626", ICON_DOLLAR, "#B91C1C",
            insight=f"{facts['churn_probability'] * 100:.0f}% churn probability")
        st.write("")
        st.markdown("**Main factors raising the risk**")
        for d in facts["drivers_up"] or [{"description": "None above the impact threshold"}]:
            st.markdown(f'<div class="driver" style="--dc:#DC2626"><span class="m">{ICON_ARROW_UP}</span>'
                        f'<span>{d["description"]}</span></div>', unsafe_allow_html=True)
        for d in facts["drivers_down"]:
            st.markdown(f'<div class="driver" style="--dc:#0D9488"><span class="m">{ICON_ARROW_DOWN}</span>'
                        f'<span>{d["description"]} <i style="color:var(--muted)">(lowers risk)</i></span></div>',
                        unsafe_allow_html=True)
    st.caption(f"Risk tiers: Medium at {MEDIUM_T:.2f} or above, High at {HIGH_T:.2f} or above.")
    if int(data.loc[data["customer_id"] == chosen, "churn"].iloc[0]) == 1:
        st.info("This customer has already churned in the dataset.")

# ---------- D. Explainable AI ----------
with tab_d:
    section("Global vs local drivers", "Model-wide importance versus this customer")
    left, right = st.columns(2)
    with left:
        show(charts.shap_global_bar(shap_df))
    with right:
        descriptions = {f: describe_feature(f, facts["_x"][f]) for f in SHAP_COLUMNS}
        show(charts.shap_waterfall(facts["_contrib"], descriptions))
    st.caption("SHAP values are in log-odds and describe what the MODEL does, not what causes churn. "
               f"Right chart: customer {chosen}.")

# ---------- E. AI Retention Advisor ----------
with tab_e:
    section("Recommendation", f"Personalised retention plan for {chosen}")
    use_llm = st.checkbox("Use LLM if an API key is configured", value=True)
    rec = generate_recommendation(public(facts), use_llm)
    st.markdown(f"**RECOMMENDATION** "
                f'<span class="pill-chip">{rec["source"]}</span>  |  priority: '
                f"**{rec['priority']}**", unsafe_allow_html=True)
    st.write(rec["recommendation"])
    st.write(f"**Suggested action:** {rec['suggested_action']}")
    st.write(f"**Expected objective:** {rec['expected_objective']}")

    st.divider()
    section("Contact list", "Highest revenue at risk — call these first")
    n = st.slider("Number of customers", 5, 100, 20, key="contact_n")
    top = active[active["risk_level"].isin(["High", "Medium"])].nlargest(n, "revenue_at_risk")
    rows = [generate_recommendation(public(customer_facts(cid)), False) for cid in top["customer_id"]]
    report = pd.DataFrame(rows)
    if report.empty:
        st.info("No High or Medium risk active customers in the current filter.")
    else:
        report["main_reasons"] = report["main_reasons"].apply("; ".join)
        show_table(report[["customer_id", "churn_probability", "risk_level", "priority",
                           "revenue_at_risk", "suggested_action", "expected_objective", "source"]])
        st.download_button("⬇ Download recommendation report (CSV)", report.to_csv(index=False),
                           file_name="retention_recommendations.csv", mime="text/csv")

st.markdown(f'<div class="foot">Model: <b>{meta["model_name"]}</b> · threshold <b>{meta["threshold"]:.2f}</b> · '
            'Business assumptions (offer cost, success rate) are estimates, not dataset facts · '
            'Associations, not causes</div>', unsafe_allow_html=True)


# ---------- F. Upload & Score ----------
with tab_f:
    upload_tab.render()


# ---------- G. What-if ----------
with tab_g:
    whatif_tab.render(data, chosen)

# ---------- H. Campaign simulator ----------
with tab_h:
    campaign_tab.render(data, meta)

# ---------- I. Model card ----------
with tab_i:
    model_card_tab.render(data, meta)
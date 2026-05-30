"""Streamlit dashboard for the Zim telco churn project.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.generate_data import (
    USD_TO_ZWL,
    MNOS,
    ISPS,
    CONTRACTS,
    PAYMENT_METHODS,
    BUNDLES,
    PROVINCES,
    LOCATION_TYPES,
    write_csv,
)

DATA_PATH = Path("data/churn_data.csv")
RNG = 42

# ---- Editorial palette -----------------------------------------------------
BRAND = "#B3361E"      # editorial coral / brick — the one strong accent
BRAND2 = "#C4583B"     # warmer brick, used for accents on darker areas
INK = "#1A1A17"        # near-black headings
BODY = "#5B564B"       # warm muted body text
GOOD = "#16794C"       # forest green (stayed / safe)
WARN = "#B4690E"       # amber (watch)
BLUE = "#3A5A8A"
GREY = "#9A9488"
SOFT = "#F3F1EA"       # warm card background
LINE = "#E7E3DA"       # warm gridlines / borders
PAPER = "#FBFAF7"      # page background
FONT = "Inter"
SERIF = "Fraunces"
CORAL_SCALE = ["#F8E8E2", "#EFC5B5", "#DF937B", "#C45434", "#8C3018"]
SEQ = [BRAND, WARN, GOOD, BLUE, "#7A5B8A", GREY]

st.set_page_config(
    page_title="Subscriber retention",
    page_icon="•",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Global styling — compact header, a responsive grid of rich KPI cards, and a
# layout that stacks cleanly on a phone (no big colour banner).
# --------------------------------------------------------------------------- #
def inject_css(accent: str, accent_dark: str, feature_bg: str) -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
        html, body, [class*="css"], .stMarkdown, p, span, div, label, input, button, textarea {{
            font-family: '{FONT}', system-ui, sans-serif; }}
        .stApp {{ background: {PAPER}; }}
        #MainMenu, footer, header[data-testid="stHeader"] {{ display: none; }}
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }}

        .head {{ margin: 0 0 18px 0; }}
        .head .eyebrow {{ font-size:12px; font-weight:600; letter-spacing:.4px;
            text-transform:uppercase; color:{accent}; }}
        .head h1 {{ font-family:'{SERIF}', serif; font-size:34px; font-weight:600;
            color:{INK}; margin:8px 0 6px 0; letter-spacing:-.5px; line-height:1.18; }}
        .head p {{ font-size:15px; color:{BODY}; margin:0; max-width:720px; line-height:1.55; }}

        .kpigrid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(215px,1fr));
            gap:14px; margin:6px 0 4px 0; }}
        .kpi {{ background:#fff; border:1px solid {LINE}; border-radius:16px; padding:18px 20px;
            box-shadow:0 1px 2px rgba(26,26,23,.03); }}
        .kpi.feature {{ background:{feature_bg}; border:1px solid {LINE}; }}
        .kpi.feature .k {{ color:{accent_dark}; opacity:.9; }}
        .kpi.feature .v {{ color:{accent_dark}; }}
        .kpi .k {{ font-size:12.5px; font-weight:500; color:{BODY};
            letter-spacing:.2px; text-transform:uppercase; }}
        .kpi .v {{ font-family:'{SERIF}', serif; font-size:30px; font-weight:500; color:{INK};
            letter-spacing:-.5px; line-height:1.12; margin-top:6px; }}
        .kpi .v2 {{ font-family:'{SERIF}', serif; font-size:26px; font-weight:500; color:{INK};
            letter-spacing:-.3px; line-height:1.15; margin-top:4px; }}
        .kpi .s {{ font-size:12px; color:{GREY}; margin-top:4px; }}
        .kpi-row {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
        .ring {{ position:relative; width:60px; height:60px; border-radius:50%; flex:0 0 auto; }}
        .ring span {{ position:absolute; inset:7px; background:#fff; border-radius:50%;
            display:grid; place-items:center; font-size:13px; font-weight:600; color:{INK}; }}
        .segbar {{ display:flex; height:9px; border-radius:6px; overflow:hidden;
            margin-top:13px; background:{SOFT}; }}
        .segbar > div {{ height:100%; }}
        .leg {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; font-size:11.5px; color:{BODY}; }}
        .leg i {{ width:9px; height:9px; border-radius:3px; display:inline-block; margin-right:4px; }}

        .callout {{ border-radius:14px; padding:14px 18px; margin:8px 0 20px 0;
            font-size:14.5px; line-height:1.6; color:{INK}; border:1px solid {LINE}; }}
        .sec {{ margin: 28px 0 6px 0; }}
        .sec h3 {{ font-family:'{SERIF}', serif; font-size:23px; font-weight:500;
            color:{INK}; margin:0; letter-spacing:-.2px; }}
        .sec p {{ font-size:14.5px; color:{BODY}; margin:4px 0 0 0; line-height:1.5; }}

        .stTabs [data-baseweb="tab-list"] {{ gap:0; background:transparent; padding:0;
            border-bottom:1px solid {LINE}; border-radius:0; flex-wrap:wrap; }}
        .stTabs [data-baseweb="tab"] {{ height:auto; padding:10px 16px; border-radius:0;
            font-weight:500; font-size:15px; color:{BODY}; background:transparent; }}
        .stTabs [aria-selected="true"] {{ background:transparent; color:{accent};
            box-shadow:none; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background:{accent}; height:2px; }}
        .stTabs [data-baseweb="tab-border"] {{ display:none; }}

        [data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE}; border-radius:14px;
            padding:14px 18px; box-shadow:0 1px 2px rgba(26,26,23,.03); }}
        [data-testid="stMetricValue"] {{ font-family:'{SERIF}', serif; font-weight:500; color:{INK}; }}
        [data-testid="stMetricLabel"] p {{ font-weight:500; color:{BODY}; }}

        .stButton > button {{ font-size:14px; font-weight:500; border-radius:10px;
            border:1px solid {LINE}; background:#fff; color:{INK}; padding:8px 18px; }}
        .stButton > button:hover {{ border-color:{accent}; color:{accent}; }}

        @media (max-width: 640px) {{
            .block-container {{ padding-left:1rem; padding-right:1rem; padding-top:1.2rem; }}
            .head h1 {{ font-size:25px; }}
            .head p {{ font-size:14px; }}
            .kpi .v {{ font-size:26px; }}
            .kpi .v2 {{ font-size:22px; }}
            .sec h3 {{ font-size:19px; }}
            .stTabs [data-baseweb="tab"] {{ padding:8px 12px; font-size:14px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ring(pct: float, color: str) -> str:
    return (f'<div class="ring" style="background:conic-gradient({color} '
            f'{pct*3.6:.0f}deg, #eef0f4 0);"><span>{pct:.0f}%</span></div>')


def segbar(parts: list[tuple[str, float, str]]) -> str:
    segs = "".join(f'<div style="width:{f*100:.1f}%;background:{c}"></div>'
                   for _, f, c in parts)
    leg = "".join(f'<span><i style="background:{c}"></i>{l}</span>' for l, _, c in parts)
    return f'<div class="segbar">{segs}</div><div class="leg">{leg}</div>'


BRAND_DARK = "#8C3018"
FEATURE_BG = "#F8E8E2"
inject_css(BRAND, BRAND_DARK, FEATURE_BG)


def usd_zwl(usd: float, dp: int = 0) -> str:
    return f"${usd:,.{dp}f} (≈ZiG {usd * USD_TO_ZWL:,.{dp}f})"


def callout(text: str, tone: str = "brand") -> None:
    bg = {"brand": "#F8E8E2", "good": "#E8EFE8", "warn": "#F3EBD8",
          "neutral": SOFT}[tone]
    bar = {"brand": BRAND, "good": GOOD, "warn": WARN, "neutral": GREY}[tone]
    st.markdown(
        f'<div class="callout" style="background:{bg};'
        f'border-left:3px solid {bar};">{text}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str | None = None) -> None:
    st.markdown(
        f'<div class="sec"><h3>{title}</h3>'
        f'{f"<p>{sub}</p>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )


def stat_card(col, k: str, v: str, s: str = "") -> None:
    col.markdown(
        f'<div class="card"><div class="k">{k}</div>'
        f'<div class="v">{v}</div><div class="s">{s}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=340, legend=False):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=34, b=8),
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
    )
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT, size=15, color=INK))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Data + model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading subscribers...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        write_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "1-2y", "2-4y", "4y+"],
    )
    return df


@st.cache_resource(show_spinner="Training the model...")
def train_and_score(df: pd.DataFrame):
    drop_cols = ["Churn", "tenure_bucket", "MonthlyChargesZWL", "TotalChargesZWL"]
    y = df["Churn"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    numeric = [
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
        "LoadSheddingHoursPerDay", "SupportCalls90d", "DataUsageGB",
    ]
    numeric = [c for c in numeric if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG
    )
    pre.fit(X_train)
    X_train_v, X_test_v = pre.transform(X_train), pre.transform(X_test)
    X_train_bal, y_train_bal = SMOTE(random_state=RNG).fit_resample(X_train_v, y_train)

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
        random_state=RNG, n_jobs=-1,
    )
    model.fit(X_train_bal, y_train_bal)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_v)[:, 1])
    all_probs = model.predict_proba(pre.transform(X))[:, 1]
    return pre, model, test_auc, all_probs


df = load_data()
pre, model, test_auc, all_probs = train_and_score(df)
df = df.assign(churn_prob=all_probs)

n_total = len(df)
n_at_risk = int((df["churn_prob"] >= 0.5).sum())
revenue_at_risk = float(df.loc[df["churn_prob"] >= 0.5, "MonthlyCharges"].sum())
annualised = revenue_at_risk * 12
top100_revenue = float(df.nlargest(100, "churn_prob")["MonthlyCharges"].sum())
pct_risk = n_at_risk / n_total * 100

# --------------------------------------------------------------------------- #
# Compact header + a grid of rich KPI cards
# --------------------------------------------------------------------------- #
contract_mix = df["Contract"].value_counts(normalize=True)
mix_parts = [
    ("Month-to-month", float(contract_mix.get("Month-to-month", 0)), BRAND),
    ("One year", float(contract_mix.get("One year", 0)), WARN),
    ("Two year", float(contract_mix.get("Two year", 0)), GOOD),
]

st.markdown(
    f"""
    <div class="head">
      <div class="eyebrow">Subscriber retention &middot; Zimbabwe ISP</div>
      <h1>Who's likely to leave this cycle — and what it costs you</h1>
      <p>A churn model trained on contract, payment, network and load-shedding
         signals. Here's who's at risk, why they go, and who to call first.</p>
    </div>
    <div class="kpigrid">
      <div class="kpi feature">
        <div class="k">Revenue at risk over the next year</div>
        <div class="v">${annualised:,.0f}</div>
        <div class="s">≈ ZiG {annualised * USD_TO_ZWL:,.0f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-row">
          <div>
            <div class="k">Likely to churn</div>
            <div class="v2">{n_at_risk:,}</div>
            <div class="s">of {n_total:,} subscribers</div>
          </div>
          {ring(pct_risk, BRAND)}
        </div>
      </div>
      <div class="kpi">
        <div class="kpi-row">
          <div>
            <div class="k">Model accuracy</div>
            <div class="v2">{test_auc*100:.0f}%</div>
            <div class="s">AUC {test_auc:.3f}</div>
          </div>
          {ring(test_auc*100, GOOD)}
        </div>
      </div>
      <div class="kpi">
        <div class="k">Contract mix</div>
        <div class="v2">{n_total:,}</div>
        <div class="s">how the base is split today</div>
        {segbar(mix_parts)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

callout(
    f"Save just the 100 highest-risk subscribers and you keep about "
    f"<b>${top100_revenue*12:,.0f}</b> a year (≈ZiG {top100_revenue*12*USD_TO_ZWL:,.0f}). "
    f"Winning a brand-new customer costs roughly 5–7× what a retention offer does — so even "
    f"a small EcoCash auto-pay discount pays for itself if it keeps a handful of them.",
    tone="brand",
)

tab_why, tab_calls, tab_score = st.tabs([
    "Why customers leave",
    "Who to call first",
    "Score a customer",
])

# --------------------------------------------------------------------------- #
with tab_why:
    section("From contract to outcome",
            "How subscribers flow from their plan, through tenure, to staying or leaving.")
    contracts = list(df["Contract"].unique())
    buckets = list(df["tenure_bucket"].cat.categories)
    outcomes = ["Stayed", "Left"]
    nodes = contracts + buckets + outcomes
    node_idx = {n: i for i, n in enumerate(nodes)}

    src, tgt, val, col = [], [], [], []
    for c in contracts:
        for b in buckets:
            n = int(((df["Contract"] == c) & (df["tenure_bucket"] == b)).sum())
            if n:
                src.append(node_idx[c]); tgt.append(node_idx[b]); val.append(n)
                col.append("rgba(255,90,95,0.18)" if c == "Month-to-month"
                           else "rgba(154,160,174,0.16)")
    for b in buckets:
        for o, mask in [("Stayed", df["Churn"] == 0), ("Left", df["Churn"] == 1)]:
            n = int(((df["tenure_bucket"] == b) & mask).sum())
            if n:
                src.append(node_idx[b]); tgt.append(node_idx[o]); val.append(n)
                col.append("rgba(255,90,95,0.42)" if o == "Left"
                           else "rgba(18,184,134,0.22)")

    node_colors = (
        [BRAND, WARN, GOOD][:len(contracts)]
        + [GREY] * len(buckets)
        + [GOOD, BRAND]
    )
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=18, thickness=16,
                  line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=col),
    ))
    st.plotly_chart(style_fig(fig, 480), use_container_width=True)

    mtm_short = df[(df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)]
    mtm_churn = mtm_short["Churn"].mean()
    two_year = df[df["Contract"] == "Two year"]["Churn"].mean()
    callout(
        f"Most people who leave are on month-to-month plans in their first year — about "
        f"<b>{mtm_churn*100:.0f}%</b> of them go. The same kind of customer on a two-year "
        f"contract leaves only <b>{two_year*100:.0f}%</b> of the time. Getting people onto "
        f"longer plans early is the single biggest lever you have.",
        tone="neutral",
    )

    section("Where the risk concentrates",
            "Darker means a higher share of that group churns.")
    pivot = df.pivot_table(
        index="tenure_bucket", columns="Contract", values="Churn",
        aggfunc="mean", observed=True,
    ).round(3)
    col_order = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[col_order]
    fig = px.imshow(
        pivot, color_continuous_scale=CORAL_SCALE, aspect="auto",
        text_auto=".0%", origin="lower", labels=dict(color="Churn rate"),
    )
    fig.update_coloraxes(colorbar=dict(tickformat=".0%"))
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    worst_idx = np.unravel_index(np.argmax(pivot.values), pivot.shape)
    worst_rate = pivot.values[worst_idx]
    worst_bucket = pivot.index[worst_idx[0]]
    worst_contract = pivot.columns[worst_idx[1]]
    n_worst = int(((df["tenure_bucket"] == worst_bucket) &
                   (df["Contract"] == worst_contract)).sum())
    callout(
        f"The hottest cell is new <b>{worst_contract}</b> subscribers in their "
        f"<b>{worst_bucket}</b> — {worst_rate*100:.0f}% of those {n_worst:,} people leave. "
        f"That's where a retention dollar goes furthest.",
        tone="warn",
    )

    section("What's specific to Zimbabwe",
            "A model trained on the US Telco dataset would miss these entirely: "
            "the network you're on, the power situation, and how you pay.")

    z1, z2 = st.columns(2)
    with z1:
        mno_view = (
            df.groupby("MNO").agg(customers=("Churn", "size"),
                                  churn_rate=("Churn", "mean")).reset_index()
            .sort_values("churn_rate")
        )
        fig = px.bar(
            mno_view, x="churn_rate", y="MNO", orientation="h",
            text=mno_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
            color="MNO",
            color_discrete_map={"Econet": BRAND, "NetOne": WARN, "Telecel": GREY},
            title="By mobile network",
        )
        fig.update_layout(xaxis_tickformat=".0%")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with z2:
        prov_view = (
            df.groupby("Province").agg(customers=("Churn", "size"),
                                       churn_rate=("Churn", "mean")).reset_index()
            .sort_values("churn_rate")
        )
        fig = px.bar(
            prov_view, x="churn_rate", y="Province", orientation="h",
            text=prov_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
            color="churn_rate", color_continuous_scale=CORAL_SCALE,
            title="By province",
        )
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    worst_mno = mno_view.iloc[-1]
    callout(
        f"<b>{worst_mno['MNO']}</b> subscribers leave most often "
        f"({worst_mno['churn_rate']*100:.0f}%), which tracks the real network-quality gap — "
        f"Econet has the widest 4G reach, NetOne is closing in on rural coverage, and Telecel "
        f"keeps losing ground. Churn is also highest where there's the most competition: in the "
        f"cities you can switch provider on the walk home; in rural areas there's often nowhere "
        f"else to go.",
        tone="neutral",
    )

    df["load_shed_bucket"] = pd.cut(
        df["LoadSheddingHoursPerDay"], bins=[-0.1, 2, 5, 8, 12, 24],
        labels=["0-2h", "2-5h", "5-8h", "8-12h", "12h+"],
    )
    ls_view = (
        df.groupby("load_shed_bucket", observed=True)
        .agg(churn_rate=("Churn", "mean"), customers=("Churn", "size")).reset_index()
    )
    fig = px.bar(
        ls_view, x="load_shed_bucket", y="churn_rate",
        text=ls_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
        color="churn_rate", color_continuous_scale=CORAL_SCALE,
        labels={"load_shed_bucket": "Daily load-shedding", "churn_rate": "Churn rate"},
        title="Power outages and churn",
    )
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    low = ls_view.iloc[0]["churn_rate"]
    high = ls_view.iloc[-1]["churn_rate"]
    callout(
        f"People living with 12+ hours of load-shedding a day leave at <b>{high*100:.0f}%</b>, "
        f"against <b>{low*100:.0f}%</b> for those barely affected — roughly {high/low:.1f}× more. "
        f"When the router's off half the day, the subscription feels like money wasted. A small "
        f"power-bank or solar-router promo tied to a one-year plan is worth testing in the "
        f"worst-hit suburbs.",
        tone="warn",
    )

    pay_view = (
        df.groupby("PaymentMethod")
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"),
             avg_monthly_usd=("MonthlyCharges", "mean")).reset_index()
        .sort_values("churn_rate")
    )
    fig = px.bar(
        pay_view, x="churn_rate", y="PaymentMethod", orientation="h",
        text=pay_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
        color="churn_rate", color_continuous_scale=CORAL_SCALE,
        title="How people pay, and whether they stay",
    )
    fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    callout(
        "Cash-deposit payers leave most — there's no standing arrangement keeping them around. "
        "EcoCash and bank debit orders sit at the other end: once someone's on auto-pay, leaving "
        "takes effort. Nudging cash payers onto EcoCash or InnBucks auto-pay, with a small ZiG "
        "discount as the carrot, is a cheap retention play.",
        tone="good",
    )

# --------------------------------------------------------------------------- #
with tab_calls:
    section("The 50 subscribers to phone this week",
            "Ranked by the model's churn probability, highest first.")

    watchlist = (
        df.nlargest(50, "churn_prob")[
            ["Contract", "tenure", "MonthlyCharges", "PaymentMethod",
             "InternetService", "MNO", "churn_prob"]
        ].rename(columns={
            "tenure": "Tenure (mo)", "MonthlyCharges": "Monthly $",
            "PaymentMethod": "Pays via", "InternetService": "Internet",
            "churn_prob": "Churn risk",
        }).reset_index(drop=True)
    )
    top_revenue = watchlist["Monthly $"].sum()
    callout(
        f"These 50 people are worth <b>${top_revenue:,.0f}</b> a month "
        f"(≈ZiG {top_revenue*USD_TO_ZWL:,.0f}) between them. The model thinks most are on their "
        f"way out — a call this week is far cheaper than winning them back later.",
        tone="brand",
    )
    st.dataframe(
        watchlist.style.format({"Monthly $": "${:.2f}", "Churn risk": "{:.0%}"})
                 .background_gradient(subset=["Churn risk"], cmap="OrRd"),
        use_container_width=True, hide_index=True, height=500,
    )

# --------------------------------------------------------------------------- #
with tab_score:
    section("Try a subscriber profile",
            "Change the details and watch the predicted risk move.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 4)
        monthly = st.slider("Monthly charge (USD)", 8.0, 130.0, 55.0, step=0.5)
        total = st.slider("Total paid so far (USD)", 0.0, 10000.0,
                          float(monthly * tenure), step=10.0)
        data_gb = st.slider("Monthly data (GB)", 0.1, 250.0, 20.0, step=0.5)
        st.caption(f"That's about ZiG {monthly * USD_TO_ZWL:,.0f} a month.")
    with col2:
        mno = st.selectbox("Mobile network", MNOS)
        internet = st.selectbox("Internet service", ISPS)
        bundle = st.selectbox("Usual bundle", BUNDLES)
        contract = st.selectbox("Contract", CONTRACTS)
        payment = st.selectbox("Pays via", PAYMENT_METHODS)
    with col3:
        province = st.selectbox("Province", PROVINCES)
        location_type = st.selectbox("Area", LOCATION_TYPES)
        load_shed = st.slider("Load-shedding (hrs/day)", 0.0, 18.0, 8.0, step=0.5)
        support = st.slider("Support calls (last 90 days)", 0, 20, 1)
        partner = st.radio("Has a partner?", ["Yes", "No"], horizontal=True)
        dependents = st.radio("Has dependents?", ["Yes", "No"], horizontal=True)
        senior = st.radio("Senior?", [0, 1], horizontal=True,
                          format_func=lambda x: "Yes" if x == 1 else "No")

    record = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "Province": province, "LocationType": location_type,
        "tenure": tenure, "MNO": mno, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet, "BundlePreference": bundle, "DataUsageGB": data_gb,
        "Contract": contract, "PaperlessBilling": "Yes", "PaymentMethod": payment,
        "LoadSheddingHoursPerDay": load_shed, "SupportCalls90d": support,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])
    prob = float(model.predict_proba(pre.transform(record))[0, 1])
    portfolio_avg = float(df["churn_prob"].mean())
    band = "high" if prob >= 0.6 else ("watch" if prob >= 0.3 else "stable")
    band_color = {"high": BRAND, "watch": WARN, "stable": GOOD}[band]
    band_label = {"high": "High risk", "watch": "Worth watching", "stable": "Comfortable"}[band]

    st.write("")
    left, right = st.columns([1, 1.2])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 52, "color": band_color, "family": FONT}},
            delta={"reference": portfolio_avg * 100,
                   "increasing": {"color": BRAND}, "decreasing": {"color": GOOD},
                   "suffix": " pts vs avg"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": band_color},
                "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#E9FBF3"},
                    {"range": [30, 60], "color": "#FFF6E9"},
                    {"range": [60, 100], "color": "#FFF1F0"},
                ],
                "threshold": {"line": {"color": INK, "width": 3},
                              "thickness": 0.75, "value": portfolio_avg * 100},
            },
        ))
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        st.caption(f"Dashed line is the average across all {n_total:,} subscribers.")

    with right:
        st.markdown(
            f'<div style="display:inline-block; background:{band_color}; color:#fff; '
            f'padding:8px 16px; border-radius:999px; font-weight:700; font-size:14px;">'
            f'{band_label} · {prob*100:.0f}%</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        similar = df[(df["Contract"] == contract) &
                     (df["tenure_bucket"] == pd.cut([tenure], [-1, 6, 12, 24, 48, 72],
                      labels=["0-6m", "7-12m", "1-2y", "2-4y", "4y+"])[0])]
        if len(similar) > 0:
            sim_churn = similar["Churn"].mean()
            st.markdown(
                f"This profile looks like **{len(similar):,}** subscribers you already have. "
                f"Of those, **{sim_churn*100:.0f}%** actually left. The model puts this "
                f"particular person at **{prob*100:.0f}%**."
            )
        discount_usd = monthly * 1.5
        if band == "high":
            st.markdown(
                f"**Worth a call.** A save offer of up to ${discount_usd:.0f} "
                f"(≈ZiG {discount_usd*USD_TO_ZWL:,.0f}) on a longer contract still beats "
                f"replacing them. If they're on cash, move them to EcoCash auto-pay; if "
                f"load-shedding is biting ({load_shed:.0f}h/day), throw in a router-backup promo. "
                f"Flag them in the CRM."
            )
        elif band == "watch":
            st.markdown(
                f"**Keep an eye on them.** Touch base in about a month, send a quick EcoCash "
                f"survey, and if they're stuck on a '{bundle}' plan that doesn't fit, suggest a "
                f"monthly bundle that does."
            )
        else:
            st.markdown(
                "**Comfortable for now.** No save offer needed — this is the kind of customer to "
                "cross-sell to, ask for a referral, or invite to review you."
            )

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

# ---- Fintech palette -------------------------------------------------------
BRAND = "#FF5A5F"      # hot coral, the one strong brand colour
BRAND2 = "#FF8A5B"     # peach, for the hero gradient
INK = "#16161D"        # near-black headings
BODY = "#5B6172"       # muted body text
GOOD = "#12B886"       # green (stayed / safe)
WARN = "#FB8C00"       # amber (watch)
BLUE = "#4C6FFF"
GREY = "#9AA0AE"
SOFT = "#F5F6FA"       # card / control background
LINE = "#EEF0F4"       # gridlines
FONT = "Manrope"
CORAL_SCALE = ["#FFF1F0", "#FFC6C3", "#FF8A85", "#FF5A5F", "#D8323C"]
SEQ = [BRAND, WARN, GOOD, BLUE, "#9B5DE5", GREY]

st.set_page_config(
    page_title="Subscriber retention",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Global styling — gives the app a modern fintech feel rather than the default
# Streamlit look.
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, button, input, textarea {{
        font-family: '{FONT}', system-ui, sans-serif;
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}

    /* Hero balance band */
    .hero {{
        background: linear-gradient(135deg, {BRAND} 0%, {BRAND2} 100%);
        border-radius: 24px; padding: 26px 30px 22px 30px; color: #fff;
        box-shadow: 0 18px 40px rgba(255,90,95,.28);
    }}
    .hero .brand {{ font-size: 14px; font-weight: 700; opacity: .92;
        display:flex; align-items:center; gap:8px; letter-spacing:.2px; }}
    .hero .dot {{ width:9px; height:9px; border-radius:50%; background:#fff; display:inline-block; }}
    .hero .label {{ font-size: 14px; opacity:.9; margin-top:18px; font-weight:600; }}
    .hero .value {{ font-size: 46px; font-weight: 800; line-height:1.05; margin-top:2px;
        letter-spacing:-1px; }}
    .hero .sub {{ font-size: 15px; opacity:.95; margin-top:6px; max-width:640px; }}
    .chips {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }}
    .chip {{ background: rgba(255,255,255,.18); backdrop-filter: blur(4px);
        border-radius: 12px; padding: 9px 14px; font-size: 13px; }}
    .chip b {{ font-size:17px; font-weight:800; display:block; }}

    /* Soft white stat cards */
    .card {{ background:#fff; border-radius:18px; padding:18px 20px;
        box-shadow: 0 1px 3px rgba(20,22,30,.06), 0 10px 28px rgba(20,22,30,.05);
        border:1px solid #F0F1F5; }}
    .card .k {{ font-size:13px; color:{BODY}; font-weight:600; }}
    .card .v {{ font-size:30px; color:{INK}; font-weight:800; letter-spacing:-.5px; }}
    .card .s {{ font-size:12.5px; color:{GREY}; }}

    /* Friendly callout */
    .callout {{ border-radius:16px; padding:15px 18px; margin:6px 0 20px 0;
        font-size:15px; line-height:1.6; color:#3a3f4d; }}

    /* Section heading */
    .sec {{ margin: 26px 0 4px 0; }}
    .sec h3 {{ font-size:20px; font-weight:800; color:{INK}; margin:0; }}
    .sec p {{ font-size:14px; color:{BODY}; margin:3px 0 0 0; }}

    /* Pill-style tabs (segmented control) */
    .stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{SOFT};
        padding:6px; border-radius:14px; }}
    .stTabs [data-baseweb="tab"] {{ height:auto; padding:9px 20px; border-radius:10px;
        font-weight:600; color:{BODY}; background:transparent; }}
    .stTabs [aria-selected="true"] {{ background:#fff; color:{INK};
        box-shadow:0 1px 3px rgba(0,0,0,.10); }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def usd_zwl(usd: float, dp: int = 0) -> str:
    return f"${usd:,.{dp}f} (≈ZiG {usd * USD_TO_ZWL:,.{dp}f})"


def callout(text: str, tone: str = "brand") -> None:
    bg = {"brand": "#FFF3F2", "good": "#E9FBF3", "warn": "#FFF6E9",
          "neutral": SOFT}[tone]
    bar = {"brand": BRAND, "good": GOOD, "warn": WARN, "neutral": GREY}[tone]
    st.markdown(
        f'<div class="callout" style="background:{bg};'
        f'border-left:4px solid {bar};">{text}</div>',
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
# Hero band — the "balance" the business is about to lose
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero">
      <div class="brand"><span class="dot"></span> Subscriber retention &middot; Zimbabwe ISP</div>
      <div class="label">Revenue at risk over the next year</div>
      <div class="value">${annualised:,.0f}</div>
      <div class="sub">≈ ZiG {annualised * USD_TO_ZWL:,.0f} — that's <b>{pct_risk:.0f}%</b>
        of your base quietly leaning towards the door. Here's who, why, and who to call first.</div>
      <div class="chips">
        <span class="chip">subscribers <b>{n_total:,}</b></span>
        <span class="chip">likely to leave <b>{n_at_risk:,}</b></span>
        <span class="chip">model accuracy <b>{test_auc*100:.0f}%</b></span>
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

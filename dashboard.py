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

# A calm, professional palette — not the alarm-red template look.
INK = "#1f2933"
MUTED = "#6b7280"
ACCENT = "#b4452f"   # warm brick, used sparingly
PLOT_TEMPLATE = "plotly_white"

st.set_page_config(
    page_title="Subscriber retention",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def usd_zwl(usd: float, dp: int = 0) -> str:
    return f"${usd:,.{dp}f} (≈ZiG {usd * USD_TO_ZWL:,.{dp}f})"


def note(text: str) -> None:
    """A quiet analyst's note — sentence-case, no shouting, woven into the page."""
    st.markdown(
        f'<div style="border-left:3px solid #d7dbe0; background:#f7f8fa; '
        f'padding:11px 16px; margin:4px 0 22px 0; color:#3a434d; '
        f'font-size:15px; line-height:1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=340):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=INK, size=13),
    )
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

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div style="margin:-6px 0 6px 0;">
      <div style="font-size:13px; letter-spacing:.8px; color:{MUTED};
                  text-transform:uppercase;">Zimbabwe ISP &middot; retention</div>
      <h1 style="margin:2px 0 4px 0; font-size:30px; font-weight:700; color:{INK};">
        Who's likely to leave this cycle</h1>
      <div style="font-size:16px; color:{MUTED};">
        A retention view for a Zimbabwean internet provider — built on contract,
        payment, network and load-shedding signals.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

n_total = len(df)
n_at_risk = int((df["churn_prob"] >= 0.5).sum())
revenue_at_risk = float(df.loc[df["churn_prob"] >= 0.5, "MonthlyCharges"].sum())
annualised = revenue_at_risk * 12
top100_revenue = float(df.nlargest(100, "churn_prob")["MonthlyCharges"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Subscribers", f"{n_total:,}")
c2.metric("Likely to churn", f"{n_at_risk:,}", f"{n_at_risk/n_total*100:.0f}% of base",
          delta_color="off")
c3.metric("Revenue at risk / year", f"${annualised:,.0f}",
          f"≈ZiG {annualised * USD_TO_ZWL:,.0f}", delta_color="off")
c4.metric("Model AUC", f"{test_auc:.3f}")

note(
    f"If the team saved just the 100 highest-risk subscribers, it would hold on to "
    f"about <b>${top100_revenue*12:,.0f}</b> a year (≈ZiG {top100_revenue*12*USD_TO_ZWL:,.0f}). "
    f"Winning a new customer here costs roughly 5–7× what a retention offer does, "
    f"so even a small EcoCash auto-pay discount pays for itself if it keeps a "
    f"fraction of them."
)

tab_why, tab_calls, tab_score = st.tabs([
    "Why customers leave",
    "Who to call first",
    "Score a customer",
])

# --------------------------------------------------------------------------- #
with tab_why:
    # --- Flow ---------------------------------------------------------------
    st.markdown("#### From contract to outcome")
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
                col.append("rgba(180,69,47,0.16)" if c == "Month-to-month"
                           else "rgba(99,110,123,0.14)")
    for b in buckets:
        for o, mask in [("Stayed", df["Churn"] == 0), ("Left", df["Churn"] == 1)]:
            n = int(((df["tenure_bucket"] == b) & mask).sum())
            if n:
                src.append(node_idx[b]); tgt.append(node_idx[o]); val.append(n)
                col.append("rgba(180,69,47,0.40)" if o == "Left"
                           else "rgba(99,110,123,0.22)")

    node_colors = (
        [ACCENT, "#c98a3a", "#4b9e7a"][:len(contracts)]
        + ["#8a929b"] * len(buckets)
        + ["#4b9e7a", ACCENT]
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
    note(
        f"Most of the people who leave are on month-to-month plans in their first "
        f"year — about <b>{mtm_churn*100:.0f}%</b> of them go. The same kind of "
        f"customer on a two-year contract leaves only <b>{two_year*100:.0f}%</b> of "
        f"the time. Getting people onto longer plans early is the single biggest lever."
    )

    # --- Contract × tenure heatmap -----------------------------------------
    st.markdown("#### Where the risk concentrates")
    pivot = df.pivot_table(
        index="tenure_bucket", columns="Contract", values="Churn",
        aggfunc="mean", observed=True,
    ).round(3)
    col_order = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[col_order]
    fig = px.imshow(
        pivot, color_continuous_scale="OrRd", aspect="auto",
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
    note(
        f"The hottest cell is new <b>{worst_contract}</b> subscribers in their "
        f"<b>{worst_bucket}</b> — {worst_rate*100:.0f}% of those {n_worst:,} people "
        f"leave. That's where a retention rand goes furthest."
    )

    # --- The Zimbabwe-specific signals -------------------------------------
    st.markdown("#### What's specific to Zimbabwe")
    st.caption(
        "A churn model trained on the US Telco dataset would miss these entirely: "
        "the network you're on, the power situation, and how you pay."
    )

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
            color_discrete_map={"Econet": ACCENT, "NetOne": "#c98a3a", "Telecel": "#8a929b"},
            title="By mobile network",
        )
        fig.update_layout(xaxis_tickformat=".0%", showlegend=False)
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
            color="churn_rate", color_continuous_scale="OrRd",
            title="By province",
        )
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    worst_mno = mno_view.iloc[-1]
    note(
        f"<b>{worst_mno['MNO']}</b> subscribers leave most often "
        f"({worst_mno['churn_rate']*100:.0f}%), which tracks the real network-quality "
        f"gap — Econet has the widest 4G reach, NetOne is closing in on rural coverage, "
        f"and Telecel keeps losing ground. Churn is also highest where there's the most "
        f"competition: in the cities you can switch provider on the walk home, in rural "
        f"areas there's often nowhere else to go."
    )

    # --- Load-shedding ------------------------------------------------------
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
        color="churn_rate", color_continuous_scale="OrRd",
        labels={"load_shed_bucket": "Daily load-shedding", "churn_rate": "Churn rate"},
        title="Power outages and churn",
    )
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    low = ls_view.iloc[0]["churn_rate"]
    high = ls_view.iloc[-1]["churn_rate"]
    note(
        f"People living with 12+ hours of load-shedding a day leave at "
        f"<b>{high*100:.0f}%</b>, against <b>{low*100:.0f}%</b> for those barely "
        f"affected — roughly {high/low:.1f}× more. When the router's off half the day, "
        f"the subscription feels like money wasted. A small power-bank or solar-router "
        f"promo tied to a one-year plan is worth testing in the worst-hit suburbs."
    )

    # --- Payment rail -------------------------------------------------------
    pay_view = (
        df.groupby("PaymentMethod")
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"),
             avg_monthly_usd=("MonthlyCharges", "mean")).reset_index()
        .sort_values("churn_rate")
    )
    fig = px.bar(
        pay_view, x="churn_rate", y="PaymentMethod", orientation="h",
        text=pay_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
        color="churn_rate", color_continuous_scale="OrRd",
        title="How people pay, and whether they stay",
    )
    fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    note(
        "Cash-deposit payers leave most — there's no standing arrangement keeping "
        "them around. EcoCash and bank debit orders sit at the other end: once "
        "someone's on auto-pay, leaving takes effort. Nudging cash payers onto "
        "EcoCash or InnBucks auto-pay, with a small ZiG discount as the carrot, "
        "is a cheap retention play."
    )

# --------------------------------------------------------------------------- #
with tab_calls:
    st.markdown("#### The 50 subscribers to phone this week")
    st.caption("Ranked by the model's churn probability, highest first.")

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
    st.dataframe(
        watchlist.style.format({"Monthly $": "${:.2f}", "Churn risk": "{:.0%}"})
                 .background_gradient(subset=["Churn risk"], cmap="OrRd"),
        use_container_width=True, hide_index=True, height=500,
    )
    top_revenue = watchlist["Monthly $"].sum()
    note(
        f"These 50 people are worth <b>${top_revenue:,.0f}</b> a month "
        f"(≈ZiG {top_revenue*USD_TO_ZWL:,.0f}) between them. The model thinks most are "
        f"on their way out — a call this week is far cheaper than winning them back later."
    )

# --------------------------------------------------------------------------- #
with tab_score:
    st.markdown("#### Try a subscriber profile")
    st.caption("Change the details and watch the predicted risk move.")

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
    band_color = {"high": ACCENT, "watch": "#c98a3a", "stable": "#4b9e7a"}[band]

    left, right = st.columns([1, 1.2])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 52, "color": band_color}},
            delta={"reference": portfolio_avg * 100,
                   "increasing": {"color": ACCENT}, "decreasing": {"color": "#4b9e7a"},
                   "suffix": " pts vs avg"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": band_color},
                "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#eaf5ef"},
                    {"range": [30, 60], "color": "#fbf1e2"},
                    {"range": [60, 100], "color": "#f7e4de"},
                ],
                "threshold": {"line": {"color": INK, "width": 3},
                              "thickness": 0.75, "value": portfolio_avg * 100},
            },
        ))
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        st.caption(f"Dashed line is the average across all {n_total:,} subscribers.")

    with right:
        similar = df[(df["Contract"] == contract) &
                     (df["tenure_bucket"] == pd.cut([tenure], [-1, 6, 12, 24, 48, 72],
                      labels=["0-6m", "7-12m", "1-2y", "2-4y", "4y+"])[0])]
        if len(similar) > 0:
            sim_churn = similar["Churn"].mean()
            st.markdown(
                f"This profile looks like **{len(similar):,}** subscribers we already "
                f"have. Of those, **{sim_churn*100:.0f}%** actually left. The model puts "
                f"this particular person at **{prob*100:.0f}%**."
            )
        discount_usd = monthly * 1.5
        if band == "high":
            st.markdown(
                f"**Worth a call.** A save offer of up to ${discount_usd:.0f} "
                f"(≈ZiG {discount_usd*USD_TO_ZWL:,.0f}) on a longer contract still beats "
                f"replacing them. If they're on cash, move them to EcoCash auto-pay; if "
                f"load-shedding is biting ({load_shed:.0f}h/day), throw in a router-backup "
                f"promo. Flag them in the CRM."
            )
        elif band == "watch":
            st.markdown(
                f"**Keep an eye on them.** Touch base in about a month, send a quick "
                f"EcoCash survey, and if they're stuck on a '{bundle}' plan that doesn't "
                f"fit, suggest a monthly bundle that does."
            )
        else:
            st.markdown(
                "**Comfortable for now.** No save offer needed — this is the kind of "
                "customer to cross-sell to, ask for a referral, or invite to review you."
            )

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


def usd_zwl(usd: float, dp: int = 0) -> str:
    """Format an amount as '$X (≈ZiG Y)'."""
    return f"${usd:,.{dp}f}  (≈ZiG {usd * USD_TO_ZWL:,.{dp}f})"

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    /* Hide Streamlit chrome that distracts from the story */
    #MainMenu, footer {visibility: hidden;}

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #E74C3C 0%, #7F0F0F 100%);
        padding: 36px 32px;
        border-radius: 16px;
        color: white;
        margin: -10px 0 24px 0;
        box-shadow: 0 12px 32px rgba(231, 76, 60, 0.25);
    }
    .hero h1 { margin: 0; font-size: 40px; font-weight: 800; letter-spacing: -0.8px; }
    .hero p  { margin: 8px 0 0 0; font-size: 17px; opacity: 0.92; }

    /* Big stat cards */
    .stat {
        background: white;
        padding: 22px 24px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-top: 4px solid #E74C3C;
        height: 100%;
    }
    .stat .label { font-size: 12px; color: #7F8C8D; text-transform: uppercase; letter-spacing: 1px; }
    .stat .value { font-size: 32px; font-weight: 800; color: #2C3E50; margin: 4px 0; }
    .stat .sub   { font-size: 13px; color: #95A5A6; }

    /* Insight callout */
    .insight {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF5F5 100%);
        border-left: 4px solid #E74C3C;
        padding: 18px 22px;
        border-radius: 10px;
        margin: 8px 0;
    }
    .insight .head { font-size: 12px; color: #E74C3C; font-weight: 700; letter-spacing: 1px; }
    .insight .body { font-size: 16px; color: #2C3E50; margin-top: 4px; line-height: 1.5; }

    /* Tabs */
    div[data-testid="stTabs"] button[data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def big_stat(label: str, value: str, sub: str = "") -> str:
    return f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def insight(head: str, body: str) -> str:
    return f'<div class="insight"><div class="head">{head}</div><div class="body">{body}</div></div>'


# --------------------------------------------------------------------------- #
# Data + model loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading customers...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        # Auto-generate on first run so the demo is one-click.
        write_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "1-2y", "2-4y", "4y+"],
    )
    return df


@st.cache_resource(show_spinner="Training XGBoost...")
def train_and_score(df: pd.DataFrame):
    # ZWL columns are deterministic from USD so we exclude them to avoid leakage.
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

    # Score every customer for the watchlist + cohort views
    all_probs = model.predict_proba(pre.transform(X))[:, 1]
    return pre, model, test_auc, all_probs


df = load_data()
pre, model, test_auc, all_probs = train_and_score(df)
df = df.assign(churn_prob=all_probs)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>Churn Intelligence</h1>
      <p>Spot the customers about to walk away &mdash; and the contract levers that put them there.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top-line dramatic stats
n_total = len(df)
n_at_risk = int((df["churn_prob"] >= 0.5).sum())
revenue_at_risk = float(df.loc[df["churn_prob"] >= 0.5, "MonthlyCharges"].sum())
annualised = revenue_at_risk * 12
top100_revenue = float(df.nlargest(100, "churn_prob")["MonthlyCharges"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.markdown(big_stat("Customer base", f"{n_total:,}"), unsafe_allow_html=True)
c2.markdown(big_stat("At risk right now", f"{n_at_risk:,}",
                      f"{n_at_risk/n_total*100:.1f}% of the book"),
            unsafe_allow_html=True)
c3.markdown(big_stat("Annual revenue at risk", f"${annualised:,.0f}",
                      f"ZiG {annualised * USD_TO_ZWL:,.0f}  ·  ${revenue_at_risk:,.0f}/mo"),
            unsafe_allow_html=True)
c4.markdown(big_stat("Model ROC-AUC", f"{test_auc:.3f}",
                      "Higher is better (max 1.000)"),
            unsafe_allow_html=True)

st.markdown(
    insight(
        "WHY THIS MATTERS",
        f"Saving just the <b>top 100 highest-risk customers</b> would protect "
        f"<b>${top100_revenue:,.0f}/month</b> "
        f"(≈ZiG {top100_revenue * USD_TO_ZWL:,.0f}) in recurring revenue &mdash; that's "
        f"<b>${top100_revenue*12:,.0f}/year</b>. "
        f"In a Zim ISP context where new-customer acquisition costs roughly "
        f"5&ndash;7&times; a retention offer, the maths is brutal: an EcoCash "
        f"auto-pay $3 discount pays for itself if it saves "
        f"<b>{(300/top100_revenue)*100:.1f}%</b> of these customers.",
    ),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
tab_story, tab_zone, tab_zim, tab_watch, tab_predict = st.tabs([
    ":books: The Churn Story",
    ":fire: Danger Zones",
    ":zimbabwe: Zim Market Context",
    ":dart: At-Risk Watchlist",
    ":magic_wand: Score a Customer",
])

# --------------------------------------------------------------------------- #
with tab_story:
    st.subheader("How customers flow into churn")
    st.caption("Sankey diagram tracing the path from contract type → tenure bucket → outcome.")

    # Build sankey nodes
    contracts = list(df["Contract"].unique())
    buckets   = list(df["tenure_bucket"].cat.categories)
    outcomes  = ["Stayed", "Churned"]
    nodes = contracts + buckets + outcomes
    node_idx = {n: i for i, n in enumerate(nodes)}

    src, tgt, val, col = [], [], [], []
    # Contract -> bucket
    for c in contracts:
        for b in buckets:
            n = int(((df["Contract"] == c) & (df["tenure_bucket"] == b)).sum())
            if n:
                src.append(node_idx[c]); tgt.append(node_idx[b]); val.append(n)
                col.append("rgba(231,76,60,0.18)" if c == "Month-to-month" else "rgba(46,134,193,0.18)")
    # Bucket -> outcome
    for b in buckets:
        for o, mask in [("Stayed", df["Churn"] == 0), ("Churned", df["Churn"] == 1)]:
            n = int(((df["tenure_bucket"] == b) & mask).sum())
            if n:
                src.append(node_idx[b]); tgt.append(node_idx[o]); val.append(n)
                col.append("rgba(231,76,60,0.45)" if o == "Churned" else "rgba(46,134,193,0.30)")

    node_colors = (
        ["#E74C3C", "#F39C12", "#27AE60"][:len(contracts)]
        + ["#34495E"] * len(buckets)
        + ["#2E86C1", "#E74C3C"]
    )
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=18, thickness=18,
                  line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=col),
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), font_size=13)
    st.plotly_chart(fig, use_container_width=True)

    # Compute concrete insight
    mtm_short = df[(df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)]
    mtm_churn = mtm_short["Churn"].mean()
    two_year  = df[df["Contract"] == "Two year"]["Churn"].mean()
    st.markdown(
        insight(
            "THE STORY IN ONE LINE",
            f"<b>{mtm_churn*100:.0f}%</b> of month-to-month customers churn in their first year. "
            f"Lock the same customer into a two-year contract and that drops to "
            f"<b>{two_year*100:.0f}%</b> &mdash; a {(mtm_churn/two_year):.1f}× reduction.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_zone:
    st.subheader("Where the risk lives")
    st.caption("Churn rate heatmap. Red cells = the danger zone.")

    pivot = df.pivot_table(
        index="tenure_bucket", columns="Contract", values="Churn", aggfunc="mean", observed=True
    ).round(3)
    # Sort columns by mean churn rate descending so high-risk contracts come first
    col_order = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[col_order]

    fig = px.imshow(
        pivot, color_continuous_scale="Reds", aspect="auto",
        text_auto=".0%", origin="lower",
        labels=dict(color="Churn rate"),
    )
    fig.update_layout(height=420, coloraxis_colorbar=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    # Worst cell
    worst_idx = np.unravel_index(np.argmax(pivot.values), pivot.shape)
    worst_rate = pivot.values[worst_idx]
    worst_bucket = pivot.index[worst_idx[0]]
    worst_contract = pivot.columns[worst_idx[1]]
    n_worst = int(((df["tenure_bucket"] == worst_bucket) & (df["Contract"] == worst_contract)).sum())

    st.markdown(
        insight(
            "DANGER ZONE",
            f"<b>{worst_bucket}</b> tenure &times; <b>{worst_contract}</b> contract = "
            f"<b>{worst_rate*100:.0f}%</b> churn rate across {n_worst:,} customers. "
            f"This is where retention spend pays back the fastest.",
        ),
        unsafe_allow_html=True,
    )

    # Secondary chart: payment method effect
    st.markdown("#### Payment method makes it worse")
    by_pay = (
        df.groupby("PaymentMethod")["Churn"].agg(["mean", "count"]).reset_index()
        .sort_values("mean", ascending=True)
    )
    by_pay.columns = ["PaymentMethod", "churn_rate", "customers"]
    fig = px.bar(
        by_pay, x="churn_rate", y="PaymentMethod", orientation="h",
        text=by_pay["churn_rate"].map(lambda x: f"{x*100:.1f}%"),
        color="churn_rate", color_continuous_scale="Reds",
        hover_data={"customers": True},
    )
    fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
with tab_zim:
    st.subheader("Why this is a Zim problem, not an IBM tutorial")
    st.caption(
        "The same churn model trained on US Telco data misses what actually "
        "drives customers out of a Zimbabwean ISP: power, payment rails, and "
        "where in the country you live."
    )

    z1, z2 = st.columns(2)

    # --- MNO churn rate -----------------------------------------------------
    with z1:
        st.markdown("#### Churn by mobile network operator")
        mno_view = (
            df.groupby("MNO").agg(
                customers=("Churn", "size"),
                churn_rate=("Churn", "mean"),
                monthly_usd=("MonthlyCharges", "mean"),
            ).reset_index().sort_values("churn_rate", ascending=True)
        )
        fig = px.bar(
            mno_view, x="churn_rate", y="MNO", orientation="h",
            text=mno_view["churn_rate"].map(lambda x: f"{x*100:.1f}%"),
            color="MNO",
            color_discrete_map={"Econet": "#E74C3C", "NetOne": "#F39C12", "Telecel": "#7F8C8D"},
            hover_data={"customers": True, "monthly_usd": ":.2f"},
        )
        fig.update_layout(
            xaxis_tickformat=".0%", showlegend=False, height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        worst_mno = mno_view.iloc[-1]
        st.markdown(
            insight(
                "NETWORK QUALITY MATTERS",
                f"<b>{worst_mno['MNO']}</b> customers churn at "
                f"<b>{worst_mno['churn_rate']*100:.1f}%</b> &mdash; the highest in the book. "
                "In market this lines up with the network-quality reputation gap: "
                "Econet has the broadest 4G coverage, NetOne is catching up on rural "
                "rollout, and Telecel keeps shrinking. Cross-sell EcoCash auto-pay or "
                "move them to a longer contract before they port out.",
            ),
            unsafe_allow_html=True,
        )

    # --- Province churn ----------------------------------------------------
    with z2:
        st.markdown("#### Churn by province")
        prov_view = (
            df.groupby("Province").agg(
                customers=("Churn", "size"),
                churn_rate=("Churn", "mean"),
            ).reset_index().sort_values("churn_rate", ascending=True)
        )
        fig = px.bar(
            prov_view, x="churn_rate", y="Province", orientation="h",
            text=prov_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
            color="churn_rate", color_continuous_scale="Reds",
            hover_data={"customers": True},
        )
        fig.update_layout(
            xaxis_tickformat=".0%", coloraxis_showscale=False, height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        worst_prov = prov_view.iloc[-1]
        st.markdown(
            insight(
                "GEOGRAPHY IS DESTINY",
                f"<b>{worst_prov['Province']}</b> leads churn at "
                f"<b>{worst_prov['churn_rate']*100:.0f}%</b>. Mostly an urban-vs-rural "
                "story: in cities there are 3-4 alternative providers within walking "
                "distance, in rural areas customers stay because there's nowhere to go.",
            ),
            unsafe_allow_html=True,
        )

    # --- Load-shedding correlation -----------------------------------------
    st.markdown("#### Load-shedding is eating retention")
    df["load_shed_bucket"] = pd.cut(
        df["LoadSheddingHoursPerDay"],
        bins=[-0.1, 2, 5, 8, 12, 24],
        labels=["0-2h", "2-5h", "5-8h", "8-12h", "12h+"],
    )
    ls_view = (
        df.groupby("load_shed_bucket", observed=True).agg(
            churn_rate=("Churn", "mean"),
            customers=("Churn", "size"),
        ).reset_index()
    )
    fig = px.bar(
        ls_view, x="load_shed_bucket", y="churn_rate",
        text=ls_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
        color="churn_rate", color_continuous_scale="Reds",
        labels={"load_shed_bucket": "Average load-shedding (hours/day)", "churn_rate": "Churn rate"},
        hover_data={"customers": True},
    )
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

    low = ls_view.iloc[0]["churn_rate"]
    high = ls_view.iloc[-1]["churn_rate"]
    st.markdown(
        insight(
            "POWER OUTAGES ARE A CHURN MULTIPLIER",
            f"Customers in <b>12h+ daily load-shedding</b> churn at "
            f"<b>{high*100:.0f}%</b> vs <b>{low*100:.0f}%</b> for the <b>0-2h</b> "
            f"bucket &mdash; that's a <b>{(high/low):.1f}&times;</b> increase. "
            "Routers offline = service unused = subscription cancelled. "
            "Practical lever: bundle a small power-bank or solar router promotion "
            "with one-year contracts in Stage-2+ load-shedding zones.",
        ),
        unsafe_allow_html=True,
    )

    # --- Payment rail churn -------------------------------------------------
    st.markdown("#### Payment rail = commitment signal")
    pay_view = (
        df.groupby("PaymentMethod").agg(
            customers=("Churn", "size"),
            churn_rate=("Churn", "mean"),
            avg_monthly_usd=("MonthlyCharges", "mean"),
        ).reset_index().sort_values("churn_rate")
    )
    pay_view["avg_monthly_zwl"] = pay_view["avg_monthly_usd"] * USD_TO_ZWL
    fig = px.bar(
        pay_view, x="churn_rate", y="PaymentMethod", orientation="h",
        text=pay_view["churn_rate"].map(lambda x: f"{x*100:.0f}%"),
        color="churn_rate", color_continuous_scale="Reds",
        hover_data={
            "customers": True,
            "avg_monthly_usd": ":.2f",
            "avg_monthly_zwl": ":.0f",
        },
    )
    fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        insight(
            "MIGRATE CASH PAYERS TO ECOCASH",
            "Cash deposit payers churn at the highest rate &mdash; they are the "
            "least committed segment because there is no auto-pay friction "
            "stopping them from leaving. Bank debit order and EcoCash sit at the "
            "other end: once a customer sets up auto-pay, churn drops sharply. "
            "Run an InnBucks / EcoCash auto-pay migration campaign with a small "
            "ZiG-denominated discount as the carrot.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_watch:
    st.subheader("Top 50 customers most likely to churn")
    st.caption("Sorted by model probability. These are the people the retention team should phone first.")

    watchlist = (
        df.assign(monthly_revenue=df["MonthlyCharges"])
          .nlargest(50, "churn_prob")[
              ["Contract", "tenure", "MonthlyCharges", "PaymentMethod",
               "InternetService", "churn_prob"]
          ]
          .rename(columns={
              "Contract": "Contract",
              "tenure": "Tenure (mo)",
              "MonthlyCharges": "Monthly $",
              "PaymentMethod": "Pays via",
              "InternetService": "Internet",
              "churn_prob": "Churn prob.",
          })
          .reset_index(drop=True)
    )

    st.dataframe(
        watchlist.style.format({"Monthly $": "${:.2f}", "Churn prob.": "{:.0%}"})
                       .background_gradient(subset=["Churn prob."], cmap="Reds"),
        use_container_width=True, hide_index=True, height=500,
    )

    top_revenue = watchlist["Monthly $"].sum()
    st.markdown(
        insight(
            "ACT FIRST ON THESE 50",
            f"These 50 customers carry <b>${top_revenue:,.0f}/month</b> "
            f"(<b>${top_revenue*12:,.0f}/year</b>) of recurring revenue. "
            f"The model says they'll likely leave within the cycle &mdash; "
            f"a phone call today is cheaper than a full re-acquisition tomorrow.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Score a single customer")
    st.caption("Adjust the inputs to see what the model predicts, and how this customer compares.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 4)
        monthly = st.slider("Monthly charges (USD)", 8.0, 130.0, 55.0, step=0.5)
        total = st.slider("Total charges (USD)", 0.0, 10000.0, float(monthly * tenure), step=10.0)
        data_gb = st.slider("Monthly data usage (GB)", 0.1, 250.0, 20.0, step=0.5)
        st.caption(f"≈ ZiG {monthly * USD_TO_ZWL:,.0f} per month at the current rate.")
    with col2:
        mno = st.selectbox("Mobile network operator", MNOS)
        internet = st.selectbox("Internet service", ISPS)
        bundle = st.selectbox("Bundle preference", BUNDLES)
        contract = st.selectbox("Contract", CONTRACTS)
        payment = st.selectbox("Payment method", PAYMENT_METHODS)
    with col3:
        province = st.selectbox("Province", PROVINCES)
        location_type = st.selectbox("Area type", LOCATION_TYPES)
        load_shed = st.slider("Load-shedding (hours/day)", 0.0, 18.0, 8.0, step=0.5)
        support = st.slider("Support calls (last 90d)", 0, 20, 1)
        partner = st.radio("Has partner?", ["Yes", "No"], horizontal=True)
        dependents = st.radio("Has dependents?", ["Yes", "No"], horizontal=True)
        senior = st.radio(
            "Senior citizen?", [0, 1], horizontal=True,
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    record = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Province": province,
        "LocationType": location_type,
        "tenure": tenure,
        "MNO": mno,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "BundlePreference": bundle,
        "DataUsageGB": data_gb,
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "LoadSheddingHoursPerDay": load_shed,
        "SupportCalls90d": support,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    prob = float(model.predict_proba(pre.transform(record))[0, 1])
    portfolio_avg = float(df["churn_prob"].mean())
    band = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.3 else "LOW")
    band_color = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}[band]

    # Big animated gauge
    left, right = st.columns([1, 1.2])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 56, "color": band_color}},
            delta={
                "reference": portfolio_avg * 100,
                "increasing": {"color": "#E74C3C"},
                "decreasing": {"color": "#27AE60"},
                "suffix": "pp vs avg",
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": band_color},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "#E8F8EF"},
                    {"range": [30, 60], "color": "#FEF5E7"},
                    {"range": [60, 100], "color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "#2C3E50", "width": 3},
                    "thickness": 0.75,
                    "value": portfolio_avg * 100,
                },
            },
            title={"text": f"<b>{band} RISK</b>", "font": {"size": 22, "color": band_color}},
        ))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(f"### What the model is seeing")
        # Compute similar-customer comparison
        similar = df[(df["Contract"] == contract) & (df["tenure_bucket"] == pd.cut([tenure], [-1, 6, 12, 24, 48, 72], labels=["0-6m","7-12m","1-2y","2-4y","4y+"])[0])]
        if len(similar) > 0:
            sim_churn = similar["Churn"].mean()
            st.markdown(
                f"- This customer's profile is similar to **{len(similar):,}** customers in the book.\n"
                f"- Of those, **{sim_churn*100:.0f}%** churned in real life.\n"
                f"- Model says this specific customer is at **{prob*100:.0f}%** risk."
            )
        discount_usd = monthly * 1.5
        discount_zwl = discount_usd * USD_TO_ZWL
        action_lines = {
            "HIGH": [
                "Phone call **today** with a tailored save offer.",
                f"Offer up to **${discount_usd:.0f}** (≈ZiG {discount_zwl:,.0f}) off "
                "a longer contract — still cheaper than re-acquisition.",
                "Migrate to **EcoCash auto-pay** with a small ZiG bonus — "
                "auto-pay customers churn far less.",
                f"If load-shedding hours are high ({load_shed:.0f}h/day), "
                "bundle a router-backup promo to remove the 'I can't use it anyway' excuse.",
                "Flag for the retention team in the CRM.",
            ],
            "MEDIUM": [
                "Add to the **monitoring list** — touch in 30 days.",
                "Send a short EcoCash-survey link to catch any unspoken issues.",
                f"Suggest a **bundle migration** away from '{bundle}' "
                "toward a monthly plan if usage justifies it.",
            ],
            "LOW": [
                "No retention action needed.",
                "Use the relationship to **cross-sell** EcoCash insurance, "
                "Liquid TV, or a value-added bundle.",
                "Ask for a referral or a Google review.",
            ],
        }[band]
        st.markdown("**Recommended action**")
        for line in action_lines:
            st.markdown(f"- {line}")

"""Realistic Zimbabwean telco churn dataset generator.

Generates a synthetic-but-plausible dataset that reflects the Zim
telecommunications market in 2025-2026. Market facts informing the schema:

Mobile network operators (MNOs)
* Econet Wireless ~65% share, dominant in urban centres, owns EcoCash
* NetOne ~28%, state-owned, owns OneMoney, stronger rural footprint
* Telecel ~7%, smallest, persistent network-quality complaints

Internet service providers (ISPs)
* TelOne — state-owned incumbent, both legacy ADSL and FTTH rollout, owns the
  national fibre backbone
* Liquid Home / ZOL Fibroniks — sister brands under Liquid Intelligent
  Technologies (Masiyiwa group), premium fibre, urban suburban
* Econet Mobile data — dominant mobile broadband everywhere
* Smaller players: Powertel, Aquiva Wireless, Africom, Utande, YoAfrica

Mobile money / payment rails
* EcoCash dominates (USD wallet + ZiG wallet)
* OneMoney (NetOne)
* InnBucks (Simbisa Brands) — grew quickly during the 2022-23 ZWL crisis
  because formal banks struggled to clear cash
* ZIPIT — instant interbank transfer used for bill payment
* Bank debit order — formal salaried customers
* Cash deposit — least committed, often informal-income customers

Currency (2024-26)
* The ZiG (Zimbabwe Gold) replaced the ZWL in April 2024
* Telco bills are USD-primary; ZiG shown as secondary, exchanged at a
  rolling rate (we use 1 USD = 27 ZiG as a 2026 baseline)

Bundle culture
* Daily data and Social/WhatsApp bundles dominate — WhatsApp is the primary
  comms app for most Zimbabweans
* Monthly bundles signal commitment; daily-only signals churn risk

Real-world churn drivers we encode
* Month-to-month contract (biggest)
* Cash-deposit payers (least committed)
* Heavy load-shedding hours (router uptime)
* Many support calls (frustration)
* Urban + premium fibre (more alternatives to switch to)
* Telecel network-quality reputation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# USD primary, ZWL (ZiG) secondary. Update this constant as the rate moves.
# 2026 baseline: ~1 USD = 27 ZiG.
USD_TO_ZWL = 27.0

# Catalogue values exposed so the dashboard can reuse them in dropdowns.
MNOS = ["Econet", "NetOne", "Telecel"]
ISPS = ["ZOL Fibroniks", "Liquid Home", "TelOne ADSL", "Econet Mobile", "None"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "EcoCash",
    "OneMoney",
    "InnBucks",
    "ZIPIT",
    "Bank debit order",
    "Cash deposit",
]
BUNDLES = ["Daily data", "Weekly data", "Monthly data", "Social bundle", "None"]
PROVINCES = [
    "Harare",
    "Bulawayo",
    "Manicaland",
    "Mashonaland West",
    "Mashonaland East",
    "Mashonaland Central",
    "Masvingo",
    "Matabeleland North",
    "Matabeleland South",
    "Midlands",
]
LOCATION_TYPES = ["Urban", "Peri-urban", "Rural"]


def generate(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    """Generate a Zimbabwean telco churn dataset with `n` rows."""
    rng = np.random.default_rng(seed)

    # --- Geographic / demographic ----------------------------------------
    province = rng.choice(
        PROVINCES, n,
        p=[0.30, 0.13, 0.10, 0.10, 0.09, 0.06, 0.08, 0.05, 0.04, 0.05],
    )
    # Urban share is much higher in Harare / Bulawayo.
    location_type = np.array([
        rng.choice(LOCATION_TYPES, p=[0.78, 0.17, 0.05]) if p in ("Harare", "Bulawayo")
        else rng.choice(LOCATION_TYPES, p=[0.32, 0.30, 0.38])
        for p in province
    ])

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.86, 0.14])
    partner = rng.choice(["Yes", "No"], n, p=[0.52, 0.48])
    dependents = rng.choice(["Yes", "No"], n, p=[0.62, 0.38])  # larger households

    # --- Tenure (months) -------------------------------------------------
    tenure = rng.integers(0, 73, n)

    # --- Mobile network operator ----------------------------------------
    # Econet dominates urban; NetOne stronger in some rural/provincial markets;
    # Telecel a smaller third.
    mno = np.array([
        rng.choice(MNOS, p=[0.66, 0.24, 0.10]) if loc == "Urban"
        else rng.choice(MNOS, p=[0.55, 0.34, 0.11])
        for loc in location_type
    ])

    # --- Phone / multi-line ---------------------------------------------
    phone = rng.choice(["Yes", "No"], n, p=[0.96, 0.04])
    multi = np.where(
        phone == "No", "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.40, 0.60]),
    )

    # --- Internet service -----------------------------------------------
    # Fibre and fixed wireless skew urban; mobile data skews rural.
    internet = np.array([
        rng.choice(ISPS, p=[0.34, 0.22, 0.22, 0.16, 0.06]) if loc == "Urban"
        else rng.choice(ISPS, p=[0.05, 0.08, 0.22, 0.50, 0.15])
        for loc in location_type
    ])

    contract = rng.choice(CONTRACTS, n, p=[0.62, 0.22, 0.16])
    paperless = rng.choice(["Yes", "No"], n, p=[0.58, 0.42])

    # --- Payment method --------------------------------------------------
    # EcoCash dominant overall; Bank debit order skews to urban formal income;
    # Cash deposit skews rural / informal.
    payment = np.array([
        rng.choice(PAYMENT_METHODS, p=[0.42, 0.14, 0.08, 0.10, 0.18, 0.08]) if loc == "Urban"
        else rng.choice(PAYMENT_METHODS, p=[0.45, 0.17, 0.07, 0.08, 0.05, 0.18])
        for loc in location_type
    ])

    bundle = rng.choice(BUNDLES, n, p=[0.30, 0.18, 0.22, 0.18, 0.12])

    # --- Load-shedding hours per day (last 30d average) -----------------
    # Urban areas were hit harder in 2024-25 stage-2 schedules; rural less
    # because they often have no grid reliance, or solar uptake is higher.
    load_shed_hours = np.where(
        location_type == "Urban",
        rng.normal(8, 3, n).clip(0, 18),
        rng.normal(5, 3, n).clip(0, 16),
    ).round(1)

    # --- Data usage GB / month ------------------------------------------
    data_usage_gb = np.where(
        internet == "None",
        rng.gamma(1.2, 0.6, n).clip(0.1, 5),                # mobile-only
        np.where(
            internet == "Econet Mobile",
            rng.gamma(2.5, 2.5, n).clip(0.5, 30),
            rng.gamma(3.0, 15.0, n).clip(2, 250),           # fixed broadband
        ),
    ).round(1)

    # --- Support calls in last 90 days ----------------------------------
    support_calls = rng.poisson(1.2 + 0.10 * load_shed_hours, n).clip(0, 20)

    # --- Pricing (USD) --------------------------------------------------
    base_price = {
        "ZOL Fibroniks": 55,
        "Liquid Home": 48,
        "TelOne ADSL": 32,
        "Econet Mobile": 22,
        "None": 10,
    }
    monthly_usd = np.array([
        round(rng.normal(base_price[s], 8), 2) for s in internet
    ]).clip(8, 130)

    total_usd = (monthly_usd * np.maximum(tenure, 1)
                 + rng.normal(0, 25, n)).clip(0, None).round(2)

    monthly_zwl = (monthly_usd * USD_TO_ZWL).round(2)
    total_zwl = (total_usd * USD_TO_ZWL).round(2)

    # --- Churn signal (logit) -------------------------------------------
    logit = (
        -1.5
        + 1.5 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        + 0.6 * (internet == "ZOL Fibroniks")
        + 0.3 * (internet == "Liquid Home")
        - 0.4 * (internet == "TelOne ADSL")
        - 0.02 * tenure
        + 0.020 * (monthly_usd - 38)
        + 0.55 * (payment == "Cash deposit")
        + 0.15 * (payment == "OneMoney")
        - 0.30 * (payment == "Bank debit order")
        - 0.15 * (payment == "EcoCash")  # autopay-friendly
        + 0.07 * load_shed_hours          # blackouts erode service value
        + 0.18 * support_calls            # frustrated = churn-ready
        + 0.30 * senior
        - 0.20 * (partner == "Yes")
        + 0.30 * ((location_type == "Rural") & (internet == "Econet Mobile"))
        - 0.25 * ((location_type == "Rural") & (internet == "TelOne ADSL"))
        + 0.15 * (mno == "Telecel")       # smaller network, network-quality complaints
        + 0.25 * (bundle == "Daily data") # casual users churn fast
        - 0.10 * (bundle == "Monthly data")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        # Demographic / geographic
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Province": province,
        "LocationType": location_type,
        # Service profile
        "tenure": tenure,
        "MNO": mno,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "BundlePreference": bundle,
        "DataUsageGB": data_usage_gb,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        # Operational pain signals
        "LoadSheddingHoursPerDay": load_shed_hours,
        "SupportCalls90d": support_calls,
        # Money (USD primary, ZWL/ZiG secondary)
        "MonthlyCharges": monthly_usd,
        "TotalCharges": total_usd,
        "MonthlyChargesZWL": monthly_zwl,
        "TotalChargesZWL": total_zwl,
        # Target
        "Churn": churn,
    })
    return df


def write_csv(path: Path | str = "data/churn_data.csv",
              n: int = 7000, seed: int = 42, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    df = generate(n=n, seed=seed)
    df.to_csv(path, index=False)
    return path


def format_usd_zwl(usd: float, rate: float = USD_TO_ZWL,
                   show_zwl: bool = True, dp: int = 0) -> str:
    """Format an amount as '$X (≈ZiG Y)' for dashboard display."""
    usd_part = f"${usd:,.{dp}f}"
    if not show_zwl:
        return usd_part
    return f"{usd_part}  (≈ZiG {usd * rate:,.{dp}f})"


if __name__ == "__main__":
    out = write_csv()
    df = pd.read_csv(out)
    print(f"Wrote {out} | shape={df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    print(f"MNO mix: {df['MNO'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Load-shed mean (urban): "
          f"{df[df['LocationType']=='Urban']['LoadSheddingHoursPerDay'].mean():.1f} h/day")

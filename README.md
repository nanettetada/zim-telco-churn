<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:E74C3C,100:7F0F0F&height=200&section=header&text=Zimbabwe%20Telco%20Churn&fontSize=48&fontColor=ffffff&fontAlignY=40&animation=fadeIn" />

<a href="https://github.com/nanettetada">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3500&pause=800&color=E74C3C&center=true&vCenter=true&width=620&lines=Spot+at-risk+customers+early;XGBoost+%2B+SMOTE+%2B+SHAP;76%25+recall+at+0.89+ROC-AUC" />
</a>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logoColor=white" />
</p>

<a href="https://huggingface.co/spaces/NanetteTada/telco-churn-zim"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Live%20Demo-FFD21E?style=for-the-badge" /></a>

</div>

---

## :dart: Why I built this

I wanted to do a churn project that actually reflected my market, not a copy-paste of the IBM US telco dataset. So I built one for a **Zimbabwean ISP**: customers pay with **EcoCash, OneMoney, ZIPIT, bank debit orders, or cash**, subscribe to **ZOL Fibroniks, Liquid Home, TelOne ADSL, or Econet Mobile**, and behave the way subscribers actually behave here.

Once I learned that keeping a customer is roughly **5–7× cheaper** than getting a new one, the churn problem stopped feeling like a textbook exercise. For a Zim ISP fighting for market share between ZOL, Liquid and TelOne, that maths is even sharper.

## :sparkles: At a glance

|  |  |
|---|---|
| **Problem** | Predict Zim ISP subscribers about to cancel |
| **Approach** | XGBoost on SMOTE-balanced features inside a leak-proof pipeline |
| **Results** | ROC-AUC **0.89**, recall **0.76**, precision **0.74** |
| **Top drivers** | Month-to-month contracts · short tenure · cash-deposit payers · ZOL Fibroniks subscribers |
| **Payment methods** | EcoCash · OneMoney · ZIPIT · Bank debit order · Cash deposit |
| **Providers** | ZOL Fibroniks · Liquid Home · TelOne ADSL · Econet Mobile |
| **Stack** | scikit-learn · XGBoost · imbalanced-learn · SHAP · Streamlit · Plotly |

## :wrench: How I approached it

1. **EDA** — distributions, churn rate by segment, correlations.
2. **Preprocessing** — categorical encoding, scaling, stratified train/test split so the class ratio stays intact.
3. **Class imbalance** — applied SMOTE on the *training set only*. Doing it before the split is a classic leakage trap I wanted to avoid.
4. **Modelling** — Logistic Regression as a baseline (always start simple), then Random Forest and XGBoost.
5. **Evaluation** — ROC-AUC, precision, recall, F1, confusion matrix.
6. **Explainability** — SHAP to figure out which features the model was actually leaning on.

## :bar_chart: Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.66 | 0.73 | 0.69 |
| Random Forest | 0.87 | 0.72 | 0.71 | 0.71 |
| **XGBoost** | **0.89** | **0.74** | **0.76** | **0.75** |

XGBoost catches about three out of four churners — enough lead time for retention offers to land. In Zim terms, that's the difference between proactively bundling an EcoCash auto-pay discount for an at-risk customer and waiting for them to port to Liquid.

## :computer: Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb   # generates the synthetic dataset
streamlit run dashboard.py                          # interactive dashboard
```

## :tv: Interactive dashboard

Three tabs:
- **Overview** — class balance, tenure-vs-churn distribution, top-line KPIs.
- **Drivers** — pick any categorical feature and see the churn rate per bucket. The biggest driver is auto-highlighted.
- **Predict** — interactive form to score a single customer with a low / medium / high risk recommendation.

> Click the **Open in Streamlit Cloud** badge at the top to deploy this dashboard publicly in 90 seconds.

## :rocket: What I'd do next

- Tune the classification threshold based on the real cost of a false positive vs a false negative — 0.5 is rarely the right cutoff.
- Try LightGBM and compare training time + ROC-AUC.
- Wrap the best model in a small FastAPI service so it can be called from a CRM.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:E74C3C,100:7F0F0F&height=100&section=footer" />

Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a> · <a href="mailto:maumbetadaishe@gmail.com">email</a>

</div>

# Deploy this dashboard

Get a public URL anyone can visit — recruiters, friends, your phone.

## Streamlit Community Cloud (free, ~3 min)

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **Sign in with GitHub** (use `nanettetada`).
2. Click **Create app** → choose **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `nanettetada/zim-telco-churn`
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
   - **App URL** (optional): `tadaishe-customer-churn` → becomes `tadaishe-customer-churn.streamlit.app`
4. Click **Deploy**. First build takes 2–3 minutes; subsequent pushes redeploy in seconds.
5. Once live, paste the URL into the README's "Live demo" badge:

   ```markdown
   [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tadaishe-customer-churn.streamlit.app)
   ```

## After deploy

- The dashboard auto-rebuilds on every push to `main`.
- Free tier: app sleeps after ~7 days of inactivity. First load wakes it (~30s).

## Alternative: Hugging Face Spaces

This project is already deployed at https://huggingface.co/spaces/NanetteTada/zim-telco-churn.

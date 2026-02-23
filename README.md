# Customer Churn Predictor

Streamlit web app that predicts the **probability a customer will churn** (leave the service).

## What the app does

- **Inputs**: customer demographics, tenure, subscribed services, contract type, payment method, and charges
- **Output**: churn probability and a simple **Low / Medium / High risk** label

## Where the features come from

The inputs match common **telecom/cable customer records**:
- **Customer profile**: gender, senior citizen, partner, dependents
- **Tenure**: how long the customer has been active (months)
- **Services**: phone/internet and add-ons (security, backup, tech support, streaming, etc.)
- **Billing**: contract, paperless billing, payment method, monthly/total charges

## Why churn is important

Churn directly impacts **revenue and growth**. Predicting churn helps teams take action early (support, offers, targeted retention) instead of reacting after customers leave.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Connect this repo to [Streamlit Community Cloud](https://share.streamlit.io) and set the main file to `app.py`.

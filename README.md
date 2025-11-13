# Shakti — Smart Fraud Intelligence (Streamlit UI)

**Tagline:** *Shakti — Predict. Prevent. Protect.*

This repo runs a Streamlit web app that demos **Shakti**, an ensemble fraud-detection system (BERT text model + XGBoost tabular model + logistic meta-stacker). The UI is single-file: `streamlit_app.py`. Use it to test transactions, see history, and manage reports.

---

## Quick TL;DR
- Train -> `python train_model_ensemble.py` (creates `model/` artifacts and `data/transactions.csv`)
- Run Streamlit UI -> `streamlit run streamlit_app.py`
- Models expected under `model/`:
  - `xgb_tabular.pkl`
  - `meta_logreg.pkl`
  - `bert_model/` and `bert_tokenizer/`  
  (If missing, app will try `model/fraud_model.pkl` as a fallback.)

---

## What’s included
- `streamlit_app.py` — main Streamlit UI (form, admin dashboard, history, reports).
- `train_model_ensemble.py` — trains synthetic dataset, XGBoost, fine-tunes DistilBERT, fits a logistic meta-model, saves artifacts to `model/`.
- `data/` — transaction dataset + runtime history + reports (created automatically if missing).
- `model/` — location for saved model artifacts (created after training).

---

## Repo layout (Streamlit-focused)

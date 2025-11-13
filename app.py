
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from functools import lru_cache

st.set_page_config(layout="wide", page_title="Shakti", initial_sidebar_state="expanded")


MODEL_DIR = "model"
XGB_PATH = os.path.join(MODEL_DIR, "xgb_tabular.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta_logreg.pkl")
BERT_MODEL_DIR = os.path.join(MODEL_DIR, "bert_model")
BERT_TOKENIZER_DIR = os.path.join(MODEL_DIR, "bert_tokenizer")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")
REPORTS_FILE = os.path.join(DATA_DIR, "reports.csv")


if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=['timestamp','amount','merchant_type','device_type','location','hour','xgb_proba','bert_proba','meta_proba','prediction','flags']).to_csv(HISTORY_FILE, index=False)
if not os.path.exists(TRANSACTIONS_FILE):
    sample = pd.DataFrame({
        'amount':[100, 5000, 2500],
        'merchant_type':['retail','ecommerce','bank'],
        'device_type':['mobile','desktop','mobile'],
        'location':['India','UAE','USA'],
        'hour':[10, 2, 18],
        'is_fraud':[0,1,0]
    })
    sample.to_csv(TRANSACTIONS_FILE, index=False)
if not os.path.exists(REPORTS_FILE):
    pd.DataFrame(columns=['timestamp','description','amount','merchant_type','location']).to_csv(REPORTS_FILE, index=False)


if hasattr(st, "experimental_singleton"):
    cache_decorator = st.experimental_singleton
elif hasattr(st, "cache_resource"):
    cache_decorator = st.cache_resource
elif hasattr(st, "cache"):
    
    def cache_decorator(func=None, **kwargs):
        if func is None:
            return lambda f: st.cache(allow_output_mutation=True, **kwargs)(f)
        return st.cache(allow_output_mutation=True, **kwargs)(func)
else:

    def cache_decorator(func=None):
        if func is None:
            return lambda f: lru_cache(maxsize=1)(f)
        return lru_cache(maxsize=1)(func)

@cache_decorator
def load_models():
  
    models = {"xgb": None, "meta": None, "bert_model": None, "bert_tokenizer": None, "legacy": None}
    
    try:
        if os.path.exists(XGB_PATH):
            models["xgb"] = joblib.load(XGB_PATH)
    except Exception as e:
        st.warning(f"XGBoost load error: {e}")
    try:
        if os.path.exists(META_PATH):
            models["meta"] = joblib.load(META_PATH)
    except Exception as e:
        st.warning(f"Meta model load error: {e}")
    
    try:
        if os.path.exists(BERT_MODEL_DIR) and os.path.exists(BERT_TOKENIZER_DIR):
            tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_DIR)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
            models["bert_tokenizer"] = tokenizer
            models["bert_model"] = bert_model
    except Exception as e:
        st.warning(f"BERT load error: {e}")
    
    try:
        if os.path.exists(LEGACY_MODEL_PATH):
            models["legacy"] = joblib.load(LEGACY_MODEL_PATH)
    except Exception as e:
        st.warning(f"Legacy model load error: {e}")
    return models

models = load_models()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if models.get("bert_model"):
    models["bert_model"].to(DEVICE)
    models["bert_model"].eval()


def to_text_row(row):
    return (
        f"amount: {row['amount']:.2f} "
        f"merchant: {row['merchant_type']} "
        f"device: {row['device_type']} "
        f"location: {row['location']} "
        f"hour: {int(row['hour'])}"
    )

def prepare_model_input(data_row):
   
    train_df = pd.read_csv(TRANSACTIONS_FILE)
    X_train = train_df.drop(columns=['is_fraud'], errors='ignore')
    X_train = pd.get_dummies(X_train, columns=['merchant_type','device_type','location'], drop_first=True)
    df = pd.DataFrame([data_row])
    df = pd.get_dummies(df, columns=['merchant_type','device_type','location'], drop_first=True)
   
    missing = set(X_train.columns) - set(df.columns)
    for c in missing:
        df[c] = 0
    
    try:
        df = df[X_train.columns]
    except Exception:
       
        pass
    return df

def bert_predict_proba(texts):
  
    if models.get("bert_model") is None or models.get("bert_tokenizer") is None:
        raise RuntimeError("BERT model/tokenizer not loaded")
    if isinstance(texts, str):
        texts = [texts]
    tokenizer = models["bert_tokenizer"]
    model = models["bert_model"]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu()
        probs = F.softmax(logits, dim=1).numpy()[:, 1]
    return probs

def xgb_predict_proba(df):
    xgb = models.get("xgb")
    if xgb is None:
        raise RuntimeError("XGBoost model not loaded")
   
    try:
        model_cols = None
        if hasattr(xgb, "get_booster"):
            try:
                model_cols = xgb.get_booster().feature_names
            except Exception:
                model_cols = None
        if model_cols is not None:
            
            for c in model_cols:
                if c not in df.columns:
                    df[c] = 0
            df = df[model_cols]
    except Exception:
        pass
    proba = xgb.predict_proba(df)[:, 1]
    return proba

def meta_predict_proba(xgb_p, bert_p):
    meta = models.get("meta")
    if meta is None:
        raise RuntimeError("Meta (stacker) model not loaded")
    arr = np.vstack([xgb_p, bert_p]).T
    return meta.predict_proba(arr)[:, 1]

def rule_flags(data):
    flags = []
    prediction_flag = 0
    if data['amount'] > 1000 and data['location'] not in ['India','USA','UK','UAE']:
        flags.append("High amount + unusual location")
        prediction_flag = 1
    if data['merchant_type'] in ['online','luxury'] and data['device_type']=='mobile' and data['hour']<6:
        flags.append("Suspicious merchant + mobile device + odd hour")
        prediction_flag = 1
    if data['merchant_type']=='upi' and data['amount']>5000:
        flags.append("High UPI amount")
        prediction_flag = 1
    return flags, prediction_flag

def write_history(row):
    hist = pd.read_csv(HISTORY_FILE)
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False)


st.title("Shakti")
st.markdown("The Power to Stop Fraud Before It Strikes")


role = st.sidebar.selectbox("Role", ["user", "admin"])
st.sidebar.markdown("---")
st.sidebar.write("Model status:")
st.sidebar.write(f"- XGBoost: {'loaded' if models.get('xgb') is not None else 'missing'}")
st.sidebar.write(f"- Meta stacker: {'loaded' if models.get('meta') is not None else 'missing'}")
st.sidebar.write(f"- BERT: {'loaded' if models.get('bert_model') is not None else 'missing'}")
st.sidebar.write(f"- Legacy single model: {'loaded' if models.get('legacy') is not None else 'missing'}")


with st.form("txn_form"):
    st.subheader("New transaction")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, step=1.0, value=100.0)
        merchant_type = st.selectbox("Merchant type", ['ecommerce','bank','retail','upi','luxury','online'])
    with col2:
        device_type = st.selectbox("Device type", ['mobile','desktop','pos','unknown'])
        location = st.selectbox("Location", ['India','USA','UK','UAE','International'])
    with col3:
        hour = st.slider("Hour (0-23)", 0, 23, 12)
        submit = st.form_submit_button("Run Detection")

if submit:
    
    input_data = {
        'amount': float(amount),
        'merchant_type': merchant_type,
        'device_type': device_type,
        'location': location,
        'hour': int(hour)
    }
    
    df_tab = prepare_model_input(input_data)

    
    xgb_proba = bert_proba = meta_proba = None
    fallback_pred = None

   
    try:
        xgb_proba = float(xgb_predict_proba(df_tab)[0]) if models.get("xgb") else None
    except Exception as e:
        st.error(f"XGBoost inference error: {e}")

    
    text = to_text_row(input_data)
    try:
        if models.get("bert_model") and models.get("bert_tokenizer"):
            bert_proba = float(bert_predict_proba(text)[0])
    except Exception as e:
        st.error(f"BERT inference error: {e}")

    
    try:
        if models.get("meta") and xgb_proba is not None and bert_proba is not None:
            meta_proba = float(meta_predict_proba(np.array([xgb_proba]), np.array([bert_proba]))[0])
    except Exception as e:
        st.error(f"Meta inference error: {e}")

    
    try:
        if meta_proba is None and models.get("legacy") is not None:
            
            legacy = models["legacy"]
            
            try:
                model_cols = None
                if hasattr(legacy, "get_booster"):
                    try:
                        model_cols = legacy.get_booster().feature_names
                    except Exception:
                        model_cols = None
                if model_cols is not None:
                    for c in model_cols:
                        if c not in df_tab.columns:
                            df_tab[c] = 0
                    df_tab = df_tab[model_cols]
            except Exception:
                pass
            fallback_pred = int(legacy.predict(df_tab)[0])
    except Exception as e:
        st.error(f"Legacy model inference error: {e}")

    
    flags, rule_flag_pred = rule_flags(input_data)

    
    final_pred = 0
    final_score = None
    if meta_proba is not None:
        final_score = meta_proba
        final_pred = int(meta_proba >= 0.5)
    elif xgb_proba is not None and bert_proba is not None and models.get("meta") is None:
        
        final_score = (xgb_proba + bert_proba) / 2.0
        final_pred = int(final_score >= 0.5)
    elif xgb_proba is not None:
        final_score = xgb_proba
        final_pred = int(xgb_proba >= 0.5)
    elif bert_proba is not None:
        final_score = bert_proba
        final_pred = int(bert_proba >= 0.5)
    elif fallback_pred is not None:
        final_pred = int(fallback_pred)
        final_score = float(final_pred)
    
    if rule_flag_pred == 1:
        final_pred = 1
        if final_score is None:
            final_score = 1.0

    
    st.subheader("Result")
    if final_pred == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success(" Legitimate Transaction")

    
    scores = {
        "XGBoost_proba": xgb_proba,
        "BERT_proba": bert_proba,
        "Meta_proba": meta_proba,
        "Final_score": final_score,
        "Final_pred": final_pred
    }
    st.write(pd.DataFrame([scores]))

    if flags:
        st.warning("Flags: " + ", ".join(flags))
    else:
        st.info("Flags: None")

    
    hist_row = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'amount': input_data['amount'],
        'merchant_type': input_data['merchant_type'],
        'device_type': input_data['device_type'],
        'location': input_data['location'],
        'hour': input_data['hour'],
        'xgb_proba': xgb_proba if xgb_proba is not None else "",
        'bert_proba': bert_proba if bert_proba is not None else "",
        'meta_proba': meta_proba if meta_proba is not None else "",
        'prediction': int(final_pred),
        'flags': ', '.join(flags) if flags else 'None'
    }
    write_history(hist_row)
    st.success("Saved to history.")


if role == "admin":
    st.markdown("---")
    st.header("Admin Dashboard")

    tx = pd.read_csv(TRANSACTIONS_FILE)
    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        fraud_count = int(tx['is_fraud'].sum()) if 'is_fraud' in tx.columns else 0
        st.metric("Fraud count (dataset)", fraud_count)
    with c2:
        legit_count = len(tx) - fraud_count if 'is_fraud' in tx.columns else len(tx)
        st.metric("Legit count (dataset)", legit_count)
    with c3:
        accuracy = round(85 + (fraud_count / max(1, len(tx))) * 5, 2) if 'is_fraud' in tx.columns else "N/A"
        st.metric("Simulated accuracy", accuracy)

    st.subheader("Merchant counts")
    if 'merchant_type' in tx.columns:
        st.bar_chart(tx['merchant_type'].value_counts())

    st.subheader("Fraud rate by hour")
    if 'hour' in tx.columns and 'is_fraud' in tx.columns:
        hr = tx.groupby('hour')['is_fraud'].mean().sort_index()
        st.line_chart(hr)

    st.subheader("Recent reports")
    reports = pd.read_csv(REPORTS_FILE)
    if not reports.empty:
        st.table(reports.tail(5))
    else:
        st.info("No reports yet. Use /data/reports.csv to add.")


st.markdown("---")
st.header("Recent Detection History")
hist_df = pd.read_csv(HISTORY_FILE)
if not hist_df.empty:
    st.dataframe(hist_df.sort_values("timestamp", ascending=False).head(50))
else:
    st.info("No history yet.")

st.markdown("Built by *Shaurya*")

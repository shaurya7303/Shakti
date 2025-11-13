
import os
import random
import numpy as np
import pandas as pd
import joblib
import torch
from datetime import datetime


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


n = 5000
data = pd.DataFrame({
    'amount': np.random.uniform(5, 10000, n),
    'merchant_type': np.random.choice(['ecommerce', 'bank', 'retail', 'upi', 'luxury'], n),
    'device_type': np.random.choice(['mobile', 'desktop', 'pos', 'unknown'], n),
    'location': np.random.choice(['India', 'USA', 'UK', 'UAE', 'International'], n),
    'hour': np.random.randint(0, 24, n)
})


data['is_fraud'] = (
    ((data['amount'] > 7000) & (data['merchant_type'].isin(['upi', 'ecommerce']))) |
    (data['hour'] < 5) | (data['hour'] > 22) |
    ((data['location'] == 'International') & (data['amount'] > 2000)) |
    ((data['merchant_type'] == 'luxury') & (data['device_type'] == 'mobile'))
).astype(int)


data['night_transaction'] = ((data['hour'] > 22) | (data['hour'] < 6)).astype(int)
data['high_amount'] = (data['amount'] > 5000).astype(int)

os.makedirs("data", exist_ok=True)
data.to_csv("data/transactions.csv", index=False)
print("Saved data/transactions.csv")


def to_text_row(row):
    return (
        f"amount: {row['amount']:.2f} "
        f"merchant: {row['merchant_type']} "
        f"device: {row['device_type']} "
        f"location: {row['location']} "
        f"hour: {int(row['hour'])}"
    )

data['text'] = data.apply(to_text_row, axis=1)


X_tab = data.drop(columns=['is_fraud', 'text'])
y = data['is_fraud']
X_tab = pd.get_dummies(X_tab, columns=['merchant_type', 'device_type', 'location'], drop_first=True)


X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X_tab, y, data['text'], test_size=0.2, random_state=SEED, stratify=y
)


X_tr, X_hold, y_tr, y_hold, text_tr, text_hold = train_test_split(
    X_train, y_train, text_train, test_size=0.15, random_state=SEED, stratify=y_train
)


xgb = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=4
)
xgb.fit(X_tr, y_tr)
print("Trained XGBoost.")


xgb_hold_proba = xgb.predict_proba(X_hold)[:, 1]
xgb_test_proba = xgb.predict_proba(X_test)[:, 1]


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def prepare_hf_dataset(texts, labels=None):
    d = {"text": list(texts)}
    if labels is not None:
        d["labels"] = list(labels)
    return Dataset.from_dict(d)

train_ds = prepare_hf_dataset(text_tr, y_tr)
hold_ds = prepare_hf_dataset(text_hold, y_hold)
test_ds = prepare_hf_dataset(text_test, y_test)


def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
hold_ds = hold_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)


train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
hold_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


import inspect
sig = inspect.signature(TrainingArguments.__init__)
params = sig.parameters

tg_kwargs = dict(
    output_dir="./bert_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    seed=SEED,
    fp16=torch.cuda.is_available()
)


if "evaluation_strategy" in params:
    tg_kwargs["evaluation_strategy"] = "no"
else:
    tg_kwargs["eval_strategy"] = "no"

training_args = TrainingArguments(**tg_kwargs)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_ds,
)


trainer.train()
print("Trained BERT classifier.")


import numpy as np

def get_probas_trainer(trainer_obj, dataset):
    preds = trainer_obj.predict(dataset)
    logits = preds.predictions  # may be shape (N,2) or (N,) or (N,1)
    if logits is None:
        raise RuntimeError("Trainer returned no predictions.")
    logits = np.array(logits)
    if logits.ndim == 1:
        # single logit (sigmoid)
        probs = 1.0 / (1.0 + np.exp(-logits))
    elif logits.ndim == 2 and logits.shape[1] == 1:
        probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    else:
        # softmax over last dim, take class 1
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp[:, 1] / np.sum(exp, axis=1)
    return probs

bert_hold_proba = get_probas_trainer(trainer, hold_ds)
bert_test_proba = get_probas_trainer(trainer, test_ds)


meta_X_hold = np.vstack([xgb_hold_proba, bert_hold_proba]).T
meta_y_hold = y_hold.values

meta_X_test = np.vstack([xgb_test_proba, bert_test_proba]).T
meta_y_test = y_test.values

meta_clf = LogisticRegression()
meta_clf.fit(meta_X_hold, meta_y_hold)
print("Trained meta-classifier (logistic regression).")


meta_test_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
meta_test_pred = (meta_test_proba >= 0.5).astype(int)

print("=== XGBoost on test ===")
xgb_test_pred = (xgb_test_proba >= 0.5).astype(int)
print("Accuracy:", accuracy_score(meta_y_test, xgb_test_pred))
print(classification_report(meta_y_test, xgb_test_pred))

print("=== BERT on test ===")
bert_test_pred = (bert_test_proba >= 0.5).astype(int)
print("Accuracy:", accuracy_score(meta_y_test, bert_test_pred))
print(classification_report(meta_y_test, bert_test_pred))

print("=== Stacked ensemble (meta) on test ===")
print("Accuracy:", accuracy_score(meta_y_test, meta_test_pred))
print(classification_report(meta_y_test, meta_test_pred))


os.makedirs("model", exist_ok=True)
joblib.dump(xgb, "model/xgb_tabular.pkl")
joblib.dump(meta_clf, "model/meta_logreg.pkl")
tokenizer.save_pretrained("model/bert_tokenizer")
trainer.model.save_pretrained("model/bert_model")

print("Saved models to model/ (xgb_tabular.pkl, meta_logreg.pkl, bert_model/, bert_tokenizer/)")

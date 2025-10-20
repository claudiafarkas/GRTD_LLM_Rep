import os, re, random
import pandas as pd
from sklearn.datasets import fetch_openml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch

# ---------- Global config ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
random.seed(42)

# Force CPU everywhere because we dont have access to GPUs
try:
    torch.set_default_device("cpu")   
except Exception:
    pass

DEVICE = torch.device("cpu")
LABEL = "income"         # rename class -> income
MODEL_NAME = "distilgpt2"
MAX_LEN = 256
BATCH_SIZE = 2
EPOCHS = 3
N_SAMPLES = 1000           # how many rows we want in our synthetic dataset
CSV_OUT = "data/synthetic_adult_predllm_mini.csv"

# ---------- Load 'Adult' dataset from OpenML ----------
print("Loading Adult…")
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.rename(columns={"class": LABEL})
df = df.dropna().reset_index(drop=True)

# Keep a smaller subset (computationally easier on our laptops)
df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

FEATURES = [c for c in df.columns if c != LABEL]

# STEP 1
# ---------- Serialization helpers: Fine-tune with Y last + Permute - learn p(Y｜X） ----------
def serialize_row(row: pd.Series, feature_order, label_col: str) -> str:
    parts = []
    for col in feature_order:
        parts.append(f"{col} is {str(row[col]).strip()}")
    parts.append(f"{label_col} is {str(row[label_col]).strip()}")
    return ", ".join(parts)

def build_training_corpus(df: pd.DataFrame, features, label_col, permute_ratio=0.7):
    """
    Build texts with:
      - original feature order (always)
      - permute-x augmentation (shuffle features ONLY; Y always last)
    """
    texts = []
    for _, r in df.iterrows():
        texts.append(serialize_row(r, features, label_col))
        if random.random() < permute_ratio:
            fperm = features[:]
            random.shuffle(fperm)
            texts.append(serialize_row(r, fperm, label_col))
    return texts

print("Building training texts…")
train_texts = build_training_corpus(df, FEATURES, LABEL, permute_ratio=0.7)

# ---------- Tokenization + training ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  

enc = tok(
    train_texts,
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)

class TxtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.enc = encodings
    def __len__(self):
        return self.enc["input_ids"].shape[0]
    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.enc["input_ids"][i],
        }

ds = TxtDataset(enc)

# Model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# Training
args = TrainingArguments(
    output_dir="runs/predllm-mini",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    logging_steps=50,
    save_steps=5000,
    save_total_limit=1,
    report_to=[],
    no_cuda=True,                 
    dataloader_pin_memory=False,  
    use_mps_device=False,         
)

trainer = Trainer(model=model, args=args, train_dataset=ds)

print("Fine-tuning (CPU)…")
trainer.train()

# Make sure the model is on CPU for generation
model = trainer.model
model.to("cpu")
model.eval()

# Save
os.makedirs("runs/predllm-mini/model", exist_ok=True)
trainer.save_model("runs/predllm-mini/model")
tok.save_pretrained("runs/predllm-mini/model")

# STEP 2
# ---------- Feature conditional Sampling phase: generate X ONLY ----------
print("\nSampling synthetic features (X) …")

def sample_feature_value(col):
    # simple marginal sampler from real data
    return random.choice(df[col].tolist())

def generate_features_only(start_feature, start_value, max_new_tokens=120):
    """
    Start with "feature is value" and let the model continue generating
    features. If we see the label field begin (`, income is`), we cut before it.
    Returns a partial dict of generated features.
    """
    start = f"{start_feature} is {start_value}"
    prompt = start
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # Stop at the beginning of label if present
    stop_idx = text.find(f", {LABEL} is")
    if stop_idx != -1:
        text = text[:stop_idx]

    # Parse comma separated "name is value" clauses
    clauses = [c.strip() for c in text.split(",")]
    seen = {}
    for c in clauses:
        m = re.match(r"^([^,]+?)\s+is\s+(.+)$", c)
        if not m:
            continue
        name = m.group(1).strip()
        val = m.group(2).strip()
        if name in FEATURES and name not in seen:
            seen[name] = val
        if len(seen) == len(FEATURES):
            break
    return seen  

synthetic_X = []
for _ in range(N_SAMPLES):
    f0 = random.choice(FEATURES)
    v0 = sample_feature_value(f0)
    feats = generate_features_only(f0, v0)
    # Fill any missing features from real marginals to keep schema complete
    for col in FEATURES:
        if col not in feats:
            feats[col] = sample_feature_value(col)
    synthetic_X.append(feats)

print(f"Generated {len(synthetic_X)} synthetic feature rows.")

# STEP 3
# ---------- Label-query phase: get Y (Patch 1: tighter parsing) ----------
print("\nQuerying labels (Y) …")

VALID_LABELS = {"<=50K", ">50K"}  

def label_query(feat_row: dict) -> str:
    # serialize features in ORIGINAL order, then let model continue to produce "income is <value>"
    parts = [f"{c} is {feat_row[c]}" for c in FEATURES]
    prompt = ", ".join(parts) + ", "
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
    **inputs,
    max_new_tokens=8,
    do_sample=False,       
    temperature=0.0,       
    top_p=1.0,
    pad_token_id=tok.eos_token_id,
    eos_token_id=tok.eos_token_id,
)

    text = tok.decode(out[0], skip_special_tokens=True)

    # Prefer exact known labels first
    for v in VALID_LABELS:
        if f"{LABEL} is {v}" in text:
            return v

    # Fallback regex (looser)
    m = re.search(rf"{LABEL}\s+is\s+([^\s,]+)", text)
    if m:
        cand = m.group(1).strip()
        # Minor normalization in case of punctuation
        cand = cand.replace(".", "")
        # Snap close variants to known labels
        for v in VALID_LABELS:
            if cand.lower() == v.lower():
                return v
    return None

synthetic_rows = []
for fx in synthetic_X:
    y = label_query(fx)
    if y is None:
        # Fallback: draw a label from real marginals if the model didn't output one cleanly
        y = random.choice(df[LABEL].tolist())
    synthetic_rows.append({**fx, LABEL: y})

syn_df = pd.DataFrame(synthetic_rows)

# Save synthetic CSV 
os.makedirs("data", exist_ok=True)
syn_df.to_csv(CSV_OUT, index=False)
print(f"\nSaved synthetic dataset to: {CSV_OUT}")

print("\nSample synthetic rows:")
print(syn_df.head(10).to_string(index=False))

# STEP 4
# ----------  TSTR: Train on Synthetic, Test on Real ----------
print("\nRunning TSTR (Train on Synthetic → Test on Real)…")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

real = df.copy()
real_y = real[LABEL]
real_X = real.drop(columns=[LABEL])

syn = syn_df.copy()
syn_y = syn[LABEL].fillna("<=50K")  
syn_X = syn.drop(columns=[LABEL])

# Identify categorical/numerical columns from REAL schema
cat_cols = [c for c in real_X.columns if real_X[c].dtype == 'object']
num_cols = [c for c in real_X.columns if c not in cat_cols]

# --- NEW: Coerce numeric columns to numeric in BOTH real and synthetic ---
for col in num_cols:
    real_X[col] = pd.to_numeric(real_X[col], errors="coerce")
    syn_X[col]  = pd.to_numeric(syn_X[col],  errors="coerce")

# Fill any NaNs created by coercion using REAL-data medians (schema anchor)
real_medians = real_X[num_cols].median(numeric_only=True)
real_X[num_cols] = real_X[num_cols].fillna(real_medians)
syn_X[num_cols]  = syn_X[num_cols].fillna(real_medians)

# (Optional but robust): ensure categoricals are strings for OneHotEncoder
for col in cat_cols:
    real_X[col] = real_X[col].astype(str)
    syn_X[col]  = syn_X[col].astype(str)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=4
)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# Encode labels to integers for XGBoost
label_map = {"<=50K": 0, ">50K": 1}
inv_label_map = {v: k for k, v in label_map.items()}

syn_y_enc = syn_y.map(label_map)
if syn_y_enc.isna().any():
    # If any weird labels slipped in, snap them to the majority class
    maj = syn_y_enc.value_counts().idxmax()
    syn_y_enc = syn_y_enc.fillna(maj).astype(int)
else:
    syn_y_enc = syn_y_enc.astype(int)

# Real test split
_, X_test_real, _, y_test_real = train_test_split(
    real_X, real_y, test_size=0.2, stratify=real_y, random_state=42
)
y_test_real_enc = y_test_real.map(label_map).astype(int)

# Train on synthetic (encoded), test on real (encoded)
pipe.fit(syn_X, syn_y_enc)
pred_enc = pipe.predict(X_test_real).astype(int)
acc = accuracy_score(y_test_real_enc, pred_enc)
print(f"TSTR Accuracy (XGB trained on synthetic, tested on real): {acc:.3f}")

# (Optional) show predicted label distribution in natural labels
pred_labels = pd.Series(pred_enc).map(inv_label_map).value_counts(normalize=True).rename("pred_label_share")
print("\nPredicted label share on real test set:")
print(pred_labels.to_string())
print("\nDone. You now have: fine-tuned LM, synthetic CSV, and a TSTR score.")


print("Majority baseline accuracy:", (y_test_real_enc.value_counts().max() / len(y_test_real_enc)))



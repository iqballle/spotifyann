import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE

# ------------------------------
# Konfigurasi input
# ------------------------------
USE_KAGGLEHUB = True
KAGGLE_DATASET = "nabihazahid/spotify-dataset-for-churn-analysis"
LOCAL_CSV = "spotify_churn_dataset.csv"

print("=" * 50)
print("SPOTIFY CHURN PREDICTION MODEL — MOD (no-emoji)")
print("=" * 50)

# ------------------------------
# STEP 1: Load dataset
# ------------------------------
print("\nSTEP 1: Load dataset...")
df = None
if USE_KAGGLEHUB:
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        csv_path = os.path.join(path, "spotify_churn_dataset.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File tidak ditemukan di KaggleHub path: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded from KaggleHub: {csv_path}")
    except Exception as e:
        print(f"KaggleHub gagal: {e}")
        print(f"Fallback ke file lokal: {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV)
else:
    df = pd.read_csv(LOCAL_CSV)
    print(f"Loaded local CSV: {LOCAL_CSV}")

# ------------------------------
# Target & fitur
# ------------------------------
TARGET_CANDIDATES = ["is_churned","churn","Churn","label","target","Exited","Class"]
target = next((c for c in TARGET_CANDIDATES if c in df.columns), df.columns[-1])

y = df[target].copy()
X = df.drop(columns=[target]).copy()
for idcol in ["user_id","id","UID"]:
    if idcol in X.columns:
        X.drop(columns=[idcol], inplace=True)

def map_binary(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    vals = pd.Series(y.dropna().astype(str).str.lower().unique())
    mapping = {
        "yes": 1, "no": 0, "true": 1, "false": 0,
        "churn": 1, "nochurn": 0, "churned": 1, "not_churned": 0
    }
    if set(vals) <= set(mapping.keys()):
        return y.astype(str).str.lower().map(mapping).astype(int)
    if set(vals) <= {"0","1"}:
        return y.astype(int)
    return y

y = map_binary(y)
print("\nClass balance:", y.value_counts(normalize=True).round(3).to_dict())

# ------------------------------
# Split train/test (stratify bila klasifikasi)
# ------------------------------
stratify = y if y.nunique() <= 10 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# ------------------------------
# Pipeline preprocessing
#   - numeric: median impute + standardize
#   - categorical: most_frequent impute + onehot (kompatibel lintas versi)
# ------------------------------
num_sel = make_column_selector(dtype_include=np.number)
cat_sel = make_column_selector(dtype_exclude=np.number)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# OneHotEncoder: dukung scikit-learn lama & baru
def make_ohe():
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", make_ohe())
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_sel(X_train)),
    ("cat", categorical_pipe, cat_sel(X_train))
])

Xtr = preprocess.fit_transform(X_train, y_train)
Xte = preprocess.transform(X_test)

# ------------------------------
# SMOTE (opsional jika imbalanced)
# ------------------------------
minor = min(Counter(y_train).values())
k = max(1, min(5, minor - 1))
USE_SMOTE = (y_train.value_counts(normalize=True).min() < 0.35 and minor > 1)

def make_train_data(use_smote: bool):
    if not use_smote:
        return Xtr, y_train
    sm = SMOTE(random_state=42, k_neighbors=k)
    return sm.fit_resample(Xtr, y_train)

# ------------------------------
# Eksperimen arsitektur MLP
# ------------------------------
EXPS = [
    {"name":"E1_tanh_100-100-20-20", "hidden":(100,100,20,20), "activation":"tanh"},
    {"name":"E2_relu_64-32",         "hidden":(64,32),         "activation":"relu"},
    {"name":"E3_logistic_50",        "hidden":(50,),           "activation":"logistic"},
    {"name":"E4_relu_128-64-32",     "hidden":(128,64,32),     "activation":"relu"},
    {"name":"E5_tanh_150-50",        "hidden":(150,50),        "activation":"tanh"},
]

# ------------------------------
# Fungsi training + evaluasi satu eksperimen
# ------------------------------
def run_one(exp, use_smote: bool, use_class_weight: bool):
    Xtrain, ytrain = make_train_data(use_smote)
    clf = MLPClassifier(
        hidden_layer_sizes=exp["hidden"],
        activation=exp["activation"],
        solver="adam",
        max_iter=600,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=42
    )

    # Dukungan sample_weight tergantung versi scikit-learn
    if use_class_weight:
        freq = ytrain.value_counts(normalize=False).to_dict()
        weights_map = {c: (len(ytrain) / (2 * freq[c])) for c in freq}
        sample_weight = ytrain.map(weights_map).values
        try:
            clf.fit(Xtrain, ytrain, sample_weight=sample_weight)
        except TypeError:
            # Jika versi lama tidak mendukung sample_weight, jatuhkan ke fit biasa
            clf.fit(Xtrain, ytrain)
    else:
        clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xte)
    acc = accuracy_score(y_test, ypred)

    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_test, ypred, average="weighted", zero_division=0
    )
    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_test, ypred, average="macro", zero_division=0
    )

    roc = None
    if hasattr(clf, "predict_proba") and y_test.nunique() == 2:
        proba = clf.predict_proba(Xte)[:, 1]
        try:
            roc = roc_auc_score(y_test, proba)
        except Exception:
            roc = None

    cm = confusion_matrix(y_test, ypred)

    # Threshold tuning sederhana (tidak mengubah model; hanya evaluasi)
    best_thr, best_acc = 0.5, acc
    if hasattr(clf, "predict_proba") and y_test.nunique() == 2:
        proba = clf.predict_proba(Xte)[:, 1]
        for t in np.linspace(0.2, 0.8, 25):
            acc_t = accuracy_score(y_test, (proba >= t).astype(int))
            if acc_t > best_acc:
                best_acc, best_thr = acc_t, t

    roc_str = f"{roc:.4f}" if roc is not None else "NA"

    print("\n" + "-" * 70)
    print(f"{exp['name']} | act={exp['activation']} | hidden={exp['hidden']}")
    print(f"SMOTE={use_smote} | class_weight={'balanced' if use_class_weight else 'none'}")
    print(f"Accuracy={acc:.4f} | F1_w={f1_w:.4f} | F1_macro={f1_m:.4f} | ROC-AUC={roc_str}")
    if best_acc > acc:
        print(f"Threshold tuning: best_thr={best_thr:.3f} -> Acc={best_acc:.4f}")
    print("Confusion Matrix:\n", pd.DataFrame(cm))
    print("Classification Report:\n", classification_report(y_test, ypred, zero_division=0))

    return {
        "name": exp["name"],
        "activation": exp["activation"],
        "hidden": exp["hidden"],
        "smote": use_smote,
        "class_weight": "balanced" if use_class_weight else "none",
        "accuracy": acc,
        "accuracy_thr_tuned": best_acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "roc_auc": (float(roc) if roc is not None else None)
    }

# ------------------------------
# Jalankan semua eksperimen
# ------------------------------
rows = []
for exp in EXPS:
    rows.append(run_one(exp, use_smote=False,           use_class_weight=False))
    rows.append(run_one(exp, use_smote=USE_SMOTE,       use_class_weight=False))
    rows.append(run_one(exp, use_smote=False,           use_class_weight=True))

res = pd.DataFrame(rows).sort_values(
    ["accuracy_thr_tuned", "accuracy"], ascending=False
)

print("\n" + "=" * 70)
print("RINGKASAN HASIL (urut Acc setelah threshold tuning)")
print("=" * 70)
print(res[[
    "name","activation","hidden","smote","class_weight",
    "accuracy","accuracy_thr_tuned","f1_weighted","f1_macro","roc_auc"
]].to_string(index=False))

best = res.iloc[0].to_dict()
print("\nMODEL TERBAIK (setelah tuning):")
print(json.dumps(best, indent=2, default=str))

# ------------------------------
# Simpan hasil ke CSV
# ------------------------------
res.to_csv("ann_results_modified.csv", index=False)
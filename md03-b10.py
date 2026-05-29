import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# =========================
# CONFIG
# =========================
DATA_FILE = 'Topic_2_fraud.xlsx'
META_FILE = 'Topic_2_fraud_metadata.xlsx'
DATASET_FILE = 'Topic_2_fraud_dataset.xlsx'
OUTDIR = 'output'
os.makedirs(OUTDIR, exist_ok=True)

CSV_METRICS = os.path.join(OUTDIR, 'evaluation_metrics.csv')
CSV_PRED = os.path.join(OUTDIR, 'predictions.csv')
CSV_KRI = os.path.join(OUTDIR, 'kri_summary.csv')
PLOT_FRAUD = os.path.join(OUTDIR, 'fraud_distribution.png')
PLOT_ROC = os.path.join(OUTDIR, 'roc_curve.png')
PLOT_CM = os.path.join(OUTDIR, 'confusion_matrix.png')
PLOT_TOP = os.path.join(OUTDIR, 'top_features.png')

TARGET_CANDIDATES = ['is_fraud', 'fraud', 'label', 'target', 'y']

def find_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'fraud' in c.lower() or 'label' in c.lower() or 'target' in c.lower():
            return c
    raise ValueError('Cannot find target column. Please rename it to is_fraud or update TARGET_CANDIDATES.')

def coerce_datetime_features(df):
    dt_cols = []
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ['time', 'date', 'timestamp', 'datetime']):
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().mean() >= 0.5:
                df[c] = parsed
                dt_cols.append(c)
    return dt_cols

def add_datetime_parts(df, dt_cols):
    for c in dt_cols:
        df[f'{c}_hour'] = df[c].dt.hour
        df[f'{c}_dayofweek'] = df[c].dt.dayofweek
        df[f'{c}_month'] = df[c].dt.month
        df[f'{c}_is_night'] = ((df[c].dt.hour >= 0) & (df[c].dt.hour <= 5)).astype('float')
    return df

# =========================
# LOAD DATA
# =========================
if os.path.exists(DATASET_FILE):
    df = pd.read_excel(DATASET_FILE)
else:
    df = pd.read_excel(DATA_FILE)

meta = pd.read_excel(META_FILE) if os.path.exists(META_FILE) else None

df.columns = [str(c).strip() for c in df.columns]
if meta is not None:
    meta.columns = [str(c).strip() for c in meta.columns]

target = find_target(df)

# =========================
# BASIC CLEANING / FEATURE ENGINEERING
# =========================
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].astype(str).str.strip()

_datetime_cols = coerce_datetime_features(df)
df = add_datetime_parts(df, _datetime_cols)

if 'amount' in df.columns:
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['amount_log'] = np.log1p(df['amount'])

if 'ip_risk' not in df.columns:
    if 'ip' in df.columns:
        df['ip_risk'] = df['ip'].astype(str).str.contains('proxy|vpn|tor|black', case=False, na=False).astype(int)

if 'device_diversity_30d' not in df.columns and 'device_id' in df.columns:
    try:
        tmp = df.groupby('device_id').size().rename('device_txn_count').reset_index()
        df = df.merge(tmp, on='device_id', how='left')
        df['device_diversity_30d'] = df['device_txn_count']
    except Exception:
        pass

if 'night_ratio_30d' not in df.columns:
    night_cols = [c for c in df.columns if c.endswith('_is_night')]
    if night_cols:
        df['night_ratio_30d'] = df[night_cols[0]]

if 'spending_trend' not in df.columns and 'amount' in df.columns:
    df['spending_trend'] = df['amount'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

clean_path = os.path.join(OUTDIR, 'cleaned_data.csv')
df.to_csv(clean_path, index=False)

# =========================
# KRI SUMMARY
# =========================
kri_candidates = [c for c in ['night_ratio_30d', 'device_diversity_30d', 'ip_risk', 'spending_trend'] if c in df.columns]
if len(kri_candidates) > 0:
    kri_rows = []
    for c in kri_candidates:
        x = df[c]
        if pd.api.types.is_numeric_dtype(x):
            kri_rows.append({
                'kri': c,
                'mean': float(pd.to_numeric(x, errors='coerce').mean()),
                'median': float(pd.to_numeric(x, errors='coerce').median()),
                'missing_rate': float(x.isna().mean())
            })
    pd.DataFrame(kri_rows).to_csv(CSV_KRI, index=False)

# =========================
# EDA PLOTS
# =========================
if target in df.columns:
    target_counts = df[target].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, palette='viridis')
    plt.title('Fraud distribution')
    plt.xlabel(target)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(PLOT_FRAUD, dpi=220)
    plt.close()

# =========================
# MODEL DATA
# =========================
y = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int)
X = df.drop(columns=[target])

id_like = [c for c in X.columns if any(k in c.lower() for k in ['id', 'uuid', 'index'])]
X = X.drop(columns=id_like, errors='ignore')

num_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None
)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', LogisticRegression(max_iter=1500, class_weight='balanced'))
])

clf.fit(X_train, y_train)

# =========================
# METRICS
# =========================
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    'AUC': roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else np.nan,
    'Accuracy': accuracy_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0)
}
cm = confusion_matrix(y_test, y_pred)
metrics.update({'TN': int(cm[0,0]), 'FP': int(cm[0,1]), 'FN': int(cm[1,0]), 'TP': int(cm[1,1])})

pd.DataFrame([metrics]).to_csv(CSV_METRICS, index=False)

pred_out = X_test.copy()
pred_out['y_true'] = y_test.values
pred_out['y_prob'] = y_prob
pred_out['y_pred'] = y_pred
pred_out.to_csv(CSV_PRED, index=False)

# ROC
if y_test.nunique() > 1:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}", color='blue')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_ROC, dpi=220)
    plt.close()

# Confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(PLOT_CM, dpi=220)
plt.close()

# =========================
# TOP FEATURES
# =========================
pre = clf.named_steps['preprocess']
feature_names = []
if len(num_cols) > 0:
    feature_names.extend(num_cols)
if len(cat_cols) > 0:
    ohe = pre.named_transformers_['cat'].named_steps['onehot']
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names.extend(ohe_names)

coef = clf.named_steps['model'].coef_.ravel()
feat_df = pd.DataFrame({'feature': feature_names, 'coef': coef})
feat_df['abs_coef'] = feat_df['coef'].abs()
feat_df = feat_df.sort_values('abs_coef', ascending=False).head(15)
feat_df.to_csv(os.path.join(OUTDIR, 'top_features.csv'), index=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_df, x='coef', y='feature', palette='coolwarm')
plt.title('Top logistic coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(PLOT_TOP, dpi=220)
plt.close()

print('DONE')
print('Metrics:', CSV_METRICS)
print('Predictions:', CSV_PRED)
print('KRI:', CSV_KRI)
print('Cleaned:', clean_path)

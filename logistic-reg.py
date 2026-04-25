import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# 1. LOAD & CLEAN
df = pd.read_csv('final_dataset-rk.csv', low_memory=False)

drop_cols = [
    'Unnamed: 0', 'trans_num', 'unix_time',
    'first', 'last', 'street', 'city',
    'random_noise_1', 'random_noise_2',
    'zip', 'merch_zipcode'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 2. FEATURE ENGINEERING
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine_vectorized(
    df['lat'], df['long'],
    df['merch_lat'], df['merch_long']
)

df = df.drop(columns=['lat', 'long', 'merch_lat', 'merch_long'])

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed', dayfirst=True)
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

df['dob'] = pd.to_datetime(df['dob'], format='mixed', dayfirst=True)
df['age'] = 2024 - df['dob'].dt.year

df['txn_count_per_card'] = df.groupby('cc_num')['cc_num'].transform('count')

merchant_col = df['merchant'].copy()

# 3. ENCODING
cat_cols = ['category', 'gender', 'state', 'job']
le = LabelEncoder()

for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

trash_cols = ['trans_date_trans_time', 'dob', 'merchant', 'cc_num']
df = df.drop(columns=[c for c in trash_cols if c in df.columns])

df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df.columns]

df = df.astype(float)

# 4. SPLIT
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test, merch_train, merch_test = train_test_split(
    X, y, merchant_col,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. TARGET ENCODE MERCHANT
merch_fraud_rate = y_train.groupby(merch_train).mean()

X_train['merchant_encoded'] = merch_train.map(merch_fraud_rate)
X_test['merchant_encoded'] = merch_test.map(merch_fraud_rate)

global_fraud_rate = y_train.mean()

X_train['merchant_encoded'] = X_train['merchant_encoded'].fillna(global_fraud_rate)
X_test['merchant_encoded'] = X_test['merchant_encoded'].fillna(global_fraud_rate)

# 6. HANDLE MISSING VALUES & SCALING
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. TRAIN LOGISTIC REGRESSION
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

# 8. EVALUATE
y_pred = lr_model.predict(X_test_scaled)
y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- LOGISTIC REGRESSION PERFORMANCE ---")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 9. SAVE THE MODEL
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

print("\n[SUCCESS] Logistic Regression model saved.")
import pandas as pd

import numpy as np

import re

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score

import lightgbm as lgb



# ── 1. LOAD ───────────────────────────────────────────────

# Using raw string or forward slashes to avoid path errors

df = pd.read_csv('final_dataset-rk.csv', low_memory=False)



# ── 2. DROP USELESS COLUMNS ───────────────────────────────

drop_cols = [

    'Unnamed: 0', 'trans_num', 'unix_time',

    'first', 'last', 'street', 'city',

    'random_noise_1', 'random_noise_2',

    'zip', 'merch_zipcode'

]

# CRITICAL: You must set df = df.drop...

df = df.drop(columns=drop_cols, errors='ignore')



# ── 3. ENGINEER FEATURES ──────────────────────────────────



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



# Time features

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed', dayfirst=True)

df['trans_hour'] = df['trans_date_trans_time'].dt.hour

df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek



# Age

df['dob'] = pd.to_datetime(df['dob'], format='mixed', dayfirst=True)

df['age'] = 2026 - df['dob'].dt.year



# Transaction count per card

df['txn_count_per_card'] = df.groupby('cc_num')['cc_num'].transform('count')



# ── 4. STORE MERCHANT FOR TARGET ENCODING ────────────────

merchant_col = df['merchant'].copy()



# ── 5. LABEL ENCODING ────────────────────────────────────

cat_cols = ['category', 'gender', 'state', 'job']

le = LabelEncoder()

for col in cat_cols:

    if col in df.columns:

        df[col] = le.fit_transform(df[col].astype(str))



# ── 5.5 CLEAN DATA TYPES & COLUMN NAMES ──────────────────

# Drop everything that isn't a number now

trash_cols = ['trans_date_trans_time', 'dob', 'merchant', 'cc_num']

df = df.drop(columns=[c for c in trash_cols if c in df.columns])



# Fix the "Special JSON Characters" error by cleaning column names

df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df.columns]



# Ensure everything is a float (this fixes the DType error)

df = df.astype(float)



# ── 6. TRAIN TEST SPLIT ───────────────────────────────────

X = df.drop(columns=['is_fraud'])

y = df['is_fraud']



X_train, X_test, y_train, y_test, merch_train, merch_test = train_test_split(

    X, y, merchant_col,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ── 7. TARGET ENCODE MERCHANT ─────────────────────────────

merch_fraud_rate = y_train.groupby(merch_train).mean()

X_train['merchant_encoded'] = merch_train.map(merch_fraud_rate)

X_test['merchant_encoded'] = merch_test.map(merch_fraud_rate)



global_fraud_rate = y_train.mean()

X_test['merchant_encoded'] = X_test['merchant_encoded'].fillna(global_fraud_rate)

X_train['merchant_encoded'] = X_train['merchant_encoded'].fillna(global_fraud_rate)



# ── 8. TRAIN LIGHTGBM ─────────────────────────────────────

model = lgb.LGBMClassifier(

    scale_pos_weight=174,

    n_estimators=1000,

    learning_rate=0.05,

    num_leaves=63,

    random_state=42,

    verbosity=-1 # Silences unnecessary warnings

)



model.fit(

    X_train, y_train,

    eval_set=[(X_test, y_test)],

    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]

)



# ── 9. EVALUATE ───────────────────────────────────────────

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]



print("\n--- PERFORMANCE REPORT ---")

print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))



import matplotlib.pyplot as plt

import pickle



# ── 10. SAVE THE MODEL ─────────────────────────────────────

# Save as .txt (Best for LightGBM compatibility)

model.booster_.save_model('fraud_detection_model.txt')



# Save as .pkl (Best for saving the whole Python object)

with open('fraud_model.pkl', 'wb') as f:

    pickle.dump(model, f)

print("\n[SUCCESS] Model saved as 'fraud_detection_model.txt' and 'fraud_model.pkl'")





# ── 11. FEATURE IMPORTANCE ────────────────────────────────

# Use X_train.columns to ensure names match the training data exactly

importance = pd.DataFrame({

    'Feature': X_train.columns,

    'Importance': model.feature_importances_

}).sort_values(by='Importance', ascending=False)



print("\n--- FEATURE IMPORTANCE RANKING ---")

print(importance)



# Create a bar chart

plt.figure(figsize=(12, 8))

plt.barh(importance['Feature'], importance['Importance'], color='skyblue')

plt.xlabel('Importance Score (Gain/Split)')

plt.title('Which Features Mattered Most for Fraud Detection?')

plt.gca().invert_yaxis()

plt.tight_layout() # Prevents labels from getting cut off

plt.show()
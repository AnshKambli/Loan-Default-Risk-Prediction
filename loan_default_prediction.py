import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load Dataset
# =========================

url = 'https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv'
df = pd.read_csv(url)

print(df.head())

# =========================
# Basic EDA
# =========================

print('\nDataset Shape:')
print(df.shape)

print('\nMissing Values:')
print(df.isnull().sum())

print('\nLoan Status Distribution:')
print(df['Loan_Status'].value_counts())

# =========================
# Visualization
# =========================

plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Approval Distribution')
plt.show()

# =========================
# Missing Value Handling
# =========================

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# =========================
# Encoding
# =========================

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_scaled = scaler.fit_transform(X)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Logistic Regression
# =========================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print('\n===== Logistic Regression =====')
print('Accuracy:', accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# =========================
# Random Forest
# =========================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print('\n===== Random Forest =====')
print('Accuracy:', accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# =========================
# Gradient Boosting
# =========================

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

print('\n===== Gradient Boosting =====')
print('Accuracy:', accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))

# =========================
# Feature Importance
# =========================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.show()

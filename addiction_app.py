import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Addiction Analyzer", layout="wide")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('Adiction_data.csv')
    return df

df = load_data()
st.title("📊 Student Social Media Addiction Analysis")
st.write("### Raw Dataset")
st.dataframe(df.head())

# --- Feature Engineering ---
df['HighAddiction'] = (df['Addicted_Score'] > 5).astype(int)
df = df.drop(columns=['Student_ID'])

# One-hot encode
df_encoded = pd.get_dummies(df, columns=[
    'Gender', 'Academic_Level', 'Country',
    'Most_Used_Platform', 'Affects_Academic_Performance',
    'Relationship_Status'
], drop_first=True)

# Split features/target
X = df_encoded.drop(columns=['Addicted_Score', 'HighAddiction'])
y = df_encoded['HighAddiction']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation ---
st.write("## ✅ Model Evaluation")
col1, col2 = st.columns(2)

with col1:
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc:.2%}")

    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred), language='text')

with col2:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# --- Feature Importance ---
st.write("## 🔍 Feature Importances")

importance = model.coef_[0]
feat_names = X.columns
feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance})
feat_df_sorted = feat_df.reindex(feat_df.Importance.abs().sort_values(ascending=False).index)
feat_df_top = feat_df_sorted.head(15)

fig_imp, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=feat_df_top, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False, ax=ax)
ax.set_title("Top 15 Important Features", fontsize=16)
ax.set_xlabel("Coefficient", fontsize=14)
ax.set_ylabel("Feature", fontsize=14)
ax.grid(axis='x', linestyle='--', alpha=0.6)
st.pyplot(fig_imp)

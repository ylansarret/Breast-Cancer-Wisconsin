# ======================
# Streamlit - Visualisation Cancer du Sein
# ======================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Visualisation Cancer du Sein", layout="wide")

st.title("📊 Visualisation du dataset Breast Cancer Wisconsin")
st.markdown("Explorez les données, la corrélation, les modèles et la PCA.")

# ======================
# Chargement des données
# ======================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

st.sidebar.header("⚙️ Options")
show_data = st.sidebar.checkbox("Afficher les premières lignes du dataset", value=True)
show_corr = st.sidebar.checkbox("Afficher la matrice de corrélation")
show_models = st.sidebar.checkbox("Afficher les performances modèles")
show_feat_imp = st.sidebar.checkbox("Afficher importance variables")
show_pca = st.sidebar.checkbox("Afficher PCA 2D")

# Affichage du dataset
if show_data:
    st.subheader("✅ Aperçu des données")
    st.dataframe(X.head())
    st.write("Dimensions :", X.shape)
    st.write("Répartition des classes : ", y.value_counts().to_dict())

# Corrélation
if show_corr:
    st.subheader("📈 Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(X.corr(), cmap='coolwarm', center=0, linewidths=0, ax=ax)
    st.pyplot(fig)

# ======================
# Split + modèles
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

pipe_lr = Pipeline([
    ('imp', imputer),
    ('sc', scaler),
    ('clf', LogisticRegression(max_iter=5000, random_state=42))
])

pipe_rf = Pipeline([
    ('imp', imputer),
    ('sc', scaler),
    ('clf', RandomForestClassifier(random_state=42))
])

grid_lr = {'clf__C': [0.01, 0.1, 1, 10]}
grid_rf = {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 5, 10]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring='roc_auc', n_jobs=-1)
gs_lr.fit(X_train, y_train)

gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=cv, scoring='roc_auc', n_jobs=-1)
gs_rf.fit(X_train, y_train)

# ======================
# Évaluation
# ======================
def eval_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return {'model': name, 'Accuracy': acc, 'Recall': rec, 'F1': f1, 'AUC': auc}

res_lr = eval_model(gs_lr.best_estimator_, X_test, y_test, "Logistic Regression")
res_rf = eval_model(gs_rf.best_estimator_, X_test, y_test, "Random Forest")

df_res = pd.DataFrame([res_lr, res_rf]).set_index('model')

if show_models:
    st.subheader("🤖 Performances des modèles")
    st.table(df_res)

# ======================
# Importance variables
# ======================
if show_feat_imp:
    st.subheader("🌟 Top 10 variables importantes (Random Forest)")
    feat_imp = pd.Series(gs_rf.best_estimator_['clf'].feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, color='royalblue', ax=ax)
    ax.set_xlabel("Importance moyenne")
    st.pyplot(fig)

# ======================
# PCA
# ======================
if show_pca:
    st.subheader("🔹 PCA - Projection 2D des données")
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7,6))
    scatter = ax.scatter(proj[:,0], proj[:,1], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    ax.set_title("Projection PCA (2D)")
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# ======================
# ☀️ CONFIGURATION GÉNÉRALE
# ======================
st.set_page_config(page_title="Météo du Cancer du Sein", page_icon="🌦️", layout="wide")

st.title("🌦️ Météo du Cancer du Sein")
st.markdown("""
Bienvenue sur **DataMétéo Santé**, la première station météo cellulaire !  
Notre mission : prédire si le climat biologique est **ensoleillé (bénin)** ☀️  
ou s’il risque de virer à la **tempête (malin)** 🌪️  
""")

# ======================
# 📊 CHARGEMENT DES DONNÉES
# ======================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# 🧠 ENTRAÎNEMENT DES MODÈLES MÉTÉO
# ======================
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

lr.fit(X_scaled, y)
rf.fit(X_scaled, y)

# ======================
# 🎛️ INTERFACE UTILISATEUR
# ======================
st.sidebar.header("🌡️ Réglez les conditions atmosphériques biologiques")

features_to_use = X.columns[:10]
user_input = {}

for feature in features_to_use:
    val_min = float(X[feature].min())
    val_max = float(X[feature].max())
    val_mean = float(X[feature].mean())
    user_input[feature] = st.sidebar.slider(f"{feature}", val_min, val_max, val_mean)

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# ======================
# 📈 PREDICTIONS
# ======================
pred_lr = lr.predict(input_scaled)[0]
pred_rf = rf.predict(input_scaled)[0]
proba_lr = lr.predict_proba(input_scaled)[0][1]
proba_rf = rf.predict_proba(input_scaled)[0][1]

# ======================
# 🌤️ AFFICHAGE DES PRÉVISIONS
# ======================
st.markdown("---")
st.subheader("🧭 Prévisions du jour")

col1, col2 = st.columns(2)

def get_weather_emoji(prob):
    if prob < 0.4:
        return "☀️"
    elif prob < 0.7:
        return "⛅️"
    else:
        return "🌪️"

with col1:
    emoji_lr = get_weather_emoji(proba_lr)
    st.markdown("### 👨‍🔬 Modèle : Régression Logistique")
    st.metric("Prévision du climat cellulaire", f"{emoji_lr} Risque : {int(proba_lr*100)} %")
    st.progress(int(proba_lr*100))
    st.caption("Modèle linéaire — prévision rapide et stable.")

with col2:
    emoji_rf = get_weather_emoji(proba_rf)
    st.markdown("### 🌲 Modèle : Forêt Aléatoire")
    st.metric("Prévision du climat cellulaire", f"{emoji_rf} Risque : {int(proba_rf*100)} %")
    st.progress(int(proba_rf*100))
    st.caption("Modèle non linéaire — prévision détaillée et robuste.")

# ======================
# ☁️ VISUELS INTERACTIFS
# ======================
st.markdown("---")
st.subheader("📊 Cartes météorologiques des cellules")

viz_choice = st.radio(
    "Choisissez la carte météo à afficher :",
    ("Carte des corrélations", "Carte PCA 2D", "Carte d’importance des variables")
)

if viz_choice == "Carte des corrélations":
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Carte des corrélations entre variables")
    st.pyplot(fig)

elif viz_choice == "Carte PCA 2D":
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel("Axe météo 1")
    ax.set_ylabel("Axe météo 2")
    ax.set_title("Projection météo des cellules")
    st.pyplot(fig)

elif viz_choice == "Carte d’importance des variables":
    importances = rf.feature_importances_[:10]
    fig, ax = plt.subplots()
    ax.barh(features_to_use, importances, color='skyblue')
    ax.set_xlabel("Impact météo")
    ax.set_title("Top 10 des variables influençant le climat cellulaire")
    st.pyplot(fig)

# ======================
# 🌈 MESSAGE FINAL
# ======================
st.markdown("---")
if proba_rf > 0.7:
    st.error("🌪️ **Alerte météo : vigilance rouge** – fortes probabilités de malignité.")
elif proba_rf > 0.4:
    st.warning("⛅️ **Vigilance orange** – risque modéré, à surveiller.")
else:
    st.success("☀️ **Ciel dégagé** – conditions bénignes confirmées.")

st.caption("Projet éducatif — Ne remplace pas un diagnostic médical. Réalisé avec ❤️ en Python, scikit-learn et Streamlit.")

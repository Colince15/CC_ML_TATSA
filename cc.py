# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle
import xgboost as xgb

# Charger le modèle correctement
model = xgb.XGBClassifier()
model.load_model("modele_xgboost.json")

menu = ["ACCUEIL","PREDICTION"]
i = st.sidebar.selectbox("CHOIX",menu)
if i == "ACCUEIL":
    # Sidebar logo
    image_path = 'keyce.jpg'
    st.sidebar.image(image_path, caption="Keyce informatique et intelligence artificielle", use_container_width=True)

    st.title("KEYCE INFORMATIQUE ET IA, MASTER II ")
    st.title("CONTROLE CONTINU DE MACHINE LEARNING DANS LE CLOUD")
    st.subheader("NOM DE L'ETUDIANT: TATSA TCHINDA Colince ")

elif i == "PREDICTION":

    st.title("🔬 Prédiction du Risque Cardiovasculaire")

    # Saisie utilisateur
    st.sidebar.header("📝 Entrez vos informations")

    def user_input_features():
        male = st.sidebar.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
        age = st.sidebar.slider("Âge", 20, 80, 50)
        currentSmoker = st.sidebar.selectbox("Fumeur actuel ?", [0, 1])
        cigsPerDay = st.sidebar.slider("Cigarettes par jour", 0, 50, 10)
        BPMeds = st.sidebar.selectbox("Prend des médicaments pour la tension ?", [0, 1])
        diabetes = st.sidebar.selectbox("Diabète ?", [0, 1])
        totChol = st.sidebar.slider("Cholestérol total", 100, 600, 200)
        sysBP = st.sidebar.slider("Pression systolique", 90, 250, 120)
        diaBP = st.sidebar.slider("Pression diastolique", 50, 150, 80)
        BMI = st.sidebar.slider("Indice de Masse Corporelle (IMC)", 10.0, 50.0, 25.0)
        heartRate = st.sidebar.slider("Fréquence cardiaque", 40, 140, 75)
        glucose = st.sidebar.slider("Taux de glucose", 50, 300, 90)

        data = {
            "male": male,
            "age": age,
            "currentSmoker": currentSmoker,
            "cigsPerDay": cigsPerDay,
            "BPMeds": BPMeds,
            "diabetes": diabetes,
            "totChol": totChol,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "BMI": BMI,
            "heartRate": heartRate,
            "glucose": glucose
        }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader("🧾 Données saisies")
    st.write(input_df)

    # Prédiction
    prediction_array = model.predict(input_df)

    if isinstance(prediction_array[0], (np.ndarray, list)):  # Si c'est un tableau
        prediction = int(np.argmax(prediction_array[0]))  # On prend l'index du max
    else:
        prediction = int(prediction_array[0])  # Sinon on convertit normalement

    # Probabilité
    prediction_proba = model.predict_proba(input_df)[0]

    if st.button("Afficher la prédiction"):
        st.subheader("📊 Résultat de la prédiction")
        st.write("**Risque prédit :**", "Oui" if prediction == 1 else "Non")
        st.write("**Probabilité :**", f"{prediction_proba[1]*100:.2f} % de risque")


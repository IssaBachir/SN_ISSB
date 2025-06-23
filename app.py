import streamlit as st
import torch
import numpy as np
import pandas as pd
from DiamondModel import DiamondModel
from standardisation import standardisation, to_tensor
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download

# Liste des features
features = ['carat', 'cut', 'depth', 'table', 'price', 'x', 'y', 'z', 'Color', 'Clarity']

# Colonnes catégorielles (à encoder)
cat_features = ['cut', 'Color', 'Clarity']

# Encoders pour les colonnes catégorielles (simples LabelEncoder avec classes fixes)
# IMPORTANT : les classes doivent être identiques à celles utilisées pour entraîner le modèle
# Tu peux soit hardcoder les classes, soit récupérer les LabelEncoders depuis ton entraînement (si possible)

# Exemple : classes utilisées (à remplacer par les vraies classes de ton train)
cut_classes = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_classes = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_classes = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

def create_label_encoder(classes):
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    return le

le_cut = create_label_encoder(cut_classes)
le_color = create_label_encoder(color_classes)
le_clarity = create_label_encoder(clarity_classes)

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = hf_hub_download(repo_id="issabachir6/test2", filename="model_final.pth")
    model = DiamondModel(input_dim=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title("Prédiction qualité diamant")

# Saisie utilisateur pour chaque feature
user_input = {}

user_input['carat'] = st.number_input("Carat", min_value=0.0, step=0.01, format="%.2f")
user_input['cut'] = st.selectbox("Cut", cut_classes)
user_input['depth'] = st.number_input("Depth", min_value=0.0, step=0.01, format="%.2f")
user_input['table'] = st.number_input("Table", min_value=0.0, step=0.01, format="%.2f")
user_input['price'] = st.number_input("Price", min_value=0)
user_input['x'] = st.number_input("X (length)", min_value=0.0, step=0.01, format="%.2f")
user_input['y'] = st.number_input("Y (width)", min_value=0.0, step=0.01, format="%.2f")
user_input['z'] = st.number_input("Z (depth)", min_value=0.0, step=0.01, format="%.2f")
user_input['Color'] = st.selectbox("Color", color_classes)
user_input['Clarity'] = st.selectbox("Clarity", clarity_classes)

if st.button("Prédire"):

    # Encodage des catégorielles
    input_list = []
    for feat in features:
        if feat == 'cut':
            input_list.append(le_cut.transform([user_input[feat]])[0])
        elif feat == 'Color':
            input_list.append(le_color.transform([user_input[feat]])[0])
        elif feat == 'Clarity':
            input_list.append(le_clarity.transform([user_input[feat]])[0])
        else:
            input_list.append(float(user_input[feat]))

    input_array = np.array(input_list).reshape(1, -1)

    # Standardisation
    input_std = standardisation(input_array)

    # Conversion en tenseur
    input_tensor = to_tensor(input_std)

    # Prédiction
    with torch.inference_mode():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1).item()

    st.write(f"Prédiction de la classe : {pred}")

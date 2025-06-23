import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DiamondModel import DiamondModel
from standardisation import standardisation, to_tensor
import json

def safe_label_encode(train_col, test_col):
    le = LabelEncoder()
    le.fit(train_col)
    known_labels = set(le.classes_)
    test_col_safe = [x if x in known_labels else le.classes_[0] for x in test_col]
    return le.transform(train_col), le.transform(test_col_safe)

def evaluate():
    # Chargement des données
    X_test = np.load("data/x_test.npy", allow_pickle=True)
    y_test = np.load("data/y_test.npy", allow_pickle=True)

    # Chargement des données d'entraînement pour réutiliser les encodages
    X_train = np.load("data/x_train.npy", allow_pickle=True)
    y_train = np.load("data/y_train.npy", allow_pickle=True)

    # Encodage des labels (comme dans train.py)
    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    known_labels = set(le_y.classes_)
    y_test = np.array([y if y in known_labels else le_y.classes_[0] for y in y_test])
    y_test = le_y.transform(y_test)

    # Conversion en DataFrame pour encodage des colonnes catégorielles
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    cat_cols = X_train.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        train_encoded, test_encoded = safe_label_encode(X_train[col], X_test[col])
        X_test[col] = test_encoded

    # Conversion en array float
    X_test_num = X_test.values.astype(float)

    # Standardisation et conversion en tenseurs
    X_test_t = to_tensor(standardisation(X_test_num))
    y_test_t = to_tensor(y_test).long()

    # Chargement du modèle
    model = DiamondModel(X_test_t.shape[1])
    model.load_state_dict(torch.load("./models/model_final.pth"))
    model.eval()

    # Prédiction
    with torch.inference_mode():
        y_logits = model(X_test_t)
        y_pred = torch.argmax(y_logits, dim=1)

    # Calcul de l'exactitude
    correct = torch.eq(y_pred, y_test_t).sum().item()
    accuracy = correct / len(y_test_t) * 100

    print(f"Evaluation Accuracy: {accuracy:.2f}%")

    # ✅ Sauvegarde dans metrics.json pour GitHub Actions
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

if __name__ == "__main__":
    evaluate()

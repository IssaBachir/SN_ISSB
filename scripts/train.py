import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DiamondModel import DiamondModel
from standardisation import standardisation, to_tensor

def safe_label_encode(train_col, test_col):
    le = LabelEncoder()
    le.fit(train_col)
    known_labels = set(le.classes_)
    test_col_safe = [x if x in known_labels else le.classes_[0] for x in test_col]
    return le.transform(train_col), le.transform(test_col_safe)

class Trainner:
    def __init__(self):
        # Générer les fichiers .npy s'ils n'existent pas
        if not os.path.exists("data/x_train.npy"):
            print("⚠️ Fichiers .npy introuvables. Génération depuis diamonds.csv...")
            df = pd.read_csv("data/diamonds.csv")

            # Utiliser la colonne 'cut' comme cible
            X = df.drop("cut", axis=1).values
            y = df["cut"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            np.save("data/x_train.npy", X_train)
            np.save("data/x_test.npy", X_test)
            np.save("data/y_train.npy", y_train)
            np.save("data/y_test.npy", y_test)
            print("✅ Fichiers .npy générés.")

        # Charger les données
        self.X_train = np.load("data/x_train.npy", allow_pickle=True)
        self.y_train = np.load("data/y_train.npy", allow_pickle=True)
        self.X_test = np.load("data/x_test.npy", allow_pickle=True)
        self.y_test = np.load("data/y_test.npy", allow_pickle=True)

        # Encoder les labels
        self.le_y = LabelEncoder()
        self.y_train = self.le_y.fit_transform(self.y_train)
        known_labels = set(self.le_y.classes_)
        self.y_test = np.array([y if y in known_labels else self.le_y.classes_[0] for y in self.y_test])
        self.y_test = self.le_y.transform(self.y_test)

        # Convertir en DataFrame pour gérer les colonnes catégorielles
        self.X_train = pd.DataFrame(self.X_train)
        self.X_test = pd.DataFrame(self.X_test)

        cat_cols = self.X_train.select_dtypes(include=['object']).columns
        for col in cat_cols:
            train_encoded, test_encoded = safe_label_encode(self.X_train[col], self.X_test[col])
            self.X_train[col] = train_encoded
            self.X_test[col] = test_encoded

        X_train_num = self.X_train.values.astype(float)
        X_test_num = self.X_test.values.astype(float)

        # Standardisation et conversion en tenseurs
        self.X_train_t = to_tensor(standardisation(X_train_num))
        self.X_test_t = to_tensor(standardisation(X_test_num))

        self.y_train_t = to_tensor(self.y_train).long()
        self.y_test_t = to_tensor(self.y_test).long()

    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train(self, model, epochs=10000, step=1000):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epoch_count = []
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []

        torch.manual_seed(42)

        for epoch in range(epochs):
            model.train()
            y_logits = model(self.X_train_t)
            y_pred = torch.argmax(y_logits, dim=1)

            loss = loss_fn(y_logits, self.y_train_t)
            acc = self.accuracy_fn(self.y_train_t, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                test_logits = model(self.X_test_t)
                test_pred = torch.argmax(test_logits, dim=1)
                test_loss = loss_fn(test_logits, self.y_test_t)
                test_acc = self.accuracy_fn(self.y_test_t, test_pred)

            if epoch % step == 0:
                epoch_count.append(epoch)
                train_acc_list.append(acc)
                test_acc_list.append(test_acc)
                train_loss_list.append(loss.item())
                test_loss_list.append(test_loss.item())

                print(
                    f"Epoch:{epoch}, | Loss:{loss:.5f} | Acc={acc:.2f}% | Test Loss:{test_loss:.5f} | Test Acc:{test_acc:.2f}%"
                )
 
        return model

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)


if __name__ == "__main__":
    trainner = Trainner()
    model = DiamondModel(trainner.X_train_t.shape[1])
    model = trainner.train(model)
    trainner.save_model(model, f"./models/model_final.pth")
    print("✅ Modèle sauvegardé avec succès")

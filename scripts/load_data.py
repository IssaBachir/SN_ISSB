import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PreprocessingPipeline:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(self.path)

    def preprocessing(self):
        # Compte des lignes dupliquées et non dupliquées
        self.data.duplicated().value_counts()

        # Suppresion de la column unnamed
        self.data.drop("Unnamed: 0", axis=1, inplace=True)

        # Convertion de type
        self.data["table"] = self.data["table"].astype("int")

        # encodage des colums
        fe = self.data.groupby("color").size() / len(self.data)
        self.data.loc[:, "Color"] = self.data.color.map(fe)
        self.data.drop("color", axis=1, inplace=True)

        fe = self.data.groupby("clarity").size() / len(self.data)
        self.data.loc[:, "Clarity"] = self.data.clarity.map(fe)
        self.data.drop("clarity", axis=1, inplace=True)

        # Encodage de la column categorielle
        lencoder = LabelEncoder()
        self.data.cut = lencoder.fit_transform(self.data.cut)

        return self.data

    def del_outliers(self):
        num_col = self.data.select_dtypes(include=["number"])

        for i in num_col:

            Q3 = np.percentile(self.data[i], 75)
            Q1 = np.percentile(self.data[i], 25)
            IQR = Q3 - Q1
            Maxi = Q3 + 1.5 * IQR
            Mini = Q1 - 1.5 * IQR

            # Ramenons les outliers a la distriution

            self.data[i][self.data[i] > Maxi] = Maxi
            self.data[i][self.data[i] < Mini] = Mini

    def split_dataset(self):
        # Separation X et Y et transformation en numpy array

        X = self.data.drop(["cut"], axis=1).values
        y = self.data["cut"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=44
        )

        output_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "x_train.npy"), self.x_train)
        np.save(os.path.join(output_dir, "x_test.npy"), self.x_test)
        np.save(os.path.join(output_dir, "y_train.npy"), self.y_train)
        np.save(os.path.join(output_dir, "y_test.npy"), self.y_test)

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    path = "../data/diamonds.csv"

    pipeline = PreprocessingPipeline(path)
    pipeline.preprocessing()
    pipeline.del_outliers()
    X_train, X_test, y_train, y_test = pipeline.split_dataset()

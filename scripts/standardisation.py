import torch
from sklearn.preprocessing import StandardScaler


def standardisation(X):
    sc = StandardScaler()
    return sc.fit_transform(X)


def to_tensor(X):
    # Transformation en tensors

    return torch.tensor(X, dtype=torch.float32)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
y = iris.target
unique, counts = np.unique(y, return_counts=True)
print("Original Distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
print("\nTrain Distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")
print("\nTest Distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")
print("\nProportion Check (Train):", np.bincount(y_train) / len(y_train))
print("Proportion Check (Test):", np.bincount(y_test) / len(y_test))

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# Load dataset
data = load_iris()
X, y = data.data, data.target

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

depths = [1,2,3,4,5,6,None]

print("Depth -- Mean CV Accuracy")
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    print(f"{d} -- {scores.mean():.4f}")

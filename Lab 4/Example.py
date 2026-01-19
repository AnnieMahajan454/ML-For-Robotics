import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Target classes:", iris.target_names)
print("Shape of X:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----- ID3 TREE (Entropy) -----
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt_entropy.fit(X_train, y_train)

y_pred_entropy = dt_entropy.predict(X_test)
print("Entropy Accuracy:", accuracy_score(y_test, y_pred_entropy))

plt.figure(figsize=(12,6))
plot_tree(dt_entropy,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()

# ----- CART TREE (Gini) -----
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
dt_gini.fit(X_train, y_train)

y_pred = dt_gini.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12,6))
plot_tree(dt_gini,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()

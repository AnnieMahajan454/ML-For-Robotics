import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data[:, 0].reshape(-1, 1)  # Take one feature only
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train, y_train)
x_values = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_values = model.predict_proba(x_values)[:, 1]
plt.plot(x_values, y_values)
plt.xlabel("Feature value")
plt.ylabel("Probability")
plt.title("Sigmoid Curve")
plt.show()

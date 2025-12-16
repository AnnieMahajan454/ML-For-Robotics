import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
X = np.array(x).reshape(-1, 1)
Y = np.array(y)
model = LinearRegression()
model.fit(X, Y)
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, Y)
print(f"Slope (Coefficient): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual data', s=80)
plt.plot(x, y_pred, color='red', linewidth=2, label='Linear regression line')
plt.xlabel('Age of car (years)')
plt.ylabel('Speed of car')
plt.title('Linear Regression: Car Age vs Speed')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

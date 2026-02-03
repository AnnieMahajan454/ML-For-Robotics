from sklearn.neighbors import KNeighborsRegressor
import numpy as np
# Simple dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=2)
knn_reg.fit(X, y)
# Prediction
print("Predicted value for 3.5:",
      knn_reg.predict([[3.5]]))

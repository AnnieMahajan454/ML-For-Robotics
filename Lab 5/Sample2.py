from sklearn.datasets import load_diabetes 
from sklearn.tree import DecisionTreeRegressor, plot_tree 
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt 
import numpy as np 
# Load dataset 
data = load_diabetes() 
X = data.data 
y = data.target 
# Train model 
model = DecisionTreeRegressor(max_depth=5, random_state=42) 
model.fit(X, y) 
# Cross validation MSE 
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error") 
mse = -scores.mean() 
print("Mean MSE:", mse)

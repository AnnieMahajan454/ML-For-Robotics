from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.model_selection import KFold, cross_val_score 
import numpy as np 
import matplotlib.pyplot as plt 
# Load dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
# Create decision tree classifier 
dt = DecisionTreeClassifier(criterion='gini') 
# K-fold setup 
kfold = KFold(n_splits=5, shuffle=True, random_state=42) 
# Perform cross validation 
scores = cross_val_score(dt, X, y, cv=kfold, scoring="accuracy") 
print("Accuracy scores:", scores) 
print("Mean accuracy:", scores.mean()) 
print("Standard deviation:", scores.std()) 
# Fit the model on the full dataset 
dt.fit(X, y) 

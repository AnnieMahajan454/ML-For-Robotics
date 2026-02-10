from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.model_selection import GridSearchCV, KFold 
import matplotlib.pyplot as plt 
# Load dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
dt = DecisionTreeClassifier(random_state=42) 
param_grid = { 
'max_depth': [2, 3, 4, 5, 6, None], 
'min_samples_split': [2, 5, 10], 
'min_samples_leaf': [1, 2, 4], 
'criterion': ['gini', 'entropy'] 
} 
kfold = KFold(n_splits=5, shuffle=True, random_state=42) 
grid_search = GridSearchCV( 
estimator=dt, 
param_grid=param_grid, 
cv=kfold, 
scoring='accuracy', 
n_jobs=-1 
) 
grid_search.fit(X, y) 
print("Best hyperparameters:", grid_search.best_params_) 
print("Best cross-validation accuracy:", grid_search.best_score_) 
# Train final model using best parameters 
best_model = grid_search.best_estimator_ 

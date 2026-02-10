import pandas as pd 
from sklearn.tree import DecisionTreeRegressor, plot_tree 
from sklearn.model_selection import GridSearchCV, KFold 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt 
# Load dataset 
train_df = pd.read_csv('train.csv') 
# Separate target and features 
y = train_df['SalePrice'] 
X = train_df.drop(columns=['SalePrice', 'Id']) 
# Identify categorical and numerical columns 
cat_cols = X.select_dtypes(include=['object']).columns 
num_cols = X.select_dtypes(exclude=['object']).columns 
# Preprocessing 
preprocessor = ColumnTransformer([ 
('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols), 
('num', 'passthrough', num_cols) 
]) 
# Decision Tree Regressor 
dt_reg = DecisionTreeRegressor(random_state=42) 
# Hyperparameter grid 
param_grid = { 
'model__max_depth': [3, 5, 7, 10, None], 
'model__min_samples_split': [2, 5, 10, 20], 
'model__min_samples_leaf': [1, 2, 5, 10] 
} 
# Pipeline 
pipeline = Pipeline([ 
('preprocess', preprocessor), 
('model', dt_reg) 
]) 
# Cross validation setup 
kfold = KFold(n_splits=5, shuffle=True, random_state=42) 
# Grid Search 
grid_search = GridSearchCV( 
estimator=pipeline, 
param_grid=param_grid, 
scoring='neg_root_mean_squared_error', 
cv=kfold, 
n_jobs=-1 
) 
# Fit grid search 
grid_search.fit(X, y) 
print("Best hyperparameters:", grid_search.best_params_) 
print("Best CV RMSE:", -grid_search.best_score_) 
# Train final best model 
best_pipeline = grid_search.best_estimator_ 
# Extract trained decision tree from pipeline 
best_tree = best_pipeline.named_steps['model'] 
# Transform X using preprocessing 
X_transformed = best_pipeline.named_steps['preprocess'].transform(X)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("adult.csv")
X = data.drop("income", axis=1)
y = data["income"]

cat = ["education"]
num = ["age","hours"]

pre = ColumnTransformer([
    ("cat",OneHotEncoder(),cat),
    ("num","passthrough",num)
])

pipe = Pipeline([
    ("prep",pre),
    ("model",DecisionTreeClassifier())
])

param={
    "model__max_depth":[3,5,None],
    "model__min_samples_leaf":[1,2,5]
}

grid=GridSearchCV(pipe,param,cv=5)
grid.fit(X,y)

print(grid.best_params_)
print(grid.best_score_)

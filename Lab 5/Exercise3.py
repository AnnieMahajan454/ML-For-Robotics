import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("titanic.csv")

X = data.drop("Survived", axis=1)
y = data["Survived"]

cat_cols = ["Sex"]
num_cols = ["Pclass","Age","Fare"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(), cat_cols),
    ("num", "passthrough", num_cols)
])

pipe = Pipeline([
    ("prep", pre),
    ("model", DecisionTreeClassifier())
])

param = {
    "model__max_depth":[2,3,5,None],
    "model__min_samples_split":[2,5,10]
}

grid = GridSearchCV(pipe, param, cv=5)
grid.fit(X,y)

print(grid.best_params_)
print(grid.best_score_)

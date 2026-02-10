import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

data = pd.read_csv("house_data.csv")
X = data.drop("Price", axis=1)
y = data["Price"]

depths = [2,3,4,5,6,8]

print("Depth -- MSE")
for d in depths:
    model = DecisionTreeRegressor(max_depth=d)
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    mse = -scores.mean()
    print(d, "--", mse)

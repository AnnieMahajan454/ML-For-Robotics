import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Load and prepare data
df = sns.load_dataset('titanic')
df_filled = df.copy()
df_filled['age'] = df_filled['age'].fillna(df_filled['age'].median())
df_filled['deck'] = df_filled['deck'].astype('object').fillna("Unknown")
df_filled.dropna(subset=['embarked'], inplace=True)

data = df_filled[['survived','pclass','age','fare','sex']]
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
depths = [1,2,3,4,5,6,7,8,9,10]
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Depth={d}, Train Acc={train_acc:.2f}, Test Acc={test_acc:.2f}")
    best_model = DecisionTreeClassifier(max_depth=3, random_state=42)
best_model.fit(X_train, y_train)
plt.figure(figsize=(12,6))
plot_tree(best_model,
          feature_names=X.columns,
          class_names=['Not Survived','Survived'],
          filled=True)
plt.show()


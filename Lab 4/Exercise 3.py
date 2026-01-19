import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# Load and prepare data
df = sns.load_dataset('titanic')
df_filled = df.copy()
df_filled['age'] = df_filled['age'].fillna(df_filled['age'].median())
df_filled['deck'] = df_filled['deck'].astype('object').fillna("Unknown")
df_filled.dropna(subset=['embarked'], inplace=True)

data = df_filled[['survived','pclass','age','fare','sex']].copy()
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
train_acc = []
test_acc = []
for alpha in ccp_alphas:
    clf_alpha = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    clf_alpha.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf_alpha.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf_alpha.predict(X_test)))
plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, train_acc, marker='o', label='Train Accuracy')
plt.plot(ccp_alphas, test_acc, marker='s', label='Test Accuracy')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Post-Pruning on Titanic Dataset')
plt.legend()
plt.grid(True)
plt.show()
best_alpha = ccp_alphas[test_acc.index(max(test_acc))]
clf_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
clf_pruned.fit(X_train, y_train)
print("Best ccp_alpha:", best_alpha)
print("Test Accuracy (pruned tree):",
      accuracy_score(y_test, clf_pruned.predict(X_test)))
plt.figure(figsize=(12,6))
plot_tree(clf_pruned,
          feature_names=X.columns,
          class_names=['Not Survived','Survived'],
          filled=True)
plt.show()


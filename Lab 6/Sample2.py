from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
y = iris.target
# Stratified split keeps class distribution same
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

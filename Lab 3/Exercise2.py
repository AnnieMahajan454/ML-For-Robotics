import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
df = pd.read_csv("diabetes.csv") 
X = df.drop('Outcome', axis=1) 
y = df['Outcome'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
model = GaussianNB() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred))

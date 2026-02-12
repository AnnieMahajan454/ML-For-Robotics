import numpy as np 
from sklearn.naive_bayes import GaussianNB 
X = np.array([[5.1],[4.9],[5.0],[6.7],[6.5],[6.8]]) 
y = np.array(['A','A','A','B','B','B']) 
model = GaussianNB() 
model.fit(X, y) 
test = np.array([[6.6]]) 
print("Prediction:", model.predict(test))

import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder 
X = pd.DataFrame({ 
'Color': ['Red','Blue','Green','Blue','Red'], 
'Size': ['Small','Medium','Large','Small','Large'] 
}) 
y = ['Yes','No','Yes','No','Yes'] 
encoder = OrdinalEncoder() 
X_encoded = encoder.fit_transform(X) 
print("Encoded Training Data:\n", X_encoded) 
X_test = pd.DataFrame({ 
'Color': ['Green','Red'], 
'Size': ['Medium','Small'] 
}) 
X_test_encoded = encoder.transform(X_test) 
print("Encoded Test Data:\n", X_test_encoded)

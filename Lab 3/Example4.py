import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder 
X = pd.DataFrame({ 
'Color': ['Red','Blue','Green','Blue','Red'], 
'Size': ['Small','Medium','Large','Small','Large'] 
}) 
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) 
encoder.fit(X) 
X_test_new = pd.DataFrame({ 
'Color': ['Yellow'],    
'Size': ['Small'] 
}) 
X_test_new_encoded = encoder.transform(X_test_new) 
print("Encoded Unseen Data:\n", X_test_new_encoded)

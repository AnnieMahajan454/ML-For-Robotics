import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.naive_bayes import GaussianNB 
data = { 
'Age': [25,40,35,50,28], 
'Income': [50000,60000,58000,80000,52000], 
'MaritalStatus': ['Single','Married','Single','Married','Single'], 
'CreditScore': [700,650,720,600,690], 
'LoanApproved': ['Yes','No','Yes','No','Yes'] 
} 
df = pd.DataFrame(data) 
X = df.drop('LoanApproved', axis=1) 
y = df['LoanApproved'] 
categorical_cols = ['MaritalStatus'] 
encoder = OrdinalEncoder() 
X[categorical_cols] = encoder.fit_transform(X[categorical_cols]) 
model = GaussianNB() 
model.fit(X, y) 
test = pd.DataFrame({ 
'Age':[30], 
'Income':[54000], 
'MaritalStatus':['Single'], 
'CreditScore':[710] 
}) 
test[categorical_cols] = encoder.transform(test[categorical_cols]) 
pred = model.predict(test) 
print("Predicted Loan Approval:", pred[0])

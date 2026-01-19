import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB


def main():
    X = pd.DataFrame({
        'Outlook': ['Sunny','Sunny','Overcast','Rainy','Rainy','Sunny'],
        'Humidity': ['High','High','High','Normal','Normal','Normal'],
        'Wind': ['Weak','Strong','Weak','Weak','Strong','Weak']
    })
    y = ['No','No','Yes','Yes','No','Yes']
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    model = CategoricalNB()
    model.fit(X_encoded, y)
    test = pd.DataFrame([['Sunny','High','Weak']], columns=X.columns)
    test_encoded = encoder.transform(test)
    print("Prediction:", model.predict(test_encoded), flush=True)


if __name__ == "__main__":
    main()

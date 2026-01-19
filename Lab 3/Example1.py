import pandas as pd
from sklearn.naive_bayes import CategoricalNB

def encode_with_training_categories(train_df, test_df):
    test_encoded = test_df.copy()
    for col in train_df.columns:
        categories = train_df[col].astype('category').cat.categories
        test_encoded[col] = pd.Categorical(test_df[col], categories=categories).codes
    return test_encoded


def main():
    X = pd.DataFrame({
        'Outlook': ['Sunny','Sunny','Overcast','Rainy','Rainy','Sunny'],
        'Humidity': ['High','High','High','Normal','Normal','Normal'],
        'Wind': ['Weak','Strong','Weak','Weak','Strong','Weak']
    })
    y = ['No','No','Yes','Yes','No','Yes']
    X_encoded = X.apply(lambda col: col.astype('category').cat.codes)
    model = CategoricalNB()
    model.fit(X_encoded, y)
    test = pd.DataFrame([['Sunny','High','Weak']], columns=X.columns)
    test_encoded = encode_with_training_categories(X, test)
    print("Prediction:", model.predict(test_encoded), flush=True)


if __name__ == "__main__":
    main()
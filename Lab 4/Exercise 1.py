import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
print(df.head())
print("\nMissing values:\n", df.isna().sum())
before = df.isna().sum()
plt.figure(figsize=(10,5))
before.plot(kind='bar')
plt.title("Missing Values Before Handling")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(12,4))
sns.heatmap(df.isna(), cbar=False)
plt.title("Heatmap Before Handling Missing Data")
plt.show()
df_filled = df.copy()
# Age – Median Imputation
df_filled['age'] = df_filled['age'].fillna(df_filled['age'].median())
# Cabin – Fill with "Unknown"
df_filled['deck'] = df_filled['deck'].astype('object').fillna("Unknown")
# Embarked – Remove rows
df_filled.dropna(subset=['embarked'], inplace=True)
print("\nAfter Handling:\n", df_filled.isna().sum())
after = df_filled.isna().sum()
comparison = pd.DataFrame({'Before': before, 'After': after})
comparison.plot(kind='bar', figsize=(10,5))
plt.ylabel('Number of missing values')
plt.title('Missing Values: Before vs After Imputation')
plt.show()
sns.heatmap(df_filled.isna(), cbar=False)
plt.title("After Handling Missing Data")
plt.show()
# Histogram Visualization
plt.figure(figsize=(8,4))
df['age'].hist(alpha=0.5, bins=20, label='Before')
df_filled['age'].hist(alpha=0.5, bins=20, label='After')
plt.legend()
plt.title('Age Distribution Before vs After Imputation')
plt.show()



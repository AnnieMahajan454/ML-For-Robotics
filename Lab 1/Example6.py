import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load
df = pd.read_csv('Lab1_StudentsPerformance.csv')

print("First 5 rows:")
print(df.head())

# Numeric Columns
numeric_cols = ["math score", "reading score", "writing score"]
df_numeric = df[numeric_cols]

print("\nSummary Statistics:")
print(df_numeric.describe())

# Boxplots
plt.figure(figsize=(12, 5))

for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f"Variability in {col}")

plt.tight_layout()
plt.show()

print("\nInterpretation Guide:")
print("- Height of box → spread of middle 50%")
print("- Whiskers → overall range")
print("- Dots → outliers")
print("- Differences in width show variability before modeling")

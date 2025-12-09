import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your tips dataset
df = pd.read_csv("Lab1_Tips.csv")   

# Select numeric columns to analyze variability
numeric_cols = ["total_bill", "tip", "size"]

print("Summary Statistics:")
print(df[numeric_cols].describe())

# Create boxplots showing variability
plt.figure(figsize=(12, 4))

for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f"Variability in {col}")

plt.tight_layout()
plt.show()

print("\nInterpretation Guide:")
print("- Larger box height = more variability in that variable.")
print("- Whiskers show the full range of values.")
print("- Points outside the whiskers = outliers.")

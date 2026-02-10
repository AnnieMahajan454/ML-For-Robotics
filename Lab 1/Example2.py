import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load your uploaded file
df = pd.read_csv('Lab1_Vanilla.csv')

# Summary statistics
print("Mean:", df.iloc[:,0].mean())
print("Median:", df.iloc[:,0].median())
print("Mode:", df.iloc[:,0].mode()[0])
print("Five-Point Summary:\n", df.describe())

# IQR Outlier Detection
Q1 = df.iloc[:,0].quantile(0.25)
Q3 = df.iloc[:,0].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

print("\nOutliers:")
print(df[(df.iloc[:,0] < lower) | (df.iloc[:,0] > upper)])

plt.boxplot(df.iloc[:,0])
plt.title("Calories – Vanilla Ice Cream Bars")
plt.ylabel("Calories")
plt.show()

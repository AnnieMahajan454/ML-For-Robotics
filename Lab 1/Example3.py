import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Lab1_Tips.csv')

df.boxplot(column=['total_bill'])
plt.title("Outlier Detection – Total Bill")
plt.show()

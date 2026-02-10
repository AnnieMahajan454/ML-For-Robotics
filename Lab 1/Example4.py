import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Lab1_Tips.csv')

df.boxplot(by='day', column=['total_bill'], grid=False)
plt.ylabel('Total Bill')
plt.xlabel('Days')
plt.title('Boxplot of Total Bill by Day')
plt.suptitle("")   # removes automatic pandas title
plt.show()

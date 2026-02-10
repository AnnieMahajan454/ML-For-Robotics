import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')
tips = sns.load_dataset('tips')

sns.boxplot(x='day', y='tip', data=tips)
plt.title("Seaborn Boxplot – Tips by Day")
plt.show()

import matplotlib.pyplot as plt
from scipy import stats
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-value (correlation coefficient): {r_value:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Standard Error: {std_err:.4f}")
def predict(x_val):
    return intercept + slope * x_val
y_pred = [predict(x_val) for x_val in x]
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual data', s=80)
plt.plot(x, y_pred, color='red', linewidth=2, label='Linear regression line')
plt.xlabel('Age of car (years)')
plt.ylabel('Speed of car')
plt.title('Linear Regression: Car Age vs Speed')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

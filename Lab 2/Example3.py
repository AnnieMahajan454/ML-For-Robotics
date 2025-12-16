import matplotlib.pyplot as plt
from scipy import stats
x = [2,4,6,8]
y = [3,7,5,10]
B1, B0, r, p, std_err = stats.linregress(x, y)
print(B0, B1, r, std_err)
# correlation coefficient, p: p-value
# (statistical significance of the slope)
def myfunc(x):
    return B0 + B1 * x
#map(myfunc, x) applies myfunc to every value in x
mymodel = list(map(myfunc, x))
#speed = myfunc(10)
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
#print(speed)

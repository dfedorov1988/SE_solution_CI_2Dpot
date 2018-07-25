import numpy as np
def exp(x):
    return 0.0009986 * np.exp(-0.09711 * x) + 0.5097 * np.exp(-1.762e-06 * x)
def poly(x):
    return -8.987e-11 * x**3 + 3.23e-08 * x**2 + -4.132e-06 * x + 0.5098
print(exp(100))

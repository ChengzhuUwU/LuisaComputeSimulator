import numpy as np

import matplotlib.pyplot as plt

def square_scalar(x):
    return x * x

def log_scalar(x):
    return np.log(x)

def NoKappa_Barrier(d2, dHat=0.01, xi=0.0):
    x0 = square_scalar(xi)
    x1 = square_scalar(dHat) + 2.0 * dHat * xi
    # To avoid log of zero or negative, ensure d2 > x0
    # valid = d2 > x0
    valid = d2 < square_scalar(dHat + xi)
    R = np.zeros_like(d2)
    R[valid] = -square_scalar(d2[valid] - x0 - x1) * log_scalar((d2[valid] - x0) / x1)
    R[~valid] = np.nan  # Mark invalid values as NaN
    return R

# Parameters
dHat = 0.01
xi = 0.0

# d2 range: start just above x0 to avoid log(0)
d2 = np.linspace(1e-8, square_scalar(0.01), 500)
R = NoKappa_Barrier(d2, dHat, xi)

plt.figure(figsize=(8, 5))
plt.plot(d2, R, label=r'$R$ vs $d^2$')
plt.xlabel(r'$d^2$')
plt.ylabel(r'$R$')
plt.title(r'NoKappa_Barrier: $R$ vs $d^2$ ($\hat{d}=0.01$, $\xi=0$)')
plt.grid(True)
plt.legend()
plt.show()
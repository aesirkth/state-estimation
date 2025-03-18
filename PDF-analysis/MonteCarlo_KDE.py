# The following code is a demonstration of how a probability density function is changed when passed through a nonlinearity.
# This is very important when handling nonlinear processes such as orientational change and pressure to altitude conversion. 
# The monte carlo approach is exact but computationally expensive when a high number of samples are considered
# It can tell us if it is suitable to use the EKF approach or the UKF approach in our application.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Step 1: Generate samples from a normal distribution for X
N = 10000
samples_x = np.random.normal(loc=1, scale=1, size=N)

# Step 2: Transform X to Y using the function f(x) = x^3 (vectorized)
samples_y = samples_x ** 3

# Step 3: Use a Kernel Density Estimator to estimate the PDF of Y
kde = gaussian_kde(samples_y)
y_vals = np.linspace(-10, 10, 1000)  # range over which to evaluate the density
pdf_estimate = kde(y_vals)


# Plot the estimated PDF
plt.plot(y_vals, pdf_estimate, label="Estimated PDF (KDE)")
plt.xlabel("y")
plt.ylabel("Density")
plt.title("PDF of Y = X^3 via KDE")
plt.legend()
plt.show()

# Area under curve
a, b = -1000, 1000
area = np.trapezoid(pdf_estimate, y_vals)

print(area) # This should come out very close to 1 (a trait of PDF:s)
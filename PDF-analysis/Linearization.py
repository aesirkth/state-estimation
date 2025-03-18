import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# This code calculates the change of a PDF when put through a nonlinear function g(x) using variable substitution: X --> Y

# Define the nonlinear function and its derivative
def g(x):
    return x ** 3

def g_prime(x):
    return 3 * x ** 2

# Parameters for the input PDF:
mu_x = 1
sigma_x = 1

x0 = mu_x
g_x0 = g(x0)
gprime_x0 = g_prime(x0)

# Define a range for x. The approximation is best near y = g(x0). This is +- 3 covariances from mean.
x_vals = np.linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 1000)
y_vals = g(x_vals)

# Define the gaussian of variable X (before variable substitution)
pdf_X = norm.pdf(x_vals, loc=mu_x, scale=sigma_x)

# Inverting the linearized relation:
x_in_terms_of_y = x0 + (y_vals - g_x0) / gprime_x0

# Apply the change-of-variables formula:
pdf_Y = norm.pdf(x_in_terms_of_y, loc=mu_x, scale=sigma_x) / abs(gprime_x0)

# Plot the approximated PDF
plt.plot(x_vals, pdf_X, label="PDF of X")
plt.plot(y_vals, pdf_Y, label="PDF of Y (linearized)")
plt.xlabel("y")
plt.ylabel("Density")
plt.title(f"Approximate PDF of Y = X^3 via First-Order Linearization\n(using x0 = {x0}, mu = {mu_x})")
plt.xlim(-10, 10)  # Only show y in [-10, 10]
plt.legend()
plt.show()

# Check normalization of the approximated PDF (should be close to 1)
areaX = np.trapezoid(pdf_X, x_vals)
areaY = np.trapezoid(pdf_Y, y_vals)

print("Area under the PDF of X:", areaX)
print("Area under the PDF of Y:", areaY)

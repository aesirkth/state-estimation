import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# This script creates a comparison between the original PDF and the different methods of approximating
# the PDF after its random variable X has been put through some nonlinear function.

# --- Monte Carlo ---

# Parameters for X ~ N(1, 1)
N = 10000
mu_x = 1
sigma_x = 1

# Generate samples and transform exactly using f(x)=x^3
samples_x = np.random.normal(loc=mu_x, scale=sigma_x, size=N)
samples_y = samples_x ** 3

# Estimate the PDF of Y using KDE
kde = gaussian_kde(samples_y)
# Choose a range for y that covers the region of interest
y_vals = np.linspace(-1000, 1000, 10000)
pdf_monte = kde(y_vals)

# --- First-Order Linearization via Change of Variables ---

# Define the nonlinear function and its derivative
def g(x):
    return x ** 3

def g_prime(x):
    return 3 * x ** 2

# Linearization is performed at x0 = mu_x
x0 = mu_x
g_x0 = g(x0)         # g(1) = 1
gprime_x0 = g_prime(x0)  # g'(1) = 3

# Invert the linearized relation:
# g(x) ≈ g(x0) + g'(x0)*(x - x0)  =>  x ≈ x0 + (y - g(x0)) / g'(x0)
x_in_terms_of_y = x0 + (y_vals - g_x0) / gprime_x0

# Apply the change-of-variables formula:
pdf_linearized = norm.pdf(x_in_terms_of_y, loc=mu_x, scale=sigma_x) / abs(gprime_x0)

# --- Original gaussian PDF ---

x_vals = np.linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 1000)

pdf_original = norm.pdf(x_vals, loc=mu_x, scale=sigma_x)

# --- Verify Normalization ---
area_original = np.trapezoid(pdf_original, x_vals)
area_monte = np.trapezoid(pdf_monte, y_vals)
area_linearized = np.trapezoid(pdf_linearized, y_vals)

print("Area under Original PDF:", area_original)
print("Area under Monte Carlo PDF:", area_monte)
print("Area under linearized PDF:", area_linearized)

# --- Calculate mean Normalization ---
mean_original = np.trapezoid(x_vals * pdf_original, x_vals) / np.trapezoid(pdf_original, x_vals)
mean_monte = np.trapezoid(y_vals * pdf_monte, y_vals) / np.trapezoid(pdf_monte, y_vals)
mean_linearized = np.trapezoid(y_vals * pdf_linearized, y_vals) / np.trapezoid(pdf_linearized, y_vals)

print("Mean of Original PDF:", mean_original)
print("Mean of Monte Carlo PDF:", mean_monte)
print("Mean of Linearized PDF:", mean_linearized)

# --- Plotting Both PDFs on the Same Graph ---

plt.figure(figsize=(8, 6))
plt.plot(y_vals, pdf_monte, label="Estimated PDF(Y) (Monte Carlo)", color="blue")
plt.plot(y_vals, pdf_linearized, label="Approx. PDF(Y) (Linearization)", color="red", linestyle="--")
plt.plot(x_vals, pdf_original, label="Original PDF(X)", color="black", linestyle="-.")
plt.xlabel("y")
plt.ylabel("Density")
plt.title("Comparison of KDE and Linearized PDF for Y = X^3")
plt.xlim(-20, 20)
plt.legend()
plt.show()
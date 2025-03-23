import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


"""
The following code is meant to collect a large sample of sensor readings when the sensor is static and with a true value X.
The code will take N readings from the sensor and create a PDF of it's gaussian output x and give the covariance (noise) and
potential bias. This information can be used when tuning filtering algorithms such as the Kalman Filter.
"""

def get_sensor():
    # Some code that retrieves the sensor reading from a selected sensor.
    # For now lets have a gaussian distribution function here
    return np.random.normal(loc=3, scale=1, size=1).item()

N = 10000 # number of samples
X = 0 # expected value (for static accelerometer = 0)
samples = [] # sample list

# retrieve samples
for i in range(N):
    samples.append(get_sensor())

# Fit a gaussian PDF onto the samples using kernel density estimation
sample_pdf = gaussian_kde(samples)

covariance = np.std(samples)
mean = np.mean(samples)

# print information
print('Noise/Covariance: ', covariance)
print('Bias: ', mean-X)

# Plot the estimated PDF
xs = np.linspace(min(samples), max(samples), N)
plt.plot(xs, sample_pdf(xs), label="Estimated PDF (KDE)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("PDF of sensor samples")
plt.axvline(x=X, color='r', linestyle='--', label='actual value')
plt.axvspan(X, mean, color='red', alpha=0.5, label='bias')
plt.legend()
plt.show()




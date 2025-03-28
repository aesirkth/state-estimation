import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpu6050 import mpu6050
import Adafruit_BMP.BMP085 as BMP085
import time


"""
The following code is meant to collect a large sample of sensor readings when the sensor is static and with a true value X.
The code will take N readings from the sensor and create a PDF of it's gaussian output x and give the covariance (noise) and
potential bias. This information can be used when tuning filtering algorithms such as the Kalman Filter.

The code here has been used on a raspberry pi.
"""

def get_sensor(mpu_sensor, baro_sensor):
    # Some code that retrieves the sensor reading from a selected sensor.
    # For now lets have a gaussian distribution function here
    accel_data = mpu_sensor.get_accel_data()
    gyro_data = mpu_sensor.get_gyro_data()
    
    ax = accel_data['x']
    ay = accel_data['y']
    az = accel_data['z']

    gx = gyro_data['x']
    gy = gyro_data['y']
    gz = gyro_data['z']

    pressure = baro_sensor.read_pressure()

    return ax
    

N = 1000 # number of samples
X = 0 # expected value (for static accelerometer = 0)
samples = [] # sample list

# Initialize the MPU6050 and BMP180 sensor
mpu_sensor = mpu6050(0x68)
baro_sensor = BMP085.BMP085(busnum=1)

counter = 0
# retrieve samples
for i in range(N):
    samples.append(get_sensor(mpu_sensor, baro_sensor))
    time.sleep(0.01)
    counter+=1
    print(counter)

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




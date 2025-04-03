# State-Estimation

This repo contains all software related to state estimation by the avionics team of AESIR.

KF - Kalman Filter

EKF - Extended Kalman Filter

PDF - Probability Density Function

Monte Carlo - "Brute Force approach", uses random sampling to estimate numerical results




The altitude-ekf is an one dimensional altitude EKF and relies on sensor data from a barometer and accelerometer.


The attitude-ekf is an orientational EKF that uses euler angles, it relies on data from an accelerometer and a gyroscope.


The PDF analysis folder is a tool for choosing which version of the kalman filter (EKF or SPKF-"UKF") one should apply for a nonlinear function. The script sensor-pdf-finder stands out, it finds noise and bias given a large amount of random variable samples.


For the filter-tester to give accurate results there needs to be an outlier value handler that can handle barometer going crazy during transonic phase (sonic boom).


Rocketpy simulation needs to be refined, gives much too large apogee
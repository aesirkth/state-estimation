# State-Estimation

This repo contains all software related to state estimation by the avionics team of AESIR.

KF - Kalman Filter

EKF - Extended Kalman Filter

Plain - A loose bit of code meant to be implementet in a larger system (flight computer), can not run itself.

PDF - Probability Density Function

Monte Carlo - "Brute Force approach", uses random sampling to estimate numerical results


The KF is an one dimensional altitude filter and relies on sensor data from a barometer and accelerometer.
The EKF is an orientational kalman filter that uses euler angles, it relies on data from an accelerometer and a gyroscope.
The PDF analysis is central when it comes to noise propagation.

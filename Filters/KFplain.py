import numpy as np

# Default definitions for Q, R, P
accelerometer_noise = 0.3
barometer_noise     = 1.0 # Should be lower than accelerometer, noise needs fixing.
P = np.eye(3) * 0.01
Q = np.eye(3) * (accelerometer_noise**2)
R = np.eye(1) * (barometer_noise**2)

# Constants
g   = 9.81
rho = 1.225
m   = 20
Cd  = 0.45
A   = 0.0186

# A matrix
def get_A(dt):
    return np.array([
        [1, dt, 0],
        [0, 1,  0],
        [0, 0,  0]
    ])

# B matrix
def get_B(dt):
    return np.array([
        [0,       0, 0.5*(dt**2)],
        [0,       0, dt],
        [0,       0, 1]
    ])

# H matrix, note that H is actually nonlinear. This is not correct.
def get_H():
    return np.array([[1, 0, 0]])

# Pressure to altitude conversion
def get_height(pbaro):
    P0 = 101325
    T0 = 288.15
    L  = 0.0065
    R = 287.05
    return np.array([(T0 / L) * (1.0 - (pbaro / P0)**(R * L / g))])

# Prediction step KF
def predict(x, P, Q, u, dt):
    az = -u[2] - g  # net acceleration (subtract gravity)
    A  = get_A(dt)
    B  = get_B(dt)
    x  = A @ x + B @ np.array([[az]])
    P  = A @ P @ A.T + Q
    return x, P

# Correction step KF
def correct(x, P, R, pbaro):
    H = get_H()
    z = get_height(pbaro)
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(3) - K @ H) @ P
    return x, P

# Needs input of Q, R and initial P
def kf_runner(x, P, Q, R, u, pbaro, dt):
    x, P = predict(x, P, Q, u, dt)
    x, P = correct(x, P, R, pbaro)
    return x, P
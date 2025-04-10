import numpy as np

# Default definitions for Q, R, P
accelerometer_noise = 0.3
barometer_noise     = 1.0 
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

# h: Predicts pressure from altitude using barometric formula
def h(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6: base = 1e-6
    return P0 * (base ** (g_const / (R_const * L)))

# H: Jacobian of pressure prediction function
def H(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6: base = 1e-6
    exponent = (g_const / (R_const * L)) - 1.0
    dpdh = -P0 * (g_const / (R_const * T0)) * (base ** exponent)
    return np.array([[dpdh, 0, 0]])

# Prediction step KF
def predict(x, P, Q, u, dt):
    u[2] -= g
    A  = get_A(dt)
    B  = get_B(dt)
    x  = A @ x + B @ u
    P  = A @ P @ A.T + Q
    return x, P

# Correction step KF
def correct(x, P, R, pbaro):
    z = np.array([[pbaro]])  # Use raw pressure directly
    h_est = x[0,0]
    z_pred = h(h_est)
    H_jac = H(h_est)
    
    y = z - np.array([[z_pred]])
    S = H_jac @ P @ H_jac.T + R
    K = P @ H_jac.T @ np.linalg.inv(S)
    
    x = x + K @ y
    P = (np.eye(3) - K @ H_jac) @ P
    return x, P

# Needs input of Q, R and initial P
def kf_runner(x, P, Q, R, u, pbaro, dt):
    x, P = predict(x, P, Q, u, dt)
    x, P = correct(x, P, R, pbaro)
    return x, P

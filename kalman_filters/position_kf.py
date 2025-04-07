import numpy as np

# Default definitions for Q, R, P
# x = [x, y, z, vx, vy, vz, ax, ay, az].T
accelerometer_noise = 0.3
gps_noise     = 1.0 
P = np.eye(9) * 0.01
Q = np.eye(9) * (accelerometer_noise**2)
R = np.eye(3) * (gps_noise**2)

# Constants
g   = 9.81
rho = 1.225
m   = 20
Cd  = 0.45
A   = 0.0186

# A matrix
def get_A(dt):
    A = np.array([
        [1, 0, 0, dt, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    return A

# B matrix
def get_B(dt):
    B = np.array([
        [0.5*(dt**2), 0, 0],
        [0, 0.5*(dt**2), 0],
        [0, 0, 0.5*(dt**2)],
        [dt, 0, 0],
        [0, dt, 0],
        [0, 0, dt],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return B

def get_H():
    R = 6371000 # Radius of earth

    x0 = 0 # meters east of origo (linearized assumption of earth)
    phi0 = 39.3897 # latitude (euroc launchsite)

    y0 = 0 # meters north of origo (linearized assumption of earth)
    theta0 = -8.28896388889 # longitude (euroc launchsite)

    # Spherical coordinates give the following approximation if do the following assumptions:
    # arc length == change in direction in orthogonal coordainte syste, x-y-z where x is east, y is north and z is altitude

    # latitude = phi0 + (x-x0)/(Rcos(theta0))
    # longitude = theta0 + (y-y0)/R
    # altitude = z
    
    H = np.array([
        [1/(R*np.cos(np.deg2rad(theta0))), 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1/R, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ])

    H_offset = np.array([
        [phi0 - x0/(R*np.cos(np.deg2rad(theta0)))],
        [theta0 - y0/R],
        [0] # potential ASL to AGL difference here
    ])

    return H, H_offset

# Prediction step KF
def predict(x, P, Q, u, dt):
    A  = get_A(dt)
    B  = get_B(dt)

    x  = A @ x + B @ u
    P  = A @ P @ A.T + Q
    return x, P

# Correction step KF
def correct(x, P, R, z):
    H, H_offset = get_H()

    y = z - H@x + H_offset
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    x = x + K @ y
    P = (np.eye(9) - K @ H) @ P
    return x, P

# Needs input of Q, R and initial P
def kf_runner(x, P, Q, R, u, z, dt):
    x, P = predict(x, P, Q, u, dt)
    x, P = correct(x, P, R, z)
    return x, P



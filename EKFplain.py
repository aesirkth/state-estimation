# Note that this script only works for objects whos acceleration in global z axis tends to be the same as g=9.81
# (not rocket friendly at the moment)

import numpy as np

# Noise
noise_a = 0.03
noise_g = 0.03

# Default definitions for Q, R, P
P = np.eye(3) * 0.01
Q = np.eye(3) * (noise_g**2)
R = np.eye(3) * (noise_a**2)

g = 9.81

# f function
def f(x, u, dt):
    phi   = x[0].item()
    theta = x[1].item()
    sp = np.sin(phi)
    cp = np.cos(phi)
    ct = np.cos(theta)
    tt = np.tan(theta)

    p = u[0].item()
    q = u[1].item()
    r = u[2].item()

    return x + dt * np.array([
        [p + q*sp*tt + r*cp*tt],
        [q*cp - r*sp],
        [(q*sp / ct) + (r*cp / ct)]
    ])

# F jacobian
def get_F(x, u, dt):
    phi   = x[0].item()
    theta = x[1].item()
    sp = np.sin(phi)
    cp = np.cos(phi)
    ct = np.cos(theta)
    sect  = 1.0 / ct
    sec2t = sect**2
    tt    = np.tan(theta)

    p = u[0].item()
    q = u[1].item()
    r = u[2].item()

    M = np.array([
        [q*cp*tt - r*sp*tt,  q*sp*sec2t + r*cp*sec2t, 0],
        [-q*sp - r*cp,       0,                     0],
        [q*cp*sect - r*sp*sect,
         q*sp*tt*sect + r*cp*tt*sect,              0]
    ])
    return np.eye(3) + dt * M

# h function
def h(x):
    phi   = x[0].item()
    theta = x[1].item()
    psi   = x[2].item()

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    ss = np.sin(psi)
    cs = np.cos(psi)

    R_ = np.array([
        [cs*ct,             -ss*cp + cs*st*sp,   ss*sp + cs*st*cp],
        [ss*ct,              cs*cp + ss*st*sp,  -cs*sp + ss*st*cp],
        [-st,               ct*sp,              ct*cp]
    ])

    g_world = np.array([[0], [0], [g]])
    return R_ @ g_world

# H jacobian
def get_H(x):
    phi   = x[0].item()
    theta = x[1].item()
    psi   = x[2].item()

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    ss = np.sin(psi)
    cs = np.cos(psi)

    H11 = ss*cp - cs*st*sp
    H12 = cs*ct*cp
    H13 = cs*sp - ss*st*cp

    H21 = -cs*cp - ss*st*sp
    H22 = ss*ct*cp
    H23 = ss*sp + cs*st*cp

    H31 = -ct*sp
    H32 = -st*cp
    H33 = 0

    return g * np.array([
        [H11, H12, H13],
        [H21, H22, H23],
        [H31, H32, H33]
    ])

# EKF prediction step
def predict(P, Q, x, u, dt):
    F = get_F(x, u, dt)
    x = f(x, u, dt)
    P = F @ P @ F.T + Q
    for i in range(3):
        x[i] = (x[i] + np.pi) % (2.0*np.pi) - np.pi
    return P, x

# EKF correction step
def correct(P, R, x, z):
    H = get_H(x)
    y = z - h(x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(3) - K @ H) @ P
    return P, x

# Function to run the EKF
def ekf_runner(P, Q, R, x, u, z, dt):
    P, x = predict(P, Q, x, u, dt)
    P, x = correct(P, R, x, z)
    return P, x


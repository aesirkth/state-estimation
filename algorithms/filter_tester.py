import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --------------------------- FILTER START ---------------------------

# Default definitions for Q, R, P
accelerometer_noise = 0.3
barometer_noise     = 100 # noise needs fixing.
P = np.eye(3) * 0.01
Q = np.eye(3) * (accelerometer_noise**2)
R = np.eye(1) * (barometer_noise**2)
x = np.array([[0], [0], [0]])

# Constants
g   = 9.81
rho = 1.225
m   = 20
A   = 0.0186

# Drag Coefficient
def Cd_function(): # assumption: angle of attack = 0
    simulation = pd.read_excel('GitHubAesir\\algorithms\\cd_simulation.xlsx') 
    velocity_vals = simulation['m/s'].tolist()
    CD_vals = simulation['CD'].tolist()

    f_interp = interp1d(velocity_vals, CD_vals, kind='linear')  # 'linear' interpolation

    return f_interp

Cd_f = Cd_function()

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
        [0.5*(dt**2)],
        [dt],
        [1]
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
    az = np.array([-u[1] - g])  # net acceleration (subtract gravity)

    A  = get_A(dt)
    B  = get_B(dt)
    x  = A @ x + B @ az

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

# --------------------------- FILTER END ---------------------------

# Function to extract data
def get_test_data(): 
    flight_data = pd.read_excel('GitHubAesir\\algorithms\\signyflight2023.xlsx') 
    ax_lst = flight_data['ax'].tolist()
    ay_lst = flight_data['ay'].tolist()
    az_lst = flight_data['az'].tolist()

    pressure_lst = flight_data['pressure'].tolist()

    return ax_lst, ay_lst, az_lst, pressure_lst

    
def main(x, P, Q, R):

    x0_lst = []
    x1_lst = []
    x2_lst = []

    ax_lst, ay_lst, az_lst, pressure_lst = get_test_data()

    for i in range(len(ax_lst)):
        u = np.array([[ax_lst[i]], [ay_lst[i]], [az_lst[i]]])

        pressure = pressure_lst[i]
        dt = 0.01

        x, P = kf_runner(x, P, Q, R, u, pressure, dt)

        x0_lst.append(x[0,0])
        x1_lst.append(x[1,0])
        x2_lst.append(x[2,0])

    horizontal_axis = np.linspace(0, 31, len(x0_lst))

    # PLOT

    # Create a figure with 3 subplots arranged vertically
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    axs[0].plot(horizontal_axis, x0_lst)
    axs[0].set_title('Altitude')

    axs[1].plot(horizontal_axis, x1_lst)
    axs[1].set_title('Velocity')

    axs[2].plot(horizontal_axis, x2_lst)
    axs[2].set_title('Acceleration')

    plt.tight_layout()
    plt.show()

main(x, P, Q, R)
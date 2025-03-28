import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import interp1d

# Default definitions for Q, R, P
accelerometer_noise = 0.3
barometer_noise     = 1.0 # Should be lower than accelerometer, noise needs fixing.
P = np.eye(3) * 0.01
Q = np.eye(3) * (accelerometer_noise**2)
R = np.eye(1) * (barometer_noise**2)

# constants
g   = 9.81
rho = 1.225
m   = 20
A   = 0.0186

# Drag Coefficient
def Cd_function(): # assumption: angle of attack = 0
    simulation = pd.read_excel('state-estimation\\algorithms\\cd_simulation.xlsx') 
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
        [0,       0, 0.5*(dt**2)],
        [0,       0, dt],
        [0,       0, 1]
    ])

# Prediction step KF
def predict(x, P, Q, u, dt):
    az = -u[2] - g  # net acceleration (subtract gravity)
    A  = get_A(dt)
    B  = get_B(dt)
    x  = A @ x + B @ np.array([[az]])
    P  = A @ P @ A.T + Q
    return x, P

# h: Predicts pressure from altitude using the barometric formula.
#    This function serves as the nonlinear measurement function in the Kalman filter,
#    mapping the state (altitude) to a predicted pressure value.
def h(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6:
        base = 1e-6
    return P0 * (base ** (g_const / (R_const * L)))

# H: Computes the Jacobian of the measurement function h with respect to altitude.
# Compute the Jacobian (measurement matrix) for the nonlinear measurement function.  
# The barometric measurement function that predicts pressure from altitude is given by: 
#     p(h) = P0 * (1 - (L * h) / T0)^(g / (R * L))
def H(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6:
        base = 1e-6
    exponent = (g_const / (R_const * L)) - 1.0
    dpdh = -P0 * (g_const / (R_const * T0)) * (base ** exponent)
    return np.array([[dpdh, 0, 0]])

# correction step
def correct(x, P, R, pbaro):
    z = np.array([[pbaro]])  # raw pressure directly
    h_est = x[0,0]
    z_pred = h(h_est)
    H_jac = H(h_est)
    
    y = z - np.array([[z_pred]])
    S = H_jac @ P @ H_jac.T + R
    K = P @ H_jac.T @ np.linalg.inv(S)
    
    x = x + K @ y
    P = (np.eye(3) - K @ H_jac) @ P
    return x, P

# Apogee estimator
def calculate_apogee(h0, v0, a0, g, rho, Cd_f, A, m): # measurements are 0 as in initial values of the differential
    if a0 < 0 and h0 > 10 and v0 > 5:

        def dvdt(t, v): 
            Cd = Cd_f(v)

            return -np.sign(v) * 0.5 * (rho * Cd * A * v**2) / m - g
        
        def event_v_zero(t, v): return v[0]
        event_v_zero.terminal, event_v_zero.direction = True, -1
        sol = solve_ivp(dvdt, (0, 50), [v0], method='RK45',
                        events=event_v_zero, t_eval=np.linspace(0, 50, 500))
        if sol.t_events[0].size > 0:
            tvz = sol.t_events[0][0]
            idx = sol.t <= tvz
            return h0 + simpson(sol.y[0][idx], sol.t[idx])
    return None

# Needs input of Q, R and initial P
def kf_runner(x, P, Q, R, u, pbaro, dt):
    x, P = predict(x, P, Q, u, dt)
    x, P = correct(x, P, R, pbaro)
    
    current_height = x[0].item()
    current_velocity = x[1].item()
    current_acceleration = x[2].item()

    apogee = calculate_apogee(current_height, current_velocity, current_acceleration, g, rho, Cd_f, A, m)
    return x, P

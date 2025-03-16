import numpy as np
from scipy.integrate import solve_ivp, simpson

# NOTE: Apogee estimator needs proper drag coefficient to work.

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

# Apogee estimator
def calculate_apogee(h0, v0, a0, g, rho, Cd, A, m): # measurements are 0 as in initial values of the differential
    if a0 < 0 and h0 > 10 and v0 > 5:

        def dvdt(t, v): 
            if v > 5:
                Cd_real = Cd*((90/v0)**0.1)
            else: Cd_real = Cd

            return -np.sign(v) * 0.5 * (rho * Cd_real * A * v**2) / m - g
        
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

    apogee = calculate_apogee(current_height, current_velocity, current_acceleration, g, rho, Cd, A, m)
    return x, P
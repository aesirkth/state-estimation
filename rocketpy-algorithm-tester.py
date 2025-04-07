import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import ode, simpson
from scipy.interpolate import interp1d
from rocketpy import Fluid, CylindricalTank, MassFlowRateBasedTank, HybridMotor
from rocketpy import Environment, Rocket, Flight
import kalman_filters.position_kf
import algorithms.ekf_barometer_apogee_predict

# ====================================================
# Part 1.1: Drag coefficient functions
# ====================================================

def Cd_function_mach():
    simulation = pd.read_excel('algorithms\\cd_simulation.xlsx')
    velocity_vals = simulation['Mach'].tolist()
    CD_vals = simulation['CD'].tolist()
    return interp1d(velocity_vals, CD_vals, kind='linear', fill_value="extrapolate")

Cd_mach = Cd_function_mach()


# ====================================================
# Part 1.2: Create rocket
# ====================================================

# --- Define Fluids and Tank ---
liquid_nox = Fluid(name="lNOx", density=786.6)
vapour_nox = Fluid(name="gNOx", density=159.4)

tank_radius = 150 / 2000  # in meters
tank_length = 0.7
tank_shape = CylindricalTank(tank_radius, tank_length)

burn_time = 7
nox_mass = 7.84
mass_flow = nox_mass / burn_time

oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer tank",
    geometry=tank_shape,
    flux_time=burn_time - 0.01,
    initial_liquid_mass=nox_mass,
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out= mass_flow,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=liquid_nox,
    gas=vapour_nox,
)

# --- Define the Hybrid Motor ---
grain_length = 0.304
nozzle_length = 0.05185
plumbing_length = 0.4859   # including pre-cc and topcap/inj
post_cc_length = 0.0605
pre_cc_length = 0.039

fafnir = HybridMotor(
    thrust_source = 213 * 9.8 * mass_flow,  # using ISP=213
    dry_mass = 24.243,
    dry_inertia = (7.65, 7.65, 0.0845),
    nozzle_radius = 70 / 2000,
    grain_number = 1,
    grain_separation = 0,
    grain_outer_radius = 54.875 / 1000,
    grain_initial_inner_radius = 38.90 / 2000,
    grain_initial_height = grain_length,
    grain_density = 1.1,
    grains_center_of_mass_position = grain_length / 2 + nozzle_length + post_cc_length,
    center_of_dry_mass_position = 0.885,
    nozzle_position = 0,
    burn_time = burn_time,
    throat_radius = 15.875 / 2000,
)

fafnir.add_tank(tank=oxidizer_tank, position=post_cc_length + plumbing_length + grain_length + nozzle_length + tank_length / 2)

# --- Define Environment ---
ground_level = 165
env = Environment(
    latitude=39.3897,
    longitude=-8.28896388889,
    elevation=ground_level,
    date=(2023, 10, 15, 12)
)
env.set_atmospheric_model("custom_atmosphere", wind_u=0, wind_v=-10)

# --- Define the Rocket ---
freya = Rocket(
    radius=0.077,
    mass=10.577,
    inertia=(13, 13, 0.0506),
    power_off_drag=Cd_mach, # Drag!
    power_on_drag=Cd_mach, # Drag!
    center_of_mass_without_motor=2.605,
    coordinate_system_orientation="tail_to_nose"
)

freya.add_motor(fafnir, 0.002)
freya.add_nose(length=0.26, kind="Von Karman", position=3.1556)
freya.add_trapezoidal_fins(4, root_chord=0.2, tip_chord=0.1, span=0.1, position=0.3, sweep_angle=25)

spill_radius = 0.4 / 2
reefed_cd = 0.8
reefed_radius = 1.3 / 2
freya.add_parachute('Reefed', cd_s=reefed_cd * math.pi * (reefed_radius**2 - spill_radius**2), trigger="apogee", lag=3)

main_cd = 0.8
main_radius = 3.8 / 2
freya.add_parachute('Main', cd_s=main_cd * math.pi * (main_radius**2 - spill_radius**2), trigger=200)

# ====================================================
# Part 1.2: Rocketpy simulation & extract values
# ====================================================


# --- Run the Flight Simulation ---
test_flight = Flight(
    rocket=freya,
    environment=env,
    rail_length=12,
    inclination=84,
    heading=0,
    time_overshoot=False,
    terminate_on_apogee=True
)

sol_array = test_flight.solution_array[:, 0]


# Find altitude at maximum velocity (end of burn phase) and apogee.
burnphase_end_index = np.argmax(test_flight.solution_array[:, 6])
sol_array_after_burn = sol_array[burnphase_end_index+1:] # Only after burn phase
z_burnphase_end = test_flight.solution_array[:, 3][burnphase_end_index]
vz_burnphase_end = test_flight.solution_array[:, 6][burnphase_end_index]

# Find the index of maximum altitude (apogee) in the simulation array.
apogee_index = np.argmax(test_flight.solution_array[:, 3])

# Extract the x and y positions at that index.
apogee_x = test_flight.solution_array[apogee_index, 1]  # x position (east)
apogee_y = test_flight.solution_array[apogee_index, 2]  # y position (north)
apogee_z = test_flight.solution_array[apogee_index, 3]  # altitude





# ====================================================
# Part 2: Kalman filter application
# ====================================================

kf = kalman_filters.position_kf

# Default definitions for Q, R, P
# x = [x, y, z, vx, vy, vz, ax, ay, az].T
accelerometer_noise = 0.3
gps_noise     = 1.0 
P = np.eye(9) * 0.01
Q = np.eye(9) * (accelerometer_noise**2)
R = np.eye(3) * (gps_noise**2)
x = initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0] 

# Loop over each time t from the flight simulation
prev_state = None
x_list = []
y_list = []
z_list = []
t_list = []

for t in sol_array:
    state = test_flight.get_solution_at_time(t) # t, x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz

    if prev_state is not None:
        t0 = prev_state[0]
        x0 = prev_state[1]
        y0 = prev_state[2]
        z0 = prev_state[3]
        vx0 = prev_state[4]
        vy0 = prev_state[5]
        vz0 = prev_state[6]

        t1 = state[0]
        x1 = state[1]
        y1 = state[2]
        z1 = state[3]
        vx1 = state[4]
        vy1 = state[5]
        vz1 = state[6]
        dt = t - state[0]
        
        # IMU sensor (simulated sensor from data without noise)
        dt = t1 - t0
        ax = (vx1-vx0) / dt
        ay = (vy1-vy0) / dt
        az = (vz1-vz0) / dt

        u = np.array([[ax], [ay], [az]])

        # GPS sensor (simulated sensor from data without noise)
        lat0 = env.latitude
        long0 = env.longitude
        lat0_rad = np.deg2rad(env.latitude)
        long0_rad = np.deg2rad(env.longitude)

        earth_radius = 6370*1000
        lat = lat0 + (x1-x0)/(earth_radius*np.cos(long0_rad))
        long = long0 + (y1-y0)/earth_radius
        alt = z1
        z = np.array([[lat], [long], [alt]])

        # Run kalman filter
        x, P = kf.kf_runner(x, P, Q, R, u, z, dt)

        x_list.append(x[0][0])
        y_list.append(x[1][0])
        z_list.append(x[2][0])
        t_list.append(t1)

        

    prev_state = test_flight.get_solution_at_time(t)

# ====================================================
# Part 3.2 Plot kalman filter output
# ====================================================


fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t_list, x_list, label='X')
axs[0].set_ylabel('X position (m)')
axs[0].set_title('Kalman Filter Estimated Positions')
axs[0].legend()

axs[1].plot(t_list, y_list, label='Y', color='orange')
axs[1].set_ylabel('Y position (m)')
axs[1].legend()

axs[2].plot(t_list, z_list, label='Z', color='green')
axs[2].set_ylabel('Z position (m)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()

plt.tight_layout()
plt.show(block = False)



# ====================================================
# Part 4.1: Call ballistic simulation for every time in Rocketpy simu
# ====================================================

apogee_predictions = []
apogee_predictions_errors = []

# Loop over each time t from the flight simulation
for t in sol_array_after_burn:
    state = test_flight.get_solution_at_time(t)
    altitude_ASL = state[3]            # z-coordinate (vertical position)
    h0 = altitude_ASL - env.elevation   # altitude above ground level
    v0 = np.linalg.norm(state[4:7])      # magnitude of velocity (vx, vy, vz)
    m_t = freya.total_mass(t)            # rocket mass at time t
    rho_t = env.density(altitude_ASL)     # local air density
    g_const = 9.81

    # Use state indices [x, y, vx, vy] for the ballistic simulation
    initial_state = [state[2], state[3], state[5], state[6]]
    constants = [9.81, env.density(t), freya.total_mass(t), math.pi * freya.radius**2] # [rho, m, g_val, A]

    # Integrate the ballistic trajectory from the current state
    apogee_value = algorithms.ekf_barometer_apogee_predict.integrate_ballistic(t, initial_state, constants, duration=30, dt=0.1)


    apogee_predictions.append(apogee_value)
    apogee_predictions_errors.append(((apogee_value-apogee_z)/apogee_z)*100)

apogee_predictions = np.array(apogee_predictions)
apogee_predictions_errors = np.array(apogee_predictions_errors)


# ====================================================
# Part 4.2: Plot apogee estimates
# ====================================================

print(f'Burn stops at: {z_burnphase_end:.2f} m')
print(f'Maximum velocity: {vz_burnphase_end} m/s')

print(f'x coordinate at apogee: {apogee_x}')
print(f'y coordinate at apogee: {apogee_y}')
print(f'z coordinate at apogee: {apogee_z} (apogee)')

# Plot: Apogee Prediction vs. Time
plt.figure()
plt.plot(sol_array_after_burn, apogee_predictions, 'r-', linewidth=2, label="Ballistic Apogee Prediction")
plt.xlim(0, 50)
plt.ylim(0, 5000)

plt.xlabel("Time (s)")
plt.ylabel("Predicted Apogee (m)")
plt.title("Apogee Prediction vs. Time")
plt.legend()
plt.grid(True)
plt.show(block=False)

# Plot: Apogee Prediction Error vs. Time
plt.figure()
plt.plot(sol_array_after_burn, apogee_predictions_errors, 'r-', linewidth=2, label="Ballistic Apogee Prediction Error")
plt.xlim(0, 50)
plt.ylim(-20, 20)
plt.xlabel("Time (s)")
plt.ylabel("Predicted Apogee Error (%)")
plt.title("Apogee Prediction Error vs. Time")
plt.legend()
plt.grid(True)
plt.show()
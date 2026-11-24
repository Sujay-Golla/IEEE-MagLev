import magpylib as magpy
import numpy as np


# SIMULATION CONFIG
ITERATIONS = 50 # num of sim iterations ()
MESHING = 100
DT = 0.001  # 1 ms time step
g = 9.81

# COIL CONFIG
ROWS = 4
COLS = 4
PITCH = 0.035
COIL_HEIGHT = 0.030
INNER_DIAMETER = 0.0125
OUTER_DIAMETER = 0.025
WINDINGS = 1000  # N coil turns
CURRENT = -3.0
LOOPS = 50

# PLATFORM CONFIG
MAGNET_DIAMETER = 0.0375
MAGNET_HEIGHT = 0.0125
MAGNET_POLARIZATION = 1.2
MAGNET_SEPARATION = 0.056
AIR_GAP = 0.0179 # IMPORTANT: range is 10mm-30mm, good simulation distance i found was 17.9mm
MASS = 0.250
WEIGHT = MASS * g

# pre-sim calcs/variables
ROW_PITCH = PITCH * np.sqrt(3) / 2
TURNS_PER_LOOP = WINDINGS / LOOPS
coils = []

# coil calcs
for row in range(ROWS):
    for col in range(COLS):
        # alternate staggered rows horizontally
        y = row * ROW_PITCH
        x = col * PITCH + (row % 2) * PITCH / 2
        # center coil configuration around origin
        x -= 1.5 * PITCH + PITCH / COLS
        y -= 1.5 * ROW_PITCH
        loops = []
        for i in range(LOOPS):
            z = (i + 0.5) / LOOPS * COIL_HEIGHT
            loop = magpy.current.Circle(
                position=(x, y, z),
                diameter=(INNER_DIAMETER + OUTER_DIAMETER) / 2,
                current=CURRENT * TURNS_PER_LOOP,
            )
            loops.append(loop)
        coils.append(magpy.Collection(*loops))
coil_array = magpy.Collection(*coils)

# pre-sim output
print()
print("SIMULATION SETTINGS")
print("----------------------------------------")
print(f"iterations: {ITERATIONS}")
print(f"meshing: {MESHING}")
print(f"timestep [s]: {DT}")
print(f"standard gravity [m/s^2]: {g}")
print()
print("COIL CONFIGURATION")
print("----------------------------------------")
print(f"Number of coils (#): {len(coils)}")
print(f"Coil height (m): {COIL_HEIGHT}")
print(f"inner diameter (m): {INNER_DIAMETER}")
print(f"outer diameter (m): {OUTER_DIAMETER}")
print(f"number of turns (#): {WINDINGS}")
print(f"numbr of loops (#): {LOOPS}")
print(f"current (A): {CURRENT}")
print()
print("PLATFORM")
print("----------------------------------------")
print(f"magnet diameter (m): {MAGNET_DIAMETER}")
print(f"magnet height (m): {MAGNET_HEIGHT}")
print(f"magnet polarization (T): {MAGNET_POLARIZATION}")
print(f"magnet separation (m): {MAGNET_SEPARATION}")
print(f"air gap (m): {AIR_GAP}")
print(f"mass (kg): {MASS}")
print(f"weight (N): {WEIGHT}")

# sim calcs
MAGNET_Z = COIL_HEIGHT + AIR_GAP + MAGNET_HEIGHT / 2
magnets = []
initial_position = np.array([0.0, 0.0, MAGNET_Z])
position = np.array([0.0, 0.0, MAGNET_Z])
velocity = np.zeros(3)
acceleration = np.zeros(3)
F_net = np.zeros(3)
T = np.zeros(3)
F = np.zeros(3)

print()
print("SIMULATION")
print("----------------------------------------")

for i in range(ITERATIONS):

    # Recreate platform at current position
    magnets = []

    for x in [-MAGNET_SEPARATION / 2,
               MAGNET_SEPARATION / 2]:

        magnet = magpy.magnet.Cylinder(
            position=position + np.array([x, 0, 0]),
            dimension=(MAGNET_DIAMETER, MAGNET_HEIGHT),
            polarization=(0, 0, MAGNET_POLARIZATION),
        )

        magnets.append(magnet)

    platform = magpy.Collection(*magnets)

    # Calculate magnetic forces
    F, T = magpy.getFT(
        coil_array,
        platform,
        meshing=MESHING
    )

    # Include gravity
    F_net = np.asarray(F) + np.array([0, 0, -WEIGHT])

    # Newton's second law
    acceleration = F_net / MASS

    # Euler integration
    velocity += acceleration * DT
    position += velocity * DT

    print()
    print(f"ITERATION: {i + 1}")
    print(f"Force [N]:      {F}")
    print(f"Torque [Nm]:    {T}")
    print(f"Position [m]:   {position}")
    print(f"Velocity [m/s]: {velocity}")

# DEVIATION CALCULATIONS
print()
print("DEVIATION RESULTS")
print("----------------------------------------")

# Position deviation (m)
position_deviation = position - initial_position

# Velocity deviation (m/s)
velocity_deviation = velocity - np.zeros(3)

# Acceleration deviation (m/s^2)
acceleration_deviation = acceleration - np.zeros(3)

# Force deviation (N)
force_deviation = F_net - np.zeros(3)

# Torque deviation (Nm)
torque_deviation = np.asarray(T) - np.zeros(3)

# Print individual components
print(f"Position deviation (m): {position_deviation}")
print(f"Velocity deviation (m/s): {velocity_deviation}")
print(f"Acceleration deviation (m/s^2): {acceleration_deviation}")
print(f"Force deviation (N): {force_deviation}")
print(f"Torque deviation (Nm): {torque_deviation}")

# Magnitudes (Euclidean norm)
print()
print("DEVIATION MAGNITUDES")
print("----------------------------------------")
print(f"Position deviation (m): {np.linalg.norm(position_deviation)}")
print(f"Velocity deviation (m/s): {np.linalg.norm(velocity_deviation)}")
print(f"Acceleration deviation (m/s^2): {np.linalg.norm(acceleration_deviation)}")
print(f"Force deviation (N): {np.linalg.norm(force_deviation)}")
print(f"Torque deviation (Nm): {np.linalg.norm(torque_deviation)}")

# for i in range(ITERATIONS):
#     print()
#     print(f"**ITERATION {i + 1}")
#     print("----------------------------------------")
#
#     # levitated platform calcs
#     MAGNET_Z = COIL_HEIGHT + AIR_GAP + MAGNET_HEIGHT / 2
#     magnets = []
#
#     for x in [-MAGNET_SEPARATION / 2, MAGNET_SEPARATION / 2]:
#         magnet = magpy.magnet.Cylinder(
#             position=(x, 0, MAGNET_Z),
#             dimension=(MAGNET_DIAMETER, MAGNET_HEIGHT),
#             polarization=(0, 0, MAGNET_POLARIZATION),
#         )
#         magnets.append(magnet)
#     platform = magpy.Collection(*magnets)
#
#     # Calculate magnetic force and torque
#     F, T = magpy.getFT(coil_array, platform, meshing=MESHING)
#
#     # output
#     print(f"Forces (N): {F}")
#     print(f"Torques (Nm): {T}")
#     print(f"Net Force in the z direction (N): {np.asarray(F)[2] - WEIGHT}")

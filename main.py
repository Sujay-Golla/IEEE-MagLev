import magpylib as magpy
import numpy as np


# COIL CONFIGURATION
ROWS = 4
COLS = 4
PITCH = 0.035
COIL_HEIGHT = 0.030
INNER_DIAMETER = 0.0125
OUTER_DIAMETER = 0.025
WINDINGS = 1000  # N coil turns
CURRENT = -3.0
LOOPS = 50

ROW_PITCH = PITCH * np.sqrt(3) / 2
TURNS_PER_LOOP = WINDINGS / LOOPS
coils = []

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


# LEVITATED PLATFORM
MAGNET_DIAMETER = 0.0375
MAGNET_HEIGHT = 0.0125
MAGNET_POLARIZATION = 1.2
MAGNET_SEPARATION = 0.056
AIR_GAP = 0.015 # IMPORTANT: range is 10mm-30mm
MESHING = 500

MAGNET_Z = COIL_HEIGHT + AIR_GAP + MAGNET_HEIGHT / 2
magnets = []

for x in [-MAGNET_SEPARATION / 2, MAGNET_SEPARATION / 2]:
    magnet = magpy.magnet.Cylinder(
        position=(x, 0, MAGNET_Z),
        dimension=(MAGNET_DIAMETER, MAGNET_HEIGHT),
        polarization=(0, 0, MAGNET_POLARIZATION),
    )

    magnets.append(magnet)

platform = magpy.Collection(*magnets)

# Calculate magnetic force and torque
F, T = magpy.getFT(coil_array, platform, meshing=MESHING)


# GRAVITY
MASS = 0.250
g = 9.81
WEIGHT = MASS * g


# OUTPUT
print()
print("magpylib SETTINGS")
print("----------------------------------------")
print(f"meshing: {MESHING}")
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
print()
print("RESULTS")
print("----------------------------------------")
print(f"Forces (N): {F}")
print(f"Torques (Nm): {T}")
print(f"Net Force in the z direction (N): {np.asarray(F)[2] - WEIGHT}")

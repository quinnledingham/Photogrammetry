from sympy import *
import numpy as np
import copy
import math

# Notes:
# 1 pixel for Y-Parallax

#focal_length = 158.358 # mm
focal_length = 152.15 # mm

bx, by, bz, omega, phi, kappa, l1, l2, r1, r2, f= symbols('bx by bz phi kappa omega l1 l2 r1 r2 f')

def transformation_matrix():
    # Create rotation matrix elements
    m11 = (cos(phi)*cos(kappa))
    m12 = (-cos(omega)*sin(kappa))+(sin(omega)*sin(phi)*cos(kappa))
    m13 = (sin(omega)*sin(kappa))+(cos(omega)*sin(phi)*cos(kappa))
    
    m21 = (cos(phi)*sin(kappa))
    m22 = (cos(omega)*cos(kappa))+(sin(omega)*sin(phi)*sin(kappa))
    m23 = (-sin(omega)*cos(kappa))+(cos(omega)*sin(phi)*sin(kappa))
    
    m31 = (-sin(phi))
    m32 = (sin(omega)*cos(phi))
    m33 = (cos(omega)*cos(phi))
    
    # Populate Matrix
    M = Matrix([
        [m11, m12, m13, bx],
        [m21, m22, m23, by],
        [m31, m32, m33, bz],
        [  0,    0,   0, 1]
    ])

    return M

def rotation_matrix_3d(axis, angle):
    R = eye(3)
    if axis == 0:  # x-axis
        R[1, 1] = cos(angle)
        R[1, 2] = -sin(angle)
        R[2, 1] = sin(angle)
        R[2, 2] = cos(angle)
    elif axis == 1:  # y-axis
        R[0, 0] = cos(angle)
        R[0, 2] = sin(angle)
        R[2, 0] = -sin(angle)
        R[2, 2] = cos(angle)
    elif axis == 2:  # z-axis
        R[0, 0] = cos(angle)
        R[0, 1] = -sin(angle)
        R[1, 0] = sin(angle)
        R[1, 1] = cos(angle)
    return R

def evaluate(eq, baseline, parameters, left, right):
    new_eq = eq.subs({
        bx: baseline,
        f: focal_length,
        l1: left[0],
        l2: left[1],
        r1: right[0],
        r2: right[1],
        by: parameters[0],
        bz: parameters[1],
        omega: parameters[2],
        phi: parameters[3],
        kappa: parameters[4]
    })
    
    return float(N(new_eq))

def relative_orientation(left, right, baseline):
    parameters = np.array([0, 0, 0, 0, 0]) # by, bz, omega, phi, kappa

    transformation_M = rotation_matrix_3d(2, kappa) * rotation_matrix_3d(1, phi) * rotation_matrix_3d(0, omega)

    print("transformation matrix:")
    pprint(transformation_M)
    right_image = Matrix([r1, r2, -f])
    right_model = transformation_M@right_image
    print("right coords:")
    pprint(right_model)

    mis = Matrix([
        [bx, by, bz],
        [l1, l2, -f],
        [right_model[0], right_model[1], right_model[2]]
    ])
    delta = mis.det()
    pprint(delta)

    partial_derivatives = []
    partial_derivatives.append(diff(delta, by))
    partial_derivatives.append(diff(delta, bz))
    partial_derivatives.append(diff(delta, omega))
    partial_derivatives.append(diff(delta, phi))
    partial_derivatives.append(diff(delta, kappa))

    print("PARTIAL")
    print(delta)

    corrections = np.array([0, 0, 0, 0, 0])

    iterations = 0

    while(1):
        A = []
        misclosure_vector = []
        
        for i in range(len(left)):
            A_row = []
            for p in partial_derivatives:
                value = evaluate(p, baseline, parameters, left[i], right[i])
                A_row.append(value)
            A.append(A_row)

            misclosure_vector.append(evaluate(delta, baseline, parameters, left[i], right[i]))

        A = np.matrix(A)

        corrections = -(np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector))
        print(misclosure_vector)
        parameters = parameters + np.array(corrections).ravel()
        iterations = iterations + 1

        # Replace your current convergence check
        rel_corrections = np.abs(corrections / (np.abs(parameters) + 1e-10))
        if np.max(rel_corrections) < 1e-10:
            break

    pprint(delta)

    parameters = np.array(parameters).ravel()
    omega_deg = parameters[2] * 180 / math.pi
    phi_deg = parameters[3] * 180 / math.pi
    kappa_deg = parameters[4] * 180 / math.pi

    print(f"parameters: [{parameters[0]}, {parameters[1]}, {omega_deg}, {phi_deg}, {kappa_deg}] ({iterations})")

base_distance = 92.0 # mm

# Verification

test_data_left = [
    [106.399, 90.426], # 30
    [18.989, 93.365],  # 40
    [70.964, 4.907],   # 72
    [-0.931, -7.284],  # 127
    [9.278, -92.926],  # 112
    [98.681, -62.769]  # 50
]

test_data_right = [
    [24.848, 81.824],  # 30
    [-59.653, 88.138], # 40
    [-15.581, -0.387], # 72 
    [-85.407, -8.351], # 127
    [-78.81, -92.62],  # 112
    [8.492, -68.873]   # 50
]

relative_orientation(test_data_left, test_data_right, base_distance)
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

def rotation_matrix():
    """ Compute the 3D rotation matrix for given omega, phi, and kappa (in radians). """

    # Rotation about X-axis (omega)
    R_x = Matrix([
        [1, 0, 0],
        [0, cos(rad(omega)), -sin(rad(omega))],
        [0, sin(rad(omega)), cos(rad(omega))]
    ])
    
    # Rotation about Y-axis (phi)
    R_y = Matrix([
        [cos(rad(phi)), 0, sin(rad(phi))],
        [0, 1, 0],
        [-sin(rad(phi)), 0, cos(rad(phi))]
    ])
    
    # Rotation about Z-axis (kappa)
    R_z = Matrix([
        [cos(rad(kappa)), -sin(rad(kappa)), 0],
        [sin(rad(kappa)), cos(rad(kappa)), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z * R_y * R_x
    return R

def evaluate(eq, baseline, parameters, left, right):
    new_eq = eq.subs(bx, baseline)

    new_eq = new_eq.subs(f, focal_length)

    new_eq = new_eq.subs(l1, left[0])
    new_eq = new_eq.subs(l2, left[1])
    new_eq = new_eq.subs(r1, right[0])
    new_eq = new_eq.subs(r2, right[1])

    new_eq = new_eq.subs(by, parameters[0])
    new_eq = new_eq.subs(bz, parameters[1])
    new_eq = new_eq.subs(omega, parameters[2])
    new_eq = new_eq.subs(phi, parameters[3])
    new_eq = new_eq.subs(kappa, parameters[4])

    return new_eq.evalf()

def relative_orientation(left, right, baseline):
    parameters = [0, 0, 0, 0, 0] # by, bz, omega, phi, kappa

    transformation_M = rotation_matrix()
    print("transformation matrix:")
    pprint(transformation_M)
    right_image = Matrix([r1, r2, -f])
    right_model = transformation_M*right_image
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

    corrections = np.ones(5)

    iterations = 0

    while(1):
        A = []
        misclosure_vector = []

        if type(parameters) is not list:
            parameters = np.array(parameters).ravel()

        for i in range(len(left)):
            A_row = []
            for p in partial_derivatives:
                value = evaluate(p, baseline, parameters, left[i], right[i])
                A_row.append(value)
            A.append(A_row)

            misclosure_vector.append(evaluate(delta, baseline, parameters, left[i], right[i]))

        A = np.matrix(A).astype(np.float64)

        corrections = -np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector).astype(np.float64)
        print(misclosure_vector)
        parameters = parameters + corrections
        iterations = iterations + 1

        done = true
        for c in np.array(corrections).ravel():
            if c >= 0.0000000001:
                done = false

        if done: break

    pprint(delta)

    parameters = np.array(parameters).ravel()
    print(f"parameters: {parameters} ({iterations})")

    print(f"OMEGA: {parameters[2] * 180 / math.pi}")
    print(f"OMEGA: {parameters[3] * 180 / math.pi}")
    print(f"OMEGA: {parameters[4] * 180 / math.pi}")


base_distance = 92 # mm

# Verification

test_data_left = [
    [106.399, 90.426], # 30
    [18.989, 93.365], # 40
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
    [8.492, -68.873]    # 50
]

relative_orientation(test_data_left, test_data_right, base_distance)
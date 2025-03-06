from sympy import *
import numpy as np
import copy
import math

# Notes:
# 1 pixel for Y-Parallax

#focal_length = 158.358 # mm
focal_length = 152.150 # mm

bx, by, bz, omega, phi, kappa, l1, l2, r1, r2, f= symbols('bx by bz phi kappa omega l1 l2 r1 r2 f')

def transformation_matrix():
    # Create rotation matrix elements
    m11 = (cos(phi)*cos(kappa))
    m12 = ((sin(omega)*sin(phi)*cos(kappa)) - (cos(omega)*sin(kappa)))
    m13 = ((cos(omega)*sin(phi)*cos(kappa)) + (sin(omega)*sin(kappa)))
    
    m21 = (cos(phi)*sin(kappa))
    m22 = ((sin(omega)*sin(phi)*sin(kappa)) + (cos(omega)*cos(kappa)))
    m23 = ((cos(omega)*sin(phi)*sin(kappa)) - (sin(omega)*cos(kappa)))
    
    m31 = (-sin(phi))
    m32 = (sin(omega)*cos(phi))
    m33 = (cos(omega)*cos(phi))
    
    # Populate Matrix
    M = Matrix([
        [m11, m12, m13, 0],
        [m21, m22, m23, 0],
        [m31, m32, m33, 0],
        [  0,   0,   0, 1]
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
    
    return float(N(new_eq, 50))

def relative_orientation(left, right, baseline):
    parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # by, bz, omega, phi, kappa

    transformation_M = transformation_matrix()
    #transformation_M = rotation_matrix_3d(2, kappa) * (rotation_matrix_3d(1, phi) * rotation_matrix_3d(0, omega))

    print("transformation matrix:")
    pprint(transformation_M)
    right_image = Matrix([r1, r2, -f, 1])
    right_model = transformation_M @ right_image
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
    print(partial_derivatives[2])

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

        rel_corrections = np.abs(corrections / (np.abs(parameters) + 1e-10))
        if np.max(rel_corrections) < 1e-10:
            w = np.array(misclosure_vector)
            apv = (w.T @ w) / (6 - 4)
            print(f"Correlation: {apv * np.linalg.inv(A.T @ A)}")
            break

    #pprint(delta)

    parameters = np.array(parameters).ravel()
    omega_deg = parameters[2] * 180 / math.pi
    phi_deg = parameters[3] * 180 / math.pi
    kappa_deg = parameters[4] * 180 / math.pi

    print(f"parameters: [{parameters[0]}, {parameters[1]}, {omega_deg}, {phi_deg}, {kappa_deg}] ({iterations})")

    return parameters

def space_intersection_point(left, right, model_right, base):

    rx, ry, rz, lx, ly, lz = symbols('rx ry rz lx ly lz')
    delta_i = (bx*rz-bz*rx) / (lx*rz+f*rx)
    micro_i = (-bx*f-bz*lx) / (lx*rz+f*rx)

    new_delta_i = delta_i.subs({
        bx: base[0],
        bz: base[2],
        f: focal_length,
        lx: left[0],
        rx: model_right[0],
        rz: model_right[2]
    })

    new_micro_i = micro_i.subs({
        bx: base[0],
        bz: base[2],
        f: focal_length,
        lx: left[0],
        rx: model_right[0],
        rz: model_right[2]
    })

    d_i = N(new_delta_i)
    m_i = N(new_micro_i)

    new_lx = d_i * left[0]
    new_ly = d_i * left[1]
    new_lz = -d_i * focal_length

    new_rx = m_i * model_right[0] + base[0]
    new_ry = m_i * model_right[1] + base[1]
    new_rz = m_i * model_right[2] + base[2]

    test_bx = d_i*left[0] - m_i*model_right[0]
    test_bz = -d_i*focal_length -m_i*model_right[2]
    #print(f"BX: {test_bx}")

    y_parallax = new_ry - new_ly
    print(f"Y_PARALLAX: { y_parallax }")
    if not math.isclose(new_lx, new_rx): print(f"XmL ({new_lx}) != XmR ({new_rx})")
    final_y = (new_ly + new_ry) / 2
    if not math.isclose(new_lz, new_rz): print(f"ZmL ({new_lz}) != ZmR ({new_rz})")
    return [new_lx, final_y, new_lz]

def space_intersection(left, right, parameters, baseline):
    # apply the transformation matrix to the right image points
    transformation_M = transformation_matrix()
    right_image = Matrix([r1, r2, -f, 1])
    right_model = transformation_M @ right_image

    model_right = []

    pprint(right_model[0])
    for i in range(len(right)):
        new_eq = right_model.subs({
            f: focal_length,
            r1: right[i][0],
            r2: right[i][1],
            bx: baseline,
            by: parameters[0],
            bz: parameters[1],
            omega: parameters[2],
            phi: parameters[3],
            kappa: parameters[4]
        })
        result = N(new_eq)
        model_right.append([float(result[0]), float(result[1]), float(result[2])])

    print(model_right)

    model_coordinates = []
    for i in range(len(left)):
        model_coordinates.append(space_intersection_point(left[i], right[i], model_right[i], [baseline, parameters[0], parameters[1]]))

    print(model_coordinates)

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

p_parameters = relative_orientation(test_data_left, test_data_right, base_distance)
#print(p_parameters)
target_parameters = [5.0455, 2.1725, math.radians(0.4392),  math.radians(1.508), math.radians(3.1575)]
space_intersection(test_data_left, test_data_right, p_parameters, base_distance)
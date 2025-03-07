from sympy import *
import numpy as np
import copy
import math
import pandas as pd

from docx import Document

# Notes:
# 1 pixel for Y-Parallax

# create equations
bx, by, bz, omega, phi, kappa, l1, l2, r1, r2, f= symbols('bx by bz phi kappa omega l1 l2 r1 r2 f')

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
rotation_matrix = Matrix([
    [m11, m12, m13, 0],
    [m21, m22, m23, 0],
    [m31, m32, m33, 0],
    [  0,   0,   0, 1]
])

right_image = Matrix([r1, r2, -f, 1])
right_prime = rotation_matrix @ right_image

# Misclosure equation
mis = Matrix([
    [bx, by, bz],
    [l1, l2, -f],
    [right_prime[0], right_prime[1], right_prime[2]]
])

mis_det = mis.det()

# Partial derivates for design matrix
partial_derivatives = []
partial_derivatives.append(diff(mis_det, by))
partial_derivatives.append(diff(mis_det, bz))
partial_derivatives.append(diff(mis_det, omega))
partial_derivatives.append(diff(mis_det, phi))
partial_derivatives.append(diff(mis_det, kappa))

# Unique scale factors
rx, ry, rz, lx, ly, lz = symbols('rx ry rz lx ly lz')
lambda_i = (bx*rz-bz*rx) / (lx*rz+f*rx)
mu_i = (-bx*f-bz*lx) / (lx*rz+f*rx)

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

def design_matrix(baseline, parameters, left, right):
    A = []
    misclosure_vector = []
    
    for i in range(len(left)):
        A_row = []
        for p in partial_derivatives:
            value = evaluate(p, baseline, parameters, left[i], right[i])
            A_row.append(float(value))
        A.append(A_row)

        misclosure_vector.append(evaluate(mis_det, baseline, parameters, left[i], right[i]))

    return np.matrix(A), misclosure_vector

def relative_orientation(left, right, baseline):
    parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # by, bz, omega, phi, kappa

    iterations = 0

    while(1):
        A, misclosure_vector = design_matrix(baseline, parameters, left, right)

        corrections = -(np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector))
        parameters = parameters + np.array(corrections).ravel()
        iterations = iterations + 1

        rel_corrections = np.abs(corrections / (np.abs(parameters) + 1e-10))
        if np.max(rel_corrections) < 1e-10:
            A, misclosure_vector = design_matrix(baseline, parameters, left, right)

            cov = np.linalg.inv(A.T @ A)
            std_devs = np.sqrt(np.diag(cov))  # Standard deviations of parameters
            corr_matrix = cov / np.outer(std_devs, std_devs)
            print("Correlation Matrix of Estimated Parameters:\n", corr_matrix)
            break

    parameters = np.array(parameters).ravel()

    # convert rads to degs to display
    omega_deg = parameters[2] * 180 / math.pi
    phi_deg = parameters[3] * 180 / math.pi
    kappa_deg = parameters[4] * 180 / math.pi

    print(f"parameters: [{parameters[0]}, {parameters[1]}, {omega_deg}, {phi_deg}, {kappa_deg}] ({iterations})")

    return parameters

def space_intersection_point(left, right, base):
    l_i = N(lambda_i.subs({
        bx: base[0],
        bz: base[2],
        f: focal_length,
        lx: left[0],
        rx: right[0],
        rz: right[2]
    }))

    m_i = N(mu_i.subs({
        bx: base[0],
        bz: base[2],
        f: focal_length,
        lx: left[0],
        rx: right[0],
        rz: right[2]
    }))

    new_lx =  l_i * left[0]
    new_ly =  l_i * left[1]
    new_lz = -l_i * focal_length

    new_rx = m_i * right[0] + base[0]
    new_ry = m_i * right[1] + base[1]
    new_rz = m_i * right[2] + base[2]

    # Calculating Y_Parallax
    y_parallax = new_ry - new_ly
    print(f"Y_PARALLAX: { y_parallax }")

    if not math.isclose(new_lx, new_rx): print(f"XmL ({new_lx}) != XmR ({new_rx})")
    mean_y = (new_ly + new_ry) / 2
    if not math.isclose(new_lz, new_rz): print(f"ZmL ({new_lz}) != ZmR ({new_rz})")

    return [new_lx, mean_y, new_lz]

def space_intersection(left, right, baseline, parameters):
    # apply the transformation matrix to the right image points
    model_right = []
    for i in range(len(right)):
        result = N(right_prime.subs({
            f: focal_length,
            r1: right[i][0],
            r2: right[i][1],
            bx: baseline,
            by: parameters[0],
            bz: parameters[1],
            omega: parameters[2],
            phi: parameters[3],
            kappa: parameters[4]
        }))
        model_right.append([float(result[0]), float(result[1]), float(result[2])])

    model_coordinates = []
    for i in range(len(left)):
        model_coordinates.append(space_intersection_point(left[i], model_right[i], [baseline, parameters[0], parameters[1]]))

    print(model_coordinates)

doc = Document()

base_distance = 92.0 # mm
focal_length = 158.358 # mm

class Data:
    def __init__(self):
        self.control = []
        self.tie = []
        self.index_map = {}

df = pd.read_excel('data.xlsx')

# Define control point IDs
control_points = {100, 102, 104, 105, 200, 201, 202, 203}

image_27 = Data()
image_28 = Data()

for index, row in df.iterrows():
    point_id, x27, y27, x28, y28 = row.to_list()

    if isinstance(point_id, int) and point_id in control_points:
        # It's a control point
        image_27.index_map[point_id] = len(image_27.control)
        image_28.index_map[point_id] = len(image_28.control)

        image_27.control.append((x27, y27))
        image_28.control.append((x28, y28))
    elif isinstance(point_id, str) and point_id.startswith("T"):
        # It's a tie point
        image_27.index_map[point_id] = len(image_27.tie)
        image_28.index_map[point_id] = len(image_28.tie)

        image_27.tie.append((x27, y27))
        image_28.tie.append((x28, y28))

parameters = relative_orientation(image_27.tie, image_28.tie, base_distance)
space_intersection(image_27.tie, image_28.tie, base_distance, parameters)
space_intersection(image_27.control, image_28.control, base_distance, parameters)

# Verification
focal_length = 152.150 # mm

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
space_intersection(test_data_left, test_data_right, base_distance, p_parameters)
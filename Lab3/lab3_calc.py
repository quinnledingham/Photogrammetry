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
            array_to_word_table(corr_matrix.tolist(), "Correlation Matrix", float, decimals=4)
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

    if not math.isclose(new_lx, new_rx): print(f"XmL ({new_lx}) != XmR ({new_rx})")
    mean_y = (new_ly + new_ry) / 2
    if not math.isclose(new_lz, new_rz): print(f"ZmL ({new_lz}) != ZmR ({new_rz})")

    return [new_lx, mean_y, new_lz], y_parallax

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

    y_parallaxs = []
    model_coordinates = []
    for i in range(len(left)):
        coordinates, y_parallax = space_intersection_point(left[i], model_right[i], [baseline, parameters[0], parameters[1]])
        model_coordinates.append(coordinates)
        y_parallaxs.append(y_parallax)

    array_to_word_table([y_parallaxs], "Y Parallax", float, decimals=4)

    print(f"Y Parallaxs {y_parallaxs}")
    print(f"Model Coordinates {model_coordinates}")
    return model_coordinates

# print a 2d array to a word table
def array_to_word_table(array, name, value_type, decimals=2):
    doc.add_paragraph(name)
    table = doc.add_table(rows=len(array), cols=len(array[0]))

    for i, row in enumerate(array):
        for j, cell in enumerate(row):
            table.cell(i, j).text = str(round(cell, decimals))

doc = Document()

base_distance = 92.0 # mm
focal_length = 158.358 # mm

class Data:
    def __init__(self):
        self.control = []
        self.tie = []
        self.index_map = {}

# Define the data
data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 100, 102, 104, 105, 200, 201, 202, 203, "T1", "T2", "T3", "T4", "T5", "T6"],
    "x1": [-106, 106.004, -106.002, 106.017, -111.995, 112.003, 0.011, 0.004, -9.59, -2.413, 18.943, 90.379, 18.149, 44.598, -7.657, 52.691, -10.105, 94.369, -10.762, 90.075, -9.489, 85.42],
    "y1": [-106.004, 105.999, 106.006, -105.982, -0.002, -0.018, 112.01, -112.01, 96.218, -5.995, -81.815, -91.092, 109.575, 7.473, -49.112, -93.178, 15.011, -4.092, -104.711, -91.378, 96.26, 103.371],
    "x2": [-105.991, 106.011, -106.011, 106.004, -111.999, 112.003, 0.015, 0.01, -105.469, -95.081, -72.547, -1.357, -77.826, -48.846, -98.855, -38.936, -103.829, 0.868, -100.169, -1.607, -105.395, -9.738],
    "y2": [-105.995, 106.006, 105.996, -106.003, -0.003, -0.006, 112.007, -112.005, 98.736, -4.848, -79.764, -86.95, 113.405, 10.131, -48.068, -90.079, 16.042, -0.022, -103.14, -87.253, 98.706, 109.306]
}

df = pd.DataFrame(data)

#df = pd.read_excel('data.xlsx')

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

print("\n\nLab Data\n")
parameters = relative_orientation(image_27.tie, image_28.tie, base_distance)
array_to_word_table([parameters], "Parameters", float, decimals=4)

tie_points = space_intersection(image_27.tie, image_28.tie, base_distance, parameters)
array_to_word_table(tie_points, "Tie Points", float, decimals=4)

control_points = space_intersection(image_27.control, image_28.control, base_distance, parameters)
array_to_word_table(control_points, "Control Points", float, decimals=4)

# Verification
print("\n\nTest Data\n")

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

# 
test_parameters = relative_orientation(test_data_left, test_data_right, base_distance)
array_to_word_table([test_parameters], "Test Parameters", float, decimals=4)

# used to test if the space intersection is working
target_parameters = [5.0455, 2.1725, math.radians(0.4392),  math.radians(1.508), math.radians(3.1575)]

test_points = space_intersection(test_data_left, test_data_right, base_distance, test_parameters)
array_to_word_table(test_points, "Test Points", float, decimals=4)

# printing answers out to a word document
doc.save("output.docx")
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from docx import Document

def load_data(data, low, high):
    load = []
    for i in range(low, high):
        same = []
        for j, value in enumerate(data[i]):
            if j % 2 == 0:
                coords = [data[i][j], data[i][j+1]]
                same.append(coords)
        load.append(same)
    return load

def right_hand(a, image_dim):
    output = []
    for coords in a:
        x = coords[0] - (image_dim[0]/2)
        y = (image_dim[1]/2) - coords[1]
        output.append([x, y])
    return output

def right_hand_array(a, image_dim):
    output = []
    for elements in a:
        output.append(right_hand(elements, image_dim))
    return output

def mean(a):
    sum = [0, 0]
    for v in a:
        if not math.isnan(v[0]):
            sum[0] = sum[0] + v[0]
        if not math.isnan(v[1]):
            sum[1] = sum[1] + v[1]
    sum[0] = sum[0] / len(a)
    sum[1] = sum[1] / len(a)
    return sum

def std(a):
    m = mean(a)
    sum = [0, 0]
    for v in a:
        if not math.isnan(v[0]):
            sum[0] = sum[0] + (v[0] - m[0])**2
        if not math.isnan(v[1]):
            sum[1] = sum[1] + (v[1] - m[1])**2
    sum[0] = math.sqrt(sum[0]/(len(a) - 1))
    sum[1] = math.sqrt(sum[1]/(len(a) - 1))
    return sum

def print_array(a):
    for v in a:
        print(v)

doc = Document()

def array_to_word_table(array, name, type):
    doc.add_paragraph(name)
    table = doc.add_table(rows=len(array), cols=len(array[0])*2)

    for i, row in enumerate(array):
        for j, cell in enumerate(row):
            index = j * 2
            table.cell(i, index).text = str(type(round(cell[0], 2)))
            table.cell(i, index + 1).text = str(type(round(cell[1], 2)))

def pool(input, image_dim, out):
    data = {key: 0 for key in input.keys()}

    for key in input:
        input[key] = right_hand_array(input[key], image_dim)
        array_to_word_table(input[key], f"{key}_{out}", int)
        print(f"{key}")
        print(input[key])

    for i, key in enumerate(input):
        a = input[key]
        values = []
        ms = []
        stds = []
        print(f"{key}:")
        for v in a:
            m = mean(v)
            s = std(v)
            print(f"mean: {m}, std: {s}")
            values.append(m)
            ms.append([m])
            stds.append([s])

        data[key] = values

        array_to_word_table(ms, f"mean_{key}_{out}", float)
        array_to_word_table(stds, f"std_{key}_{out}", float)

    return data
    
def design_matrix(fiducial):
    n = len(fiducial)
    A = np.zeros((2 * n, 6))

    for i in range(n):
        x, y = fiducial[i]
        A[2*i, 0] = x
        A[2*i, 1] = y
        A[2*i, 2] = 1
        A[2*i + 1, 3] = x
        A[2*i + 1, 4] = y
        A[2*i + 1, 5] = 1

    return A

def residual_plot(og, residuals):
    print("\n\nPlot")

    og = np.array(og)
    r = np.array(r)

    plt.figure(figsize=(8, 6))

    for i in range(len(og)):
        plt.arrow(og[i, 0], og[i, 1], r[i, 0], r[i, 1], head_width=2.5, head_length=5.5, color='red')

    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    #plt.axhline(y=0, color='black', linewidth=0.7, linestyle='--')
    plt.legend()
    plt.show()

def RMS(a):
    sum = 0
    for v in a:
        sum = sum + v**2
    
    return math.sqrt(sum/len(a))

# data is the fiducials in a [[]]
def affine_transformation(data, t_data):
    l = []
    for coords in t_data:
        l.append(coords[0])
        l.append(coords[1])

    l = np.array(l)
    A = design_matrix(data)

    x_hat = np.linalg.inv(A.T @ A) @ A.T @ l
    print(f"x_hat: {x_hat}")
    # a b x c d y

    residuals = A @ x_hat - l
    print(f"residuals: {residuals}")

    output = []
    for coords in data:
        x = (x_hat[0] * coords[0] + x_hat[1] * coords[1]) + x_hat[2]
        y = (x_hat[3] * coords[0] + x_hat[4] * coords[1]) + x_hat[5]
        output.append([x, y])

    # convert 1D-array into 2D-array
    r = []
    for i in range(len(residuals)):
        if i % 2 == 0:
            r.append([residuals[i], residuals[i + 1]])

    #residual_plot(t_data, residuals)

    r = np.array(r)
    print(f"RMS x: {RMS(r[:, 0])}, y: {RMS(r[:, 1])}")

    return output

df = pd.read_excel('data.xlsx')

data = []
for index, row in df.iterrows():
    data.append(row.to_list())

og_27 = {
    "fiducials": load_data(data, 0, 8),
    "control": load_data(data, 16, 24),
    "tie": load_data(data, 32, 38),
}

og_28 = {
    "fiducials": load_data(data, 8, 16),
    "control": load_data(data, 24, 32),
    "tie": load_data(data, 38, 44),
}

image_dim_27 = [20448, 20480]
image_dim_28 = [20462, 20494]

# part b

data_27 = pool(og_27, image_dim_27, "27")
data_28 = pool(og_28, image_dim_28, "28")

doc.save("build/output.docx")

# part c

# fiducial marks from certificate
reseaux = [
    [-105.997, -105.995],
    [106.004, 106.008],
    [-106.000, 106.009],
    [106.012, -105.995],
    [-112.000, 0.007],
    [112.006, 0.007],
    [0.005, 112.007],
    [0.002, -111.998]
]

print("\nimage 27:")
print(data_27["fiducials"])
print(affine_transformation(data_27['fiducials'], reseaux))

print("\nimage 28")
print(data_28['fiducials'])
print(affine_transformation(data_28['fiducials'], reseaux))

# part d

test_data = [
    [-113.767,	-107.400],
    [-43.717,	-108.204],
    [36.361,	-109.132],
    [106.408,	-109.923],
    [107.189,	-39.874],
    [37.137,	-39.070],
    [-42.919,	-38.158],
    [-102.968,	-37.446],
    [-112.052,	42.714],
    [-42.005,	41.903],
    [38.051,	40.985],
    [108.089,	40.189],
    [108.884,	110.221],
    [38.846,	111.029],
    [-41.208,	111.961],
    [-111.249,	112.759],
]

test_data_reseaux = [
    [-110,	-110],
    [-40,	-110],
    [40,	-110],
    [110,	-110],
    [110,	-40],
    [40,	-40],
    [-40,	-40],
    [-100,	-40],
    [-110,	40],
    [-40,	40],
    [40,	40],
    [110,	40],
    [110,	110],
    [40,	110],
    [-40,	110],
    [-110,	110],
]
print("\nlecture data:")
print(affine_transformation(test_data, test_data_reseaux))

# part e

ppo = [-0.006, 0.006] # mm

k1 = -0.1528E-7
k2 = 0.5256E-12
k3 = 0

p1 = 0.1346E-6
p2 = 0.1224E-7

f = 0.153358 # focal length (m)

def principal_point_offest(a):
    for coords in a:
        coords[0] = (coords[0] - ppo[0])
        coords[1] = (coords[1] - ppo[1])

def radial_lens_correction(a):
    for coords in a:
        x, y = coords
        r = math.sqrt(x**2 + y**2)
        corr = ((k1 * r**2) + (k2 * r**4) + (k3 * r**6))
        x_rad = x * corr
        y_rad = y * corr

        coords[0] = x + x_rad
        coords[1] = y + y_rad

def decentering_lens_correction(a):
    for coords in a:
        x, y = coords
        r = math.sqrt(x**2 + y**2)
        x_dec = p1 * (r**2 + 2 * x**2) + 2 * p2 * x * y
        y_dec = p2 * (r**2 + 2 * y**2) + 2 * p1 * x * y

        coords[0] = x + x_dec
        coords[1] = y + y_dec

# a = [], h = [], H = int
def atmospheric_refraction_correction(a, h, H):
    for coords in a:
        K = ((2410 * H)/(H**2 - 6*H + 250)) - ((2410*h)/(h**2 - 6*h + 250)) * (h/H)
        x, y = coords
        r = math.sqrt(x**2 + y**2)
        x_atm = x * K * (1 + (r**2/f**2))
        y_atm = y * K * (1 + (r**2/f**2))

object_coords = [
    [-399.28, -679.72, 1090.96],
    [109.70, -642.35, 1086.43],
    [475.55, -538.18, 1090.50],
    [517.62, -194.43, 1090.65],
    [-466.39, -542.31, 1091.55],
    [42.73, -412.19, 1090.82],
    [321.09, -667.45, 1083.49],
    [527.78, -375.72, 1092.00]
]

avg_h = 0
for coords in object_coords:
    x, y, z = coords
    avg_h = avg_h + z
avg_h = avg_h / len(object_coords)

print(avg_h)

'''
def dist_3D(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

longest_dist = 0
for a in object_coords:
    for b in object_coords:
        if a is b:
            continue
        dist = dist_3D(a, b)
        if dist > longest_dist:
            longest_dist = dist
            print(f"A: {a}, B: {b}")
'''
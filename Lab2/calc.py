import pandas as pd
import math
import numpy as np

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

def pool(input, image_dim):
    data = {key: 0 for key in input.keys()}

    for key in input:
        input[key] = right_hand_array(input[key], image_dim)

    for i, key in enumerate(input):
        a = input[key]
        values = []
        stds = []
        for v in a:
            values.append(mean(v))
            stds.append(std(v))
        #print(stds)
        data[key] = values

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

data_27 = pool(og_27, image_dim_27)
data_28 = pool(og_28, image_dim_28)

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
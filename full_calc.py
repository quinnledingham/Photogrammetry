import sys
import pandas as pd
import numpy as np

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './Lab2')
sys.path.insert(2, './Lab3')
sys.path.insert(2, './Lab4')

import lab2_calc
import lab3_calc
import lab4_calc

# Lab 2
print("\n\nCorrections\n")
df = pd.read_excel('./Lab2/data.xlsx')

data = []
for index, row in df.iterrows():
    data.append(row.to_list())

og_27 = {
    "fiducials": lab2_calc.load_data(data, 0, 8),
    "control": lab2_calc.load_data(data, 16, 24),
    "tie": lab2_calc.load_data(data, 32, 38),
}

og_28 = {
    "fiducials": lab2_calc.load_data(data, 8, 16),
    "control": lab2_calc.load_data(data, 24, 32),
    "tie": lab2_calc.load_data(data, 38, 44),
}

camera = lab2_calc.Camera()

images = [
    lab2_calc.Image(camera, "image_27", 20448, 20480, og_27),
    lab2_calc.Image(camera, "image_28", 20462, 20494, og_28)
]

object_coords = np.array([
    [-399.28, -679.72, 1090.96], # 100
    [109.70, -642.35, 1086.43],  # 102
    [475.55, -538.18, 1090.50],  # 104
    [517.62, -194.43, 1090.65],  # 105
    [-466.39, -542.31, 1091.55], # 200
    [42.73, -412.19, 1090.82],   # 201
    [321.09, -667.45, 1083.49],  # 202
    [527.78, -375.72, 1092.00]   # 203
])

avg_h = 0
for coords in object_coords:
    x, y, z = coords
    avg_h = avg_h + z
avg_h = avg_h / len(object_coords)

flying_height = 751.4637599031875 # flying height from Lab 1
H = flying_height + avg_h

for i in images: 
    i.pool()
    i.get_affine_transformation_parameters()
    i.apply_transformation()
    i.apply_corrections(avg_h, H)

# Lab 3
print("\n\nRelative Orientation\n")

print(np.array(images[1].tie))
rel_ori = lab3_calc.Relative_Orientation(images[0].tie, images[1].tie, 92, camera.f)

model_tie_points = np.array(rel_ori.space_intersection(images[0].tie, images[1].tie))
model_control_points = np.array(rel_ori.space_intersection(images[0].control, images[1].control))

# Lab 4
print("\n\nAbsolute Orientation\n")

selected_indices = [1, 2, 3, 4] # what indices are control points
control_points = model_control_points[selected_indices]
control_object_coords = object_coords[selected_indices]
check_points = np.delete(model_control_points, selected_indices, axis=0)

abs_ori = lab4_calc.Absolute_Orientation("full calc", control_points, control_object_coords, rel_ori.right_pc, rel_ori.rotation_parameters)

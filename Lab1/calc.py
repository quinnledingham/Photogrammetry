import math

def subtract(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return [x, y]

def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

class Line:
    def __init__(self, slope, point):
        self.m = slope
        self.b = -slope * point[0] + point[1]

    @staticmethod
    def intersection(line_a, line_b):
        x = (line_b.b - line_a.b) / (line_a.m - line_b.m)
        y = (line_b.m * line_a.b - line_a.m * line_b.b) / (line_b.m - line_a.m)
        return [x, y]


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

fiducial_pixels = [
    [1346, 19285],
    [19170,	1478],
    [1353, 1472],
    [19163,	19291],
    [846, 10378],
    [19670,	10385],
    [10262,	971],
    [10254,	19792],
]

image_dim = [20456, 20488]

fiducial = []

for coords in fiducial_pixels:
    x = coords[0] - (image_dim[0]/2)
    y = (image_dim[1]/2) - coords[1]
    fiducial.append([x, y])

cali_dists = [
    299.816,
    299.825,
    224.005,
    224.006,
] # mm

dists = []
pixel_spacing = 0
for i in range(4):
    cali_dist = cali_dists[i]
    it = i * 2
    pix_dist = dist(fiducial_pixels[it + 1], fiducial_pixels[it])
    dists.append(cali_dist / pix_dist)
    pixel_spacing += cali_dist / pix_dist
pixel_spacing = pixel_spacing / 4

print(pixel_spacing) # mm / px

for coords in fiducial:
    coords[0] = pixel_spacing * coords[0]
    coords[1] = pixel_spacing * coords[1]

print(fiducial)

m1_2 = slope(fiducial[1], fiducial[0])
m3_4 = slope(fiducial[3], fiducial[2])

print(m1_2)
print(m3_4)

line_1 = Line(m1_2, fiducial[0])
line_2 = Line(m3_4, fiducial[2])

fiducial_center = Line.intersection(line_1, line_2)
print(fiducial_center)

ppo = [-0.006, 0.006] # mm

for coords in fiducial:
    coords[0] = (coords[0] - fiducial_center[0]) - ppo[0]
    coords[1] = (coords[1] - fiducial_center[1]) - ppo[1]

print(fiducial)

cfl = 153.358 # mm
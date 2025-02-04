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

def right_hand(a, image_dim):
    output = []
    for coords in a:
        x = coords[0] - (image_dim[0]/2)
        y = (image_dim[1]/2) - coords[1]
        output.append([x, y])
    return output

def scale(a, scale):
    for coords in a:
        coords[0] = scale * coords[0]
        coords[1] = scale * coords[1]

def reduce(a, f_c, ppo):
    for coords in a:
        coords[0] = (coords[0] - f_c[0]) - ppo[0]
        coords[1] = (coords[1] - f_c[1]) - ppo[1]

def mm_to_m(a):
    for coords in a:
        coords[0] = coords[0] * 1E-3
        coords[1] = coords[0] * 1E-3


def pixel_to_image(a, image_dim, pixel_spacing, fiducial_center, principal_point_offset):
    a = right_hand(a, image_dim)
    scale(a, pixel_spacing)
    reduce(a, fiducial_center, principal_point_offset)
    return a

# part a: Fiducial mark measurement and refinement

cali_dists = [
    299.816,
    299.825,
    224.005,
    224.006,
] # mm

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

# calculate mm / px aka pixel spacing
dists = []
pixel_spacing = 0
for i in range(4):
    cali_dist = cali_dists[i]
    it = i * 2
    pix_dist = dist(fiducial_pixels[it + 1], fiducial_pixels[it])
    dists.append(cali_dist / pix_dist)
    pixel_spacing += cali_dist / pix_dist
pixel_spacing = pixel_spacing / 4 # mm / px

fiducial = right_hand(fiducial_pixels, image_dim)

scale(fiducial, pixel_spacing)

m1_2 = slope(fiducial[1], fiducial[0])
m3_4 = slope(fiducial[3], fiducial[2])

line_1 = Line(m1_2, fiducial[0])
line_2 = Line(m3_4, fiducial[2])

fiducial_center = Line.intersection(line_1, line_2)

ppo = [-0.006, 0.006] # mm

reduce(fiducial, fiducial_center, ppo)

# part b: Flying height estimation

cp = [
    [4544, 17489], # 300, px
    [2546, 5779]   # 304, px
]

rlg_cp = [
    [497.49, -46.90, 1090.56],   # 300, m
    [-186.74, -151.54, 1093.12]  # 304, m
]

cp = right_hand(cp, image_dim)
scale(cp, pixel_spacing)
reduce(cp, fiducial_center, ppo)

cfl = 153.358 # mm

cp_dist = dist(cp[0], cp[1]) * 1E-3   # m
rlg_dist = dist(rlg_cp[0], rlg_cp[1]) # m 

S = rlg_dist / cp_dist

flying_height = S * (cfl * 1E-3) # m

print(flying_height)

# part c: Building height estimation

ict = [
    [6148, 11608], # Top, px
    [6353, 11559]  # Bottom, px
]

#ict = right_hand(ict, image_dim)
#scale(ict, pixel_spacing)
#reduce(ict, fiducial_center, ppo)

ict = pixel_to_image(ict, image_dim, pixel_spacing, fiducial_center, ppo)

rt = dist(ict[0], [0, 0])
rb = dist(ict[1], [0, 0])

h = ((flying_height * 1E3) * (rt - rb)) / rt # mm

print(h * 1E-3)
import math

def subtract(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return [x, y]

def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

fiducial_pixels = [
    [1346, 19284],
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
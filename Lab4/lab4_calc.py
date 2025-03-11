# Comment on accuracy
#   Any sort of consistent error

from sympy import *
import numpy as np
import math

# define equations
b_omega, b_phi, b_kappa, b_lambda, tx, ty, tz = symbols('b_omega, b_phi, b_kappa, b_lambda, tx, ty, tz')
mx, my, mz = symbols('mx, my, mz')

# Create rotation matrix elements
m11 = (cos(b_phi)*cos(b_kappa))
m12 = ((sin(b_omega)*sin(b_phi)*cos(b_kappa)) + (cos(b_omega)*sin(b_kappa)))
m13 = (-(cos(b_omega)*sin(b_phi)*cos(b_kappa)) + (sin(b_omega)*sin(b_kappa)))

m21 = (-cos(b_phi)*sin(b_kappa))
m22 = (-(sin(b_omega)*sin(b_phi)*sin(b_kappa)) + (cos(b_omega)*cos(b_kappa)))
m23 = ((cos(b_omega)*sin(b_phi)*sin(b_kappa)) + (sin(b_omega)*cos(b_kappa)))

m31 = (sin(b_phi))
m32 = (-sin(b_omega)*cos(b_phi))
m33 = (cos(b_omega)*cos(b_phi))

# Populate Matrix
rotation_matrix = Matrix([
    [m11, m12, m13],
    [m21, m22, m23],
    [m31, m32, m33],
])

O = b_lambda * rotation_matrix * Matrix([mx, my, mz]) + Matrix([tx, ty, tz])
xO = O[0]
yO = O[1]
zO = O[2]

partial_derivatives = []
for o in O:
    row = []
    row.append(diff(o, b_omega))
    row.append(diff(o, b_phi))
    row.append(diff(o, b_kappa))
    row.append(diff(o, b_lambda))
    row.append(diff(o, tx))
    row.append(diff(o, ty))
    row.append(diff(o, tz))
    partial_derivatives.append(row)

pprint(partial_derivatives[0][0])

class Absolute_Data:
    def __init__(self, model, object):
        self.model = model
        self.object = object

def evaluate(eq, parameters, point):
    new_eq = eq.subs({
        b_omega: parameters[0],
        b_phi: parameters[1], 
        b_kappa: parameters[2], 
        b_lambda: parameters[3], 
        tx: parameters[4], 
        ty: parameters[5], 
        tz: parameters[6],
        mx: point[0],
        my: point[1],
        mz: point[2]
    })
    
    return float(N(new_eq, 50))

def misclosure(parameters, model, object):
    w = [
        evaluate(xO, parameters, model) - object[0],
        evaluate(yO, parameters, model) - object[1],
        evaluate(zO, parameters, model) - object[2]
    ]
    return w

# points [x, y, z]
def design_matrix(parameters, points, object_points):
    A = []
    misclosure_vector = []
    for point_index in range(len(points)):
        for coord_index in range(3):
            A_row = []
            
            for p in partial_derivatives[coord_index]:
                value = evaluate(p, parameters, points[point_index])
                A_row.append(value)

            A.append(A_row)
        misclosure_vector += misclosure(parameters, points[point_index], object_points[point_index])
    return np.matrix(A), misclosure_vector

def r3(angle):
    R = np.array([[np.cos(angle), np.sin(angle), 0],
                 [-np.sin(angle), np.cos(angle), 0],
                 [0, 0, 1]])
    
    return R

def inital_approx(o1, o2, m1, m2):
    ao = atan((o2[0] - o1[0])/(o2[1] - o1[1]))
    am = atan((m2[0] - m1[0])/(m2[1] - m1[1]))

    kappao = ao - am
    
    Mo = r3(float(kappao))

    delta_xo = o2[0] - o1[0]
    delta_yo = o2[1] - o1[1]
    delta_zo = o2[2] - o1[2]

    delta_xm = m2[0] - m1[0]
    delta_ym = m2[1] - m1[1]
    delta_zm = m2[2] - m1[2]

    do = math.sqrt(delta_xo**2 + delta_yo**2 + delta_zo**2)
    dm = math.sqrt(delta_xm**2 + delta_ym**2 + delta_zm**2)

    lambdao = do / dm

    to = o1.T - (lambdao * (Mo @ o2.T))

    parameters = np.array([0.0, 0.0, kappao, lambdao, to[0], to[1], to[2]]) # omega, phi, kappa, lambda, tx, ty, tz
    print(f"Initial parameters: {parameters}")
    return parameters

def deg_parameters(p):
    p[0] = p[0] * 180 / math.pi
    p[1] = p[1] * 180 / math.pi
    p[2] = p[2] * 180 / math.pi
    return p


def absolute_orientation(model_coordinates, object_coordinates):
    parameters = inital_approx(np.array(object_coordinates[0]), np.array(object_coordinates[1]), np.array(model_coordinates[0]), np.array(model_coordinates[1]))

    while(1):
        A, misclosure_vector = design_matrix(parameters, model_coordinates, object_coordinates)

        corrections = -(np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector))
        parameters = parameters + np.array(corrections).ravel()

        rel_corrections = np.abs(corrections / (np.abs(parameters) + 1e-10))
        if np.max(rel_corrections) < 1e-3:
            print(deg_parameters(parameters))
            A, misclosure_vector = design_matrix(parameters, model_coordinates, object_coordinates)
            break

model_coordinates = [

    [108.9302,  92.5787, -155.7696], # 30
    [ 19.5304,  96.0258, -156.4878], # 40
    [ 71.8751,   4.9657, -154.1035], # 72
    [ -0.9473,  -7.4078, -154.8060], # 127
    [  9.6380, -96.5329, -158.0535], # 112
    [100.4898, -63.9177, -154.9389], # 50
]

object_coordinates = [
    [7350.27,	4382.54,	276.42], # 30
    [6717.22,	4626.41,	280.05], # 40
    [6869.09,	3844.56,	283.11], # 72
    [6316.06,	3934.63,	283.03], # 127
    [6172.84,	3269.45,	248.10], # 112
    [6905.26,	3279.84,	266.47], # 50
]

test_data = Absolute_Data(model_coordinates, object_coordinates)

absolute_orientation(model_coordinates, object_coordinates)
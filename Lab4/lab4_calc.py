# Comment on accuracy
#   Any sort of consistent error

from sympy import *
import numpy as np
import math
import copy

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

def rad_parameters(p):
    dx = p[0] * math.pi / 180
    dy = p[1] * math.pi / 180
    dz = p[2] * math.pi / 180
    return [dx, dy, dz]

def deg_parameters(p):
    dx = p[0] * 180 / math.pi
    dy = p[1] * 180 / math.pi
    dz = p[2] * 180 / math.pi
    return [dx, dy, dz]

class Absolute_Data:
    def __init__(self, model, object):
        self.model = model
        self.object = object

        self.inital_approx(np.array(self.object[0]), np.array(self.object[1]), np.array(self.model[0]), np.array(self.model[1]))
        self.estimate_parameters()
        self.set_deg_parameters()
        print(f"Final parameters {self.deg_parameters}")

        self.get_ground_coordinates()
        print(self.transformed_points)

        print(self.extract_angles([0.4392, 1.508, 3.1575], self.parameters))

    # points [x, y, z]
    def design_matrix(self):
        A = []
        misclosure_vector = []
        for point_index in range(len(self.model)):
            for coord_index in range(3):
                A_row = []
                
                for p in partial_derivatives[coord_index]:
                    value = self.evaluate(p, self.model[point_index])
                    A_row.append(value)

                A.append(A_row)
            misclosure_vector += self.misclosure(self.model[point_index], self.object[point_index])
        return np.matrix(A), misclosure_vector

    # does the LSE to get parameters for absolute orientation
    def estimate_parameters(self):
        iterations = 0
        while(1):
            A, misclosure_vector = self.design_matrix()

            corrections = -(np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector))
            self.parameters = self.parameters + np.array(corrections).ravel()

            rel_corrections = np.abs(corrections / (np.abs(self.parameters) + 1e-10))
            iterations += 1
            if np.max(rel_corrections) < 1e-3:
                print(f"Iterations: {iterations}")
                A, misclosure_vector = self.design_matrix()
                break

    def evaluate(self, eq, point):
        new_eq = eq.subs({
            b_omega: self.parameters[0],
            b_phi: self.parameters[1], 
            b_kappa: self.parameters[2], 
            b_lambda: self.parameters[3], 
            tx: self.parameters[4], 
            ty: self.parameters[5], 
            tz: self.parameters[6],
            mx: point[0],
            my: point[1],
            mz: point[2]
        })
        
        return float(N(new_eq, 50))

    def misclosure(self, model_coord, object_coord):
        w = [
            self.evaluate(xO, model_coord) - object_coord[0],
            self.evaluate(yO, model_coord) - object_coord[1],
            self.evaluate(zO, model_coord) - object_coord[2]
        ]
        return w

    def inital_approx(self, o1, o2, m1, m2):
        ao = atan((o2[0] - o1[0])/(o2[1] - o1[1]))
        am = atan((m2[0] - m1[0])/(m2[1] - m1[1]))

        kappao = ao - am
        
        # r3
        Mo = np.array([[np.cos(float(kappao)),  np.sin(float(kappao)), 0],
                    [-np.sin(float(kappao)), np.cos(float(kappao)), 0],
                    [0, 0, 1]])

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

        self.parameters = np.array([0.0, 0.0, kappao, lambdao, to[0], to[1], to[2]]) # omega, phi, kappa, lambda, tx, ty, tz
        print(f"Initial parameters: {self.parameters}")

    def get_ground(self, coord):
        x = self.evaluate(xO, coord)
        y = self.evaluate(yO, coord)
        z = self.evaluate(zO, coord)
        return [x, y, z]

    def get_ground_coordinates(self):
        self.transformed_points = []
        for coord in self.model:
            self.transformed_points.append(self.get_ground(coord))

    def evaluate_rotation(self, parameters):
        new_eq = rotation_matrix.subs({
            b_omega: parameters[0],
            b_phi: parameters[1], 
            b_kappa: parameters[2], 
        })
        
        return N(new_eq, 50)

    def set_deg_parameters(self):
        self.deg_parameters = copy.copy(self.parameters)
        self.deg_parameters[0] *= 180 / math.pi
        self.deg_parameters[1] *= 180 / math.pi
        self.deg_parameters[2] *= 180 / math.pi

    def get_angle(self, M):
        print(M)
        omega = np.degrees(np.arctan2(-M[2, 1], M[2, 2]))
        phi = np.degrees(np.arcsin(M[2, 0]))
        kappa = np.degrees(np.arctan2(-M[1, 0], M[0, 0]))
        return [omega, phi, kappa]

    # extract angles from the relative parameters and the absolutae parmeters
    # the angles for going from image to object space
    def extract_angles(self, rel_p, abs_p):
        M_i_m_left = self.evaluate_rotation(rad_parameters([0, 0, 0]))
        M_i_m_right = self.evaluate_rotation(rad_parameters(rel_p)) # relative orientation rotation parameters
        M_o_m = self.evaluate_rotation(abs_p)
        M_i_o_left = M_i_m_left * M_o_m.T
        M_i_o_right = M_i_m_right * M_o_m.T

        angles = []
        angles.append(self.get_angle(np.array(M_i_o_left).astype(np.float64)))
        angles.append(self.get_angle(np.array(M_i_o_right).astype(np.float64)))
        return angles

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

#test_parameters = absolute_orientation(model_coordinates, object_coordinates)
#print(get_ground_coordinates(test_parameters, model_coordinates))
print(test_data.get_ground([92, 5.0455, 2.1725]))

#print(extract_angles([0.4392, 1.508, 3.1575], test_parameters))
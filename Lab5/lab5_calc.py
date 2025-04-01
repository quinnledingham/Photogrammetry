from sympy import *
import numpy as np
import math

Xc, Yc, Zc, omega, phi, kappa = symbols('Xc, Yc, Zc, omega, phi, kappa')
Xi, Yi, Zi, a, ix, b, iy, tx, ty, c, Xp, Yp = symbols('Xi, Yi, Zi, a, ix, b, iy, tx, ty, c, Xp, Yp')

# Create rotation matrix elements
m11 = (cos(phi)*cos(kappa))
m12 = ((sin(omega)*sin(phi)*cos(kappa)) + (cos(omega)*sin(kappa)))
m13 = (-(cos(omega)*sin(phi)*cos(kappa)) + (sin(omega)*sin(kappa)))

m21 = (-cos(phi)*sin(kappa))
m22 = (-(sin(omega)*sin(phi)*sin(kappa)) + (cos(omega)*cos(kappa)))
m23 = ((cos(omega)*sin(phi)*sin(kappa)) + (sin(omega)*cos(kappa)))

m31 = (sin(phi))
m32 = (-sin(omega)*cos(phi))
m33 = (cos(omega)*cos(phi))

U = (m11*(Xi-Xc)+m12*(Yi-Yc)+m13*(Zi-Zc))
V = (m21*(Xi-Xc)+m22*(Yi-Yc)+m23*(Zi-Zc))
W = (m31*(Xi-Xc)+m32*(Yi-Yc)+m33*(Zi-Zc))

xij = Xp - c * (U/W)
yij = Yp - c * (V/W)

xij_bar, yij_bar = symbols('xij_bar, yij_bar')

def get_partial(o):
    row = []
    row.append(diff(o, Xc))
    row.append(diff(o, Yc))
    row.append(diff(o, Zc))
    row.append(diff(o, omega))
    row.append(diff(o, phi))
    row.append(diff(o, kappa))
    return row

partial_derivatives = []
partial_derivatives.append(get_partial(xij))
partial_derivatives.append(get_partial(yij))

def get_redundancy_numbers(A):
    # Compute hat matrix and redundancy numbers
    H = A @ np.linalg.inv(A.T @ A) @ A.T
    redundancy_numbers = 1 - np.diag(H)  # Redundancy numbers for each observation

    num_rows = len(A)
    num_cols = redundancy_numbers.size // num_rows
    redundancy_table = redundancy_numbers.reshape(num_rows, num_cols)

    s = np.sum(redundancy_numbers)
    print(f"Redundancy Numbers: {redundancy_numbers} Sum: {s}")
    #array2d_to_word_table(redundancy_table.tolist(), f"{self.name} Redundancy Numbers", decimals=4)

class Resection():

    def __init__(self, image, object, focal_length, variance):
        self.image = image
        self.object = object
        self.focal_length = focal_length
        self.variance = variance

        self.parameters = [
            6338.6,  # Xc (m)
            3984.6,  # Yc (m)
            1453.1,  # Zc (m)
            0,       # w (dd)
            0,       # p (dd)
            np.radians(-18.854)  # k (dd)
        ]

        self.estimate_parameters()
        print(f"Estimated Parameters: {self.parameters}")

    def estimate_parameters(self):
        iterations = 0
        while(1):
            A, misclosure_vector = self.design_matrix()

            corrections = -(np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector))
            self.parameters = self.parameters + np.array(corrections).ravel()

            rel_corrections = np.abs(corrections / (np.abs(self.parameters) + 1e-10))
            iterations += 1
            if np.max(rel_corrections) < 1e-3: # if converged
                print(f"Iterations: {iterations}")
                A, misclosure_vector = self.design_matrix()
                print(f"Residuals: {np.array(misclosure_vector)}")

                break

    def design_matrix(self):
        A = []
        misclosure_vector = []
        for point_index in range(len(self.image)):
            for coord_index in range(2):
                A_row = []
                
                for p in partial_derivatives[coord_index]:
                    value = self.evaluate(p, self.object[point_index])
                    A_row.append(value)

                A.append(A_row)
            misclosure_vector += self.misclosure(self.object[point_index], self.image[point_index])
        return np.matrix(A), misclosure_vector

    def misclosure(self, object_coord, image_coord):
        w = [
            self.evaluate(xij, object_coord) - image_coord[0],
            self.evaluate(yij, object_coord) - image_coord[1],
        ]
        return w

    def evaluate(self, eq, point):
        new_eq = eq.subs({
            Xc: self.parameters[0],
            Yc: self.parameters[1], 
            Zc: self.parameters[2], 
            omega: self.parameters[3], 
            phi: self.parameters[4], 
            kappa: self.parameters[5], 
            c: self.focal_length,
            Xi: point[0],
            Yi: point[1],
            Zi: point[2],
            Xp: 0,
            Yp: 0,
        })
        
        return float(N(new_eq, 50))
        
class Intersection():
    def evaluate(self, eq, eops, point):
        new_eq = eq.subs({
            Xc: eops[0],
            Yc: eops[1], 
            Zc: eops[2], 
            omega: eops[3], 
            phi: eops[4], 
            kappa: eops[5], 
            c: self.focal_length,
            Xi: point[0],
            Yi: point[1],
            Zi: point[2],
            Xp: 0,
            Yp: 0,
        })
        return float(N(new_eq, 50))

    def get_rad_parameters(self, p):
        dx = p[3] * math.pi / 180
        dy = p[4] * math.pi / 180
        dz = p[5] * math.pi / 180
        return [p[0], p[1], p[2], dx, dy, dz]

    def __init__(self, iops, eops, focal_length):
        self.focal_length = focal_length
        self.left_eops = eops[:, 0]
        self.right_eops = eops[:, 1]

        self.left_eops = self.get_rad_parameters(self.left_eops)
        self.right_eops = self.get_rad_parameters(self.right_eops)

        self.partial_derivatives = []
        self.partial_derivatives.append(self.get_partial(xij))
        self.partial_derivatives.append(self.get_partial(yij))

    def evaluate_b(self, eq, eops, image):
        new_eq = eq.subs({
            Xc:  eops[0],
            Yc:  eops[1], 
            Zc:  eops[2], 
            omega:  eops[3], 
            phi:  eops[4], 
            kappa:  eops[5], 
            c: self.focal_length,
            xij_bar: image[0],
            yij_bar: image[1]
        })
        return np.float64(N(new_eq, 50))

    def approximate_parameters(self):
        a11 = xij_bar*m31 + c*m11
        a12 = xij_bar*m32 + c*m12
        a13 = xij_bar*m33 + c*m13

        a21 = yij_bar*m31 + c*m21
        a22 = yij_bar*m32 + c*m22
        a23 = yij_bar*m33 + c*m23

        eq_A = np.matrix([
            [a11, a12, a13],
            [a21, a22, a23],
            [a11, a12, a13],
            [a21, a22, a23],
        ])

        A = np.zeros((4,3))

        for i in range(3): A[0, i] = self.evaluate_b(eq_A[0, i], self.left_eops, [self.image[0], self.image[1]])
        for i in range(3): A[1, i] = self.evaluate_b(eq_A[1, i], self.left_eops, [self.image[0], self.image[1]])
        for i in range(3): A[2, i] = self.evaluate_b(eq_A[2, i], self.right_eops, [self.image[2], self.image[3]])
        for i in range(3): A[3, i] = self.evaluate_b(eq_A[3, i], self.right_eops, [self.image[2], self.image[3]])

        b1 = (xij_bar*m31+c*m11)*Xc + (xij_bar*m32+c*m12)*Yc + (xij_bar*m33+c*m13)*Zc
        b2 = (yij_bar*m31+c*m21)*Xc + (yij_bar*m32+c*m22)*Yc + (yij_bar*m33+c*m23)*Zc

        b = [
            self.evaluate_b(b1, self.left_eops, [self.image[0], self.image[1]]),
            self.evaluate_b(b2, self.left_eops, [self.image[0], self.image[1]]),
            self.evaluate_b(b1, self.right_eops, [self.image[2], self.image[3]]),
            self.evaluate_b(b2, self.right_eops, [self.image[2], self.image[3]])
        ]

        self.parameters = np.linalg.inv(A.T @ A) @ A.T @ b

    def do_iterations(self, image):
        self.image = image
        self.approximate_parameters()
        print(self.parameters)
        A, w = self.design_matrix()
        print(A)
        print(w)

        iterations = 0
        while(1):
            A, misclosure_vector = self.design_matrix()

            corrections = -(np.linalg.inv(A.T @ A) @ A.T @ np.array(misclosure_vector))
            self.parameters = self.parameters + np.array(corrections).ravel()

            rel_corrections = np.abs(corrections / (np.abs(self.parameters) + 1e-10))
            iterations += 1
            if np.max(rel_corrections) < 1e-3: # if converged
                print(f"Iterations: {iterations}")
                print(f"Parameters: {np.array(self.parameters)}")
                A, misclosure_vector = self.design_matrix()
                print(f"Residuals: {np.array(misclosure_vector)}")
                get_redundancy_numbers(A)
                sigma_x = 15E-3**2 * np.linalg.inv(A.T @ A)
                print(f"STD: {np.sqrt(np.diag(sigma_x))}")
                break

    def get_partial(self, o):
        row = []
        row.append(diff(o, Xc))
        row.append(diff(o, Yc))
        row.append(diff(o, Zc))
        return row
    
    def design_matrix(self):
        A = []
        misclosure_vector = []
        for coord_index in range(4):
            if coord_index == 0 or coord_index == 1:
                eops = self.left_eops
            else:
                eops = self.right_eops

            if coord_index % 2 == 0: # x
                eq = xij
                p_e_index = 0
            else: # y
                eq = yij
                p_e_index = 1
            p_e = self.partial_derivatives[p_e_index]

            A_row = []
            for p in p_e:
                value = -self.evaluate(p, eops, self.parameters)
                A_row.append(value)

            A.append(A_row)
            misclosure_vector += [self.evaluate(eq, eops, self.parameters) - self.image[coord_index]]

        return np.matrix(A), misclosure_vector

def main():
    # test data 
    test_resection_image_points = [
        [106.399, 90.426], # 30
        [18.989,  93.365],  # 40
        [98.681, -62.769], # 50
        [9.278,  -92.926]   # 112
    ] # mm (reduced to PP)

    test_resection_object_points = [
        [7350.27, 4382.54, 276.42],
        [6717.22, 4626.41, 280.05],
        [6905.26, 3279.84, 266.47],
        [6172.84, 3269.45, 248.10]
    ] # m

    #test_res = Resection(test_resection_image_points, test_resection_object_points, 152.150, 15)

    # [x_left, y_left, x_right, y_right]
    test_intersection_image_points = np.array([
        [70.964, 4.907,	-15.581, -0.387], # 72
        [-0.931, -7.284, -85.407, -8.351] # 127
    ]) # mm

    # [Left, Right]
    test_intersection_eops = np.array([
        [6349.488,	7021.897], # Xc (m)		
        [3965.252,	3775.680], # Yc (m)	
        [1458.095,	1466.702], # Zc (m)	
        [0.9885,	1.8734],   # w (dd)		
        [0.4071,	1.6751],   # p (dd)	
        [-18.9049,	-15.7481], # k (dd)	
    ])

    test_inter = Intersection(test_intersection_image_points, test_intersection_eops, 152.150)
    test_inter.do_iterations(test_intersection_image_points[0])
    test_inter.do_iterations(test_intersection_image_points[1])

if __name__ == '__main__':
    main()
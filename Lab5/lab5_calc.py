from sympy import *
import numpy as np

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

xij = Xp - c * ((m11*(Xi-Xc)+m12*(Yi-Yc)+m13*(Zi-Zc))/(m31*(Xi-Xc)+m32*(Yi-Yc)+m33*(Zi-Zc)))
yij = Yp - c * ((m21*(Xi-Xc)+m22*(Yi-Yc)+m23*(Zi-Zc))/(m31*(Xi-Xc)+m32*(Yi-Yc)+m33*(Zi-Zc)))

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

pprint(partial_derivatives[1][4])

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
    def __init__(self):
        pass

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

    test_res = Resection(test_resection_image_points, test_resection_object_points, 152.150, 15)

    # [x_left, y_left, x_right, y_right]
    test_intersection_impage_points = [
        [70.964, 4.907,	-15.581, -0.387], # 72
        [-0.931, -7.284, -85.407, -8.351] # 127
    ] # mm

    # [Left, Right]
    test_intersection_eops = [
        [6349.488,	7021.897], # Xc (m)		
        [3965.252,	3775.680], # Yc (m)	
        [1458.095,	1466.702], # Zc (m)	
        [0.9885,	1.8734],   # w (dd)		
        [0.4071,	1.6751],   # p (dd)	
        [-18.9049,	-15.7481], # k (dd)	
    ]

if __name__ == '__main__':
    main()
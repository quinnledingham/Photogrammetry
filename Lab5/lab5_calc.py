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
        #b1 = (xij - Xp) * W
        #print(evaluate(b1, self.left_eops))

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
        return float(N(new_eq, 50))

    def do_iterations(self, image):
        self.image = image
        self.object = [0, 0, 0] # what we are solving for
        A, w = self.design_matrix()

        b1 = (xij_bar*m31+c*m11)*Xc + (xij_bar*m32+c*m12)*Yc + (xij_bar*m33+c*m13)*Zc
        b2 = (yij_bar*m31+c*m21)*Xc + (yij_bar*m32+c*m22)*Yc + (yij_bar*m33+c*m23)*Zc

        b = [
            self.evaluate_b(b1, self.left_eops, [self.image[0], self.image[1]]),
            self.evaluate_b(b2, self.left_eops, [self.image[0], self.image[1]]),
            self.evaluate_b(b1, self.right_eops, [self.image[2], self.image[3]]),
            self.evaluate_b(b2, self.right_eops, [self.image[2], self.image[3]])
        ]
        print(A)
        #self.object = np.linalg.inv(A.T @ A) @ A.T @ b
        print(b)

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
            A_row = []
            
            p_e_index = 0
            if coord_index % 2 != 0:
                p_e_index = 1
            p_e = self.partial_derivatives[p_e_index]

            for p in p_e:
                value = self.evaluate(p, self.left_eops, self.object)
                A_row.append(value)

            A.append(A_row)
            misclosure_vector += self.misclosure(self.object, self.image[coord_index])
        return np.matrix(A), misclosure_vector

    def misclosure(self, object_coord, image_coord):
        w = [
            self.evaluate(xij, self.left_eops, object_coord) - image_coord,
        ]
        return w


def deg_to_rad(degrees):
    return np.radians(degrees)

def rotation_matrix(omega, phi, kappa):
    """Compute the rotation matrix from omega, phi, kappa (in radians)."""
    cw, cp, ck = np.cos([omega, phi, kappa])
    sw, sp, sk = np.sin([omega, phi, kappa])
    
    R = np.array([
        [cp * ck, sw * sp * ck - cw * sk, cw * sp * ck + sw * sk],
        [cp * sk, sw * sp * sk + cw * ck, cw * sp * sk - sw * ck],
        [-sp, sw * cp, cw * cp]
    ])
    return R

def approximate_intersection(image_points, eops, focal_length=152.0):
    """Compute initial approximations of 3D intersection points using least squares."""
    # Convert angles to radians
    w_rad, p_rad, k_rad = map(deg_to_rad, eops[3:])
    
    # Compute rotation matrices for left and right images
    R_left = rotation_matrix(w_rad[0], p_rad[0], k_rad[0])
    R_right = rotation_matrix(w_rad[1], p_rad[1], k_rad[1])
    
    # Camera centers
    Xc = eops[0]
    Yc = eops[1]
    Zc = eops[2]
    
    # Forming direction vectors from image coordinates
    approx_points = []
    for i in range(image_points.shape[0]):
        xl, yl, xr, yr = image_points[i]
        
        # Image coordinates in homogeneous form
        left_vec = R_left @ np.array([xl, yl, -focal_length])
        right_vec = R_right @ np.array([xr, yr, -focal_length])
        
        # Normalize vectors
        left_vec /= np.linalg.norm(left_vec)
        right_vec /= np.linalg.norm(right_vec)
        
        # Midpoint method for ray intersection
        P1 = np.array([Xc[0], Yc[0], Zc[0]])
        P2 = np.array([Xc[1], Yc[1], Zc[1]])
        
        A = np.vstack([left_vec, -right_vec]).T
        b = P2 - P1
        
        # Solve for lambda and mu
        lambdas = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Compute approximate 3D point
        Xp = P1 + lambdas[0] * left_vec
        
        approx_points.append(Xp)
    
    return np.array(approx_points)



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

    # Example usage
    #approx_3D_points = approximate_intersection(test_intersection_image_points, test_intersection_eops)
    #print("Approximate 3D Points:")
    #print(approx_3D_points)

if __name__ == '__main__':
    main()
from sympy import *
import numpy as np
import copy

# Notes:
# 1 pixel for Y-Parallax

focal_length = 158.358 # mm

bx, by, bz, phi, kappa, omega, l1, l2, r1, r2, f= symbols('bx by bz phi kappa omega l1 l2 r1 r2 f')

def misclosure():

    d_11 = bx
    d_12 = by
    d_13 = bz

    d_21 = l1
    d_22 = l2
    d_23 = -f

    d_31 = cos(phi)*cos(kappa)*r1 + (cos(omega)*sin(kappa) + sin(omega)*sin(phi)*cos(kappa))*r2 + (sin(omega)*sin(kappa) - cos(omega)*sin(phi)*cos(kappa))*f + bx
    d_32 = -(cos(phi)*sin(kappa)*r1) + (cos(omega)*cos(kappa) - sin(omega)*sin(phi)*sin(kappa))*r2 + (sin(omega)*cos(kappa) + cos(omega)*sin(phi)*sin(kappa))*f + by
    d_33 = sin(phi)*r1 + -(sin(omega)*cos(phi)*r2) + cos(omega)*cos(phi)*f + bz

    return [
        [d_11, d_12, d_13],
        [d_21, d_22, d_23],
        [d_31, d_32, d_33]
    ]    

def diff_misclosure(mis, sym):
    result = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    for i in range(len(mis)):
        for j in range(len(mis[0])):
            result[i][j] = diff(mis[i][j], sym)
    return result

def transformation(bx, by, bz, omega, phi, kappa):
    # Create rotation matrix elements
    m11 = np.cos(phi)*np.cos(kappa)
    m12 = (np.cos(omega)*np.sin(kappa))+(np.sin(omega)*np.sin(phi)*np.cos(kappa))
    m13 = (np.sin(omega)*np.sin(kappa))-(np.cos(omega)*np.sin(phi)*np.cos(kappa))
    
    m21 = -(np.cos(phi)*np.sin(kappa))
    m22 = (np.cos(omega)*np.cos(kappa))-(np.sin(omega)*np.sin(phi)*np.sin(kappa))
    m23 = (np.sin(omega)*np.cos(kappa))+(np.cos(omega)*np.sin(phi)*np.sin(kappa))
    
    m31 = np.sin(phi)
    m32 = -(np.sin(omega)*np.cos(phi))
    m33 = np.cos(omega)*np.cos(phi)
    
    # Populate Matrix
    M = np.array([
        [m11, m12, m13, bx],
        [m21, m22, m23, by],
        [m31, m32, m33, bz],
        [  0,    0,   0, 1]
    ])

    return M

def design_matrix_element(m, baseline, left, right, parameters):
    result = []
    for a in m:
        row = []
        for e in a:
            new_e = copy.deepcopy(e)
            new_e = new_e.subs(bx, baseline)
            new_e = new_e.subs(f, focal_length)
            
            new_e = new_e.subs(l1, left[0])
            new_e = new_e.subs(l2, left[1])
            new_e = new_e.subs(r1, right[0])
            new_e = new_e.subs(r2, right[1])

            new_e = new_e.subs(by, parameters[0])
            new_e = new_e.subs(bz, parameters[1])
            new_e = new_e.subs(omega, parameters[2])
            new_e = new_e.subs(phi, parameters[3])
            new_e = new_e.subs(kappa, parameters[4])
            row.append(new_e)
        result.append(row)
    return result

def relative_orientation(left, right, baseline):
    parameters = [0, 0, 0, 0, 0] # by, bz, omega, phi, kappa

    #by, bz, omega, phi, kappa = parameters
    #M = transformation(baseline, by, bz, omega, phi, kappa)

    #bx, by, bz, phi, kappa, omega, r1, r2, f= symbols('bx by bz phi kappa omega r1 r2 f')
    mis = misclosure()
    parital_derivatives = []
    parital_derivatives.append(diff_misclosure(mis, by))
    parital_derivatives.append(diff_misclosure(mis, bz))
    parital_derivatives.append(diff_misclosure(mis, omega))
    parital_derivatives.append(diff_misclosure(mis, phi))
    parital_derivatives.append(diff_misclosure(mis, kappa))

    A = []

    for i in range(len(left)):
        #right_coord = np.array([right[i][0], right[i][1], -focal_length, 1])
        #right_prime = M @ right_coord
        A_row = []
        for p in parital_derivatives:
            A_row.append(design_matrix_element(p, baseline, left[i], right[i], parameters))
        A.append(A_row)

        #right_prime_x = cos(phi)*cos(kappa)*r1 + (cos(omega)*sin(kappa) + sin(omega)*sin(phi)*cos(kappa))*r2 + (sin(omega)*sin(kappa) - cos(omega)*sin(phi)*cos(kappa))*f + bx
        #print(diff(right_prime_x, omega))

        # creating design matrix
        #misclosure = np.matrix([
        #    [baseline, parameters[0], parameters[1]],
        #    [left[i][0], left[i][1], -focal_length],
        #    [right_prime[0], right_prime[1], right_prime[2]]    
        #])

        #print(misclosure)

        #resdiuals = 

    print(np.array(A).shape)

base_distance = 92 # mm

# Verification

test_data_left = [
    [106.399, 90.426], # 30
    [18.989, 93.365,], # 40
    [70.964, 4.907],   # 72
    [-0.931, -7.284],  # 127
    [9.278, -92.926],  # 112
    [98.681, -62.769]  # 50
]

test_data_right = [
    [24.848, 81.824],  # 30
    [-59.653, 88.138], # 40
    [-15.581, -0.387], # 72 
    [-85.407, -8.351], # 127
    [-78.81, -92.62],  # 112
    [8.492, -68.873]    # 50
]

relative_orientation(test_data_left, test_data_right, base_distance)
# Python code for Robotics course assignment
# author: Branislav Jenco

import numpy as np
import sympy as sy
from functools import reduce

# Helper function to create A matrices
def A(theta, d, a, alpha, symbolic=False):
    if symbolic:
        return np.matrix(
            [[sy.cos(theta), -sy.sin(theta)*sy.cos(alpha), sy.sin(theta)*sy.sin(alpha), a*sy.cos(theta)],
            [sy.sin(theta), sy.cos(theta)*sy.cos(alpha), -sy.cos(theta)*sy.sin(alpha), a*sy.sin(theta)],
            [0, sy.sin(alpha), sy.cos(alpha), d],
            [0, 0, 0, 1]])
    else:
        return np.matrix(
                [[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]])

# Helper function to create the final Denavit-Hartenberg matrix
def DH(*matrices):
    return reduce((lambda x, y: x * y), list(matrices)) 

# Task 1 of Assignment 2. Create FK and IK functions

# Forward kinematics function
# takes a list of three angles in radians
# outputs a list of three cartesian coordinates
def forward(angles, constants, symbolic=False):
    print "Computing forward kinematics"
    L1 = constants["L1"]
    L2 = constants["L2"]
    L3 = constants["L3"]
    if symbolic:
        A1 = A(angles[0], L1, 0, sy.pi/2, True)
        A2 = A(angles[1], 0, L2, 0, True)
        A3 = A(angles[2], 0, L3, 0, True)
    else:
        A1 = A(angles[0], L1, 0, np.pi/2.0)            
        A2 = A(angles[1], 0, L2, 0)
        A3 = A(angles[2], 0, L3, 0)
    res = DH(A1, A2, A3)
    print res
    return np.array([res[0,3], res[1,3], res[2,3]])
    

# Inverse kinematics function
# takes in a list of three cartesian coordinates x, y, z
# outputs a set of four possible solutions to the IK equation (a list of four lists of three angles)
def inverse(cart_coord, constants, symbolic=False):
    print "Computing inverse kinematics"
    print cart_coord
    Xc = cart_coord[0]
    Yc = cart_coord[1]
    Zc = cart_coord[2]
    L1 = constants["L1"]
    L2 = constants["L2"]
    L3 = constants["L3"]
    
    if symbolic:
        theta1_1 = sy.simplify(sy.atan2(Yc, Xc))
        theta1_2 = sy.simplify(theta1_1 + sy.pi)
        D = (Xc*Xc + Yc*Yc + (Zc - L1)*(Zc - L1) - L2*L2 - L3*L3) / (2*L2*L3)
        theta3_1 = sy.simplify(sy.atan2(sy.sqrt(1-D*D),D))
        theta3_2 = sy.simplify(sy.atan2(-sy.sqrt(1-D*D),D))

        theta2_1 = sy.simplify(sy.atan2(Zc - L1, sy.sqrt(Xc*Xc + Yc*Yc)) - sy.atan2(L3*sy.sin(theta3_1), L2 + L3*sy.cos(theta3_1)))
        theta2_2 = sy.simplify(sy.atan2(Zc - L1, sy.sqrt(Xc*Xc + Yc*Yc)) - sy.atan2(L3*sy.sin(theta3_2), L2 + L3*sy.cos(theta3_2)))
        theta2_3 = sy.simplify(sy.atan2(Zc - L1, -sy.sqrt(Xc*Xc + Yc*Yc)) - sy.atan2(L3*sy.sin(theta3_1), L2 + L3*sy.cos(theta3_1)))
        theta2_4 = sy.simplify(sy.atan2(Zc - L1, -sy.sqrt(Xc*Xc + Yc*Yc)) - sy.atan2(L3*sy.sin(theta3_2), L2 + L3*sy.cos(theta3_2)))
    else:
        theta1_1 = np.arctan2(Yc, Xc)
        theta1_2 = theta1_1 + np.pi
        D = (Xc*Xc + Yc*Yc + (Zc - L1)*(Zc - L1) - L2*L2 - L3*L3) / (2*L2*L3)
        if (1.0 - D*D) < 0:
            print "No solution"
            return
        theta3_1 = np.arctan2(np.sqrt(1.0-D*D),D)
        theta3_2 = np.arctan2(-np.sqrt(1.0-D*D),D)
        
        theta2_1 = np.arctan2(Zc - L1, np.sqrt(Xc*Xc + Yc*Yc)) - np.arctan2(L3*np.sin(theta3_1), L2 + L3*np.cos(theta3_1))
        theta2_2 = np.arctan2(Zc - L1, np.sqrt(Xc*Xc + Yc*Yc)) - np.arctan2(L3*np.sin(theta3_2), L2 + L3*np.cos(theta3_2))
        theta2_3 = np.arctan2(Zc - L1, -np.sqrt(Xc*Xc + Yc*Yc)) - np.arctan2(L3*np.sin(theta3_1), L2 + L3*np.cos(theta3_1))
        theta2_4 = np.arctan2(Zc - L1, -np.sqrt(Xc*Xc + Yc*Yc)) - np.arctan2(L3*np.sin(theta3_2), L2 + L3*np.cos(theta3_2))
    

    return np.array(([theta1_1, theta2_1, theta3_1], [theta1_1, theta2_2, theta3_2], [theta1_2, theta2_3, theta3_1], [theta1_2, theta2_4, theta3_2]))

# Task 3c of Assignment 2. Verification function
# verifies whether one of the solutions computed in the IK function corresponds to the initial angles given to the FK function
def verify(result_angles, initial_angles):
    for solution in result_angles:
        # if one of the solutions corresponds to the initial angles
        if (np.allclose(solution, initial_angles)):
            return True
        
    return False


# Task 3 of Assignment 2. Create jacobian function
# takes in a list of three joint angles and a list of three joint velocities
# constructs the jacobian matrix 
# outputs the cartesian velocities of the end effectors
def jacobian(joint_angles, joint_velocities, constants, symbolic=False):
    # helper function to compute the Jv_i term in the jacobian
    def Jv_i(z_i_minus_1, o_n, o_i_minus_1, is_revolute=True, symbolic=False):
        if is_revolute:
            if symbolic:
                return np.cross(z_i_minus_1, (o_n - o_i_minus_1))
            else:
                return np.cross(z_i_minus_1, (o_n - o_i_minus_1))
        else:
            return z_i_minus_1

    # helper function to compute the Jomega_i term in the jacobian
    def Jomega_i(z_i_minus_1, is_revolute=True, symbolic=False):
        if is_revolute:
            if symbolic:
                return z_i_minus_1
            else:
                return z_i_minus_1
        else:
            return 0
    
    print "Computing jacobian"
    
    L1 = constants["L1"]
    L2 = constants["L2"]
    L3 = constants["L3"]
    theta1 = constants["theta1"]
    theta2 = constants["theta2"]
    theta3 = constants["theta3"]
    # the A matrices that we use in the forward kinematics function
    if symbolic:
        A1 = A(theta1, L1, 0, sy.pi/2, True)
        A2 = A(theta2, 0, L2, 0, True)
        A3 = A(theta3, 0, L3, 0, True)
    else:
        A1 = A(theta1, L1, 0, np.pi/2)
        A2 = A(theta2, 0, L2, 0)
        A3 = A(theta3, 0, L3, 0)
    
    T1_0 = A1
    T2_0 = A1*A2
    T3_0 = A1*A2*A3
    z0 = np.mat([0,0,1])
    z1 = T1_0[0:3, 2].T
    z2 = T2_0[0:3, 2].T
    z3 = T3_0[0:3, 2].T
    o0 = np.mat([0,0,0])
    o1 = T1_0[0:3, 3].T
    o2 = T2_0[0:3, 3].T
    o3 = T3_0[0:3, 3].T
    
    
    if symbolic:
        Jv1 = Jv_i(z0, o3, o0, symbolic=True)
        Jv2 = Jv_i(z1, o3, o1, symbolic=True)
        Jv3 = Jv_i(z2, o3, o2, symbolic=True)
    Jv1 = Jv_i(z0, o3, o0)
    Jv2 = Jv_i(z1, o3, o1)
    Jv3 = Jv_i(z2, o3, o2)
    Jomega1 = Jomega_i(z0)
    Jomega2 = Jomega_i(z1)
    Jomega3 = Jomega_i(z2)
    
    J = np.concatenate([np.vstack([Jv1,Jv2,Jv3]).T, np.vstack([Jomega1,Jomega2, Jomega3]).T])
    if symbolic:
        J_s = sy.simplify(sy.Matrix(J))
        print "Jacobian:\n", J_s
        print "\nDet(J):\n", sy.simplify(J_s[0:3, 0:3].det())
    print J
    return J.dot(joint_velocities)[0, 0:3]
    


def main():
    np.set_printoptions(suppress=True)
    use_symbolic = False
    if use_symbolic:
        theta1, theta2, theta3, L1, L2, L3, q1, q2, q3 = sy.symbols('t1 t2 t3 L1 L2 L3 q1 q2 q3')
        joint_velocities = [q1, q2, q3]
    else:
        L1 = 11.05 # cm
        L2 = 22.21 # cm
        L3 = 35.0 # cm
        theta1 = -np.pi/2 # 270 / -90
        theta2 = np.pi/6 # 30
        theta3 = -np.pi/4 # -45
        joint_velocities = [0.1, 0.05, 0.05] #rad/s
        

    initial_angles = np.array([theta1, theta2, theta3])
    print "----------------------"
    print "Initial angles:", initial_angles/np.pi*180
    print "----------------------"
    cart_coord = forward(initial_angles, locals(), use_symbolic)
    print "FK coordinates: ", cart_coord
    print "----------------------"
    result_angles = inverse(cart_coord, locals(), use_symbolic)
    print "IK angles: \n", result_angles/np.pi*180
    test = forward(result_angles[3], locals(), use_symbolic)
    print "test ", test
    if not use_symbolic:
        if verify(result_angles, initial_angles):
            print "Verified"
        else:
            print "Resulting angles and initial angles do not match"

    print "----------------------"
    velocities = jacobian(initial_angles, joint_velocities, locals(), use_symbolic)
    print "Cartesian velocity of the end effector (mm/s):", velocities
    print "Finished"

if __name__ == "__main__":
    main()
    


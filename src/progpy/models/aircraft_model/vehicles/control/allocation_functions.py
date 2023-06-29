# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

def rotorcraft_cam(n, length, b, d, constrained=False):
    """
    Generate control allocation matrix (CAM) to transform rotor's angular velocity into thrust and torques around three main body axes.

    [T, Mx, My, Mz]^{\top} = \Gamma [\Omega_1^2, \Omega_2^2, ..., \Omega_n^2]^{\top}

    Where:
        [T, Mx, My, Mz]^{\top} is the (4 x 1) column vector containing thrust (T) and moments along main body axis (Mx, My, Mz)
        \Gamma is the (4 x n) CAM
        [\Omega_1^2, \Omega_2^2, ..., \Omega_n^2]^{\top} is the (n x 1) column vector of the rotor angular speed squared.

    The control allocation matrix is built under the assumption of symmetric rotor configuration, for generality.
    Special CAM should be built ad-hoc per UAV model.

    :param n:           number of rotors
    :param length:           rotor arm's length (from center of mass to rotor center)
    :param b:           thrust constant, function of rotor type
    :param d:           torque constant, function of rotor type
    :return:            control allocation matrix of dimensions (4, n) and its pseudo-inverse of dimensions (n, 4)
    """

    Gamma = np.empty((4, n))
    optional = {}
    if n == 8 and not constrained:
        l_b = length * b
        l_b_sq2o2 = l_b * np.sqrt(2.0) / 2.0
        # This CAM is assuming there's no rotor pointing towards the drone forward direction (x-axis)
        # See Dmitry Luchinsky's report for details (TO BE VERIFIED)
        Gamma = np.array([[b, b, b, b, b, b, b, b],
                          [l_b, l_b_sq2o2, 0.0, -l_b_sq2o2, -l_b, -l_b_sq2o2, 0.0, l_b_sq2o2],
                          [0.0, -l_b_sq2o2, -l_b, -l_b_sq2o2, 0.0, l_b_sq2o2, l_b, l_b_sq2o2],
                          [-d, d, -d, d, -d, d, -d, d]])
    if n == 8 and constrained:
        bl = b * length
        b2 = 2.0 * b
        d2 = 2.0 * d
        blsqrt2 = np.sqrt(2.0) * bl

        Gamma = np.array([[b2, b2, b2, b2],
                          [bl, 0.0, -bl, 0.0],
                          [-bl, -blsqrt2, bl, blsqrt2],
                          [-d2, d2, -d2, d2]])
        optional['selector'] = np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 1]]).T

    return Gamma, np.linalg.pinv(Gamma), optional

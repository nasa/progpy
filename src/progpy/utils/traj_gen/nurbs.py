# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
NURBS curve functions
"""

import numpy as np
import scipy.interpolate as interp


def normalize_t_1(t, a, b):
    """ 
    Normalization Function 1: normalize t in (a,b) range as:
            
            (t - a) / (b - a)
    
    Variable t can be a scalar or 1D array / vector.
    Values a, b can be scalars or 1D arrays or vectors with same length of t.

    :param t:       scalar or n x 1, value to be normalized
    :param a:       scalar or n x 1, first normalization parameter
    :param b:       scalar or n x 1, second normalization parameter
    :return:        scalar or n x 1, normalized t value
    """
    with np.errstate(divide="ignore"):  # ignore 0/0 warnings
        return (t-a) / (b - a)


def normalize_t_2(t, a, b):
    """ 
    Normalization Function 2: normalize -t in (a,b) range as:
            
            (b - t) / (b - a)
    
    Variable t can be a scalar or 1D array / vector.
    Values a, b can be scalars or 1D arrays or vectors with same length of t.

    :param t:       scalar or n x 1, value to be normalized
    :param a:       scalar or n x 1, first normalization parameter
    :param b:       scalar or n x 1, second normalization parameter
    :return:        scalar or n x 1, normalized t value
    """
    with np.errstate(divide="ignore"):  # ignore 0/0 warnings
        return (b-t) / (b-a)


class NURBS:
    def __init__(self,
                 points: dict,
                 weights,
                 times,
                 yaw=None,
                 **kwargs) -> None:

        self.points = points
        self.dims = list(self.points.keys())
        self.weights = weights
        self.times = times
        self.yaw = yaw
        self.knotv = None

        self.parameters = dict(order=4, basis_length=1000)
        self.parameters.update(**kwargs)

    def point_values(self,):
        return np.vstack(list(self.points.values())).T

    def points_new_values(self,):
        return np.vstack(list(self.points_new.values())).T

    def generate(self, timestep_size):
        """
        Generate NURBS curve given the timestep_size.
        This function first add intermediate way-points to ensure the control over what way-points need to be approached closely and what
        could be passed by far away.
        Then the function generate the position profile as a function of such way-points, using the desired timestep_size.
        
        The last step consists of the generation of a smooth yaw curve.

        :param timestep_size:           scalar, double, time step size to be used to generate the smooth trajectory.
        :return:                        three m x 1 arrays corresponding to the interpolated position and yaw angle, and the corresponding time vector
        """
        # Generate intermediate way-points to control which way-points should/could be passed by closely or far away
        points_new, self.times_new, self.yaw_new, self.weight_vector_new = self.generate_intermediate_points()
        self.points_new = {dim: points_new[:, i] for i, dim in enumerate(self.dims)}    # Save way-points with intermediate values:

        # Generate the knot vector necessary for the NURBS-based interpolation
        self.gen_knot_vector(points_new.shape[0]-1)
        
        # Interpolate position profile
        position_interp = {self.dims[ii]: None for ii in range(len(self.dims))}
        timevec_interp = None
        for ii, dim_i in enumerate(self.dims):
            pos_interp, timevec = self.interpolate(pos=points_new[:, ii],
                                                   times=self.times_new,
                                                   timestepsize=timestep_size,
                                                   weightvector=self.weight_vector_new)
            position_interp[dim_i] = pos_interp
        timevec_interp = timevec

        # Interpolate yaw
        # -----------------
        yaw_interp, timevec = self.interpolate(self.yaw_new,  # anchor points: yaw points
                                               self.times_new,  # time instants of the anchor points
                                               np.ones_like(self.yaw_new),  # weight vector, can be all the same for the yaw points
                                               timestep_size,  # time step size for the curve (same as for position)
                                               order=3,  # overriding order of the nurbs: higher order help the sharp yaw changes
                                               basis_length=2000,  # basis length: 1000, could be changed but would either increase computation cost or coarse the trajectory too much
                                               interp_order='zero')  # order of the yaw curves: the yaw should be kept constant in-between waypoints (no smooth variations, except for the proximity of turns), keep order 0 so is step-wise constant
        return position_interp, yaw_interp, timevec_interp

    def gen_knot_vector(self, n):
        """
        Generate knot vector of k-th order for a NURBS with n-points.
        :param n:       number of anchor points of the curve -1  (e.g., number waypoints of trajectory - 1)
        :return:        normalized knot vector, [0,1]
        """
        k = self.parameters['order']
        t = np.zeros((n + k + 1,))  # initialize vector
        idx_vec = np.arange(1, n + k + 2)  # generate indices of the vector
        msk1 = (idx_vec >= k) * (idx_vec <= n)  # create a mask for values between k and n
        t[msk1] = (idx_vec[msk1] - 1) - k + 1  # assign values to the masked elements
        t[idx_vec > n] = n - k + 2  # assign fixed value n-k+2 for everything above index n
        t = (t - min(t)) / (max(t) - min(t))  # normalize vector
        self.knotv = t
    
    def basisfunction(self, u):
        """
        Basis function used to generate a smooth NURBS given order and knot vector t.
        Variable u is the independent variable of the basis function.

        :param u:           scalar, double, independent variable u used to generate basis function N(u)
        :param t:           knot vector defining the control points
        :return:            basis function of order 'order', N_order(u)
        """
        order = self.parameters['order']
        t = self.knotv
        
        n_points = len(t)
        N = np.zeros((n_points,))
        N[(u > t) * np.insert((u <= t[1:]), -1, True)] = 1
        if N.any() != 0:
            for n in range(1, order):
                t_pn = np.concatenate((t[n:], np.ones((n,))))
                d = normalize_t_1(u, t, t_pn) * N
                t_pn1 = np.concatenate((t[n+1:], np.ones((n+1,))))
                t_p1 = np.concatenate((t[1:], np.ones((1,))))
                e = normalize_t_2(u, t_p1, t_pn1) * np.insert(N[1:], -1, 0)
                N = np.nan_to_num(e) + np.nan_to_num(d)
                
        return N[:n_points - order]

    def evaluate(self, pointvec, weightvec, order=None, basis_length=None):
        """
        Evaluate NURBS function given order, knot vector t, weights of the waypoints, waypoint vector, and basis length

        :param order:               order of the NURBS curve
        :param t:                   knot vector with control points
        :param weights:             weights of the waypoints
        :param waypoint_vector:     matrix containing waypoints
        :param basis_length:        length of the basis function, number of points used to define independent variable u
        :return:                    NURBS curve evaluated at the desired points (defined by knot vector t. Form t to u to C(u))
                                    2 x q, where the first row is time, second row is quantity x. q is the number of points len(U)
        """
        # Check validity of waypoint vector
        # It should be a 2 x m array where m is the number of waypoints, the first row is the ETA vector and the second row is the actual waypoint vector.

        n, m = pointvec.shape
        if n != 2:
            pointvec = pointvec.T

        if order is None:
            order = self.parameters['order']
            
        if basis_length is None:
            basis_length = self.parameters['basis_length']

        # Define independent variable vector U (composed of all points of the independent variable u_0, u_1, ...)
        U = np.linspace(self.knotv[order-1], self.knotv[-1], int(round(abs(self.knotv[-1] - self.knotv[order-1])*basis_length)))
        q = len(U)

        # Calculate basis functions
        # -----------------------------------------
        N = np.zeros((q, pointvec.shape[1]))
        for ii in range(q):
            N[ii, :] = self.basisfunction(U[ii])   # first piece of basis function
        N[0, 0] = 1.0   # Numerical issue: the first point of the first basis is not computed but remains 0 when it should be 1. This fixes it.

        S = np.dot(N, weightvec)        # Calculate denominator of rational basis function
        R = N * np.repeat(weightvec.reshape((1, -1)), repeats=N.shape[0], axis=0)       # Calculate rational basis function
        with np.errstate(divide="ignore"):  # ignore warning divide by zero
            R /= np.repeat(S.reshape((-1, 1)), repeats=m, axis=1)
        R[np.isnan(R)] = 0.

        # Generate points of the planned path
        # ----------------------------------
        C = np.zeros((2, q))
        C[0, :] = np.dot(R, pointvec[0, :].reshape((-1,)))
        C[1, :] = np.dot(R, pointvec[1, :].reshape((-1,)))
        return C
    
    def interpolate(self, pos, times, weightvector, timestepsize, order=None, basis_length=None, interp_order='cubic'):
        """
        Interpolate a set of anchor points with corresponding time instants and weight vector using NURBS bases.
        This is the core of a NURBS interpolation.

        :param pos:     n x 1 array of anchor or position points
        :param times:   n x 1 array of time instants corresponding to the anchor points
        :param weightvector:            n x 1 array of weights associated with each anchor point
        :param timestepsize:            scalar, double, dt of the resulting nurbs-based curve.
        :param oder:                    scalar, int, order of the NURBS used for the interpolation. Deafult is None, which uses the one defined at the class instantiation
        :param basis_length:            scalar, int, length of the basis function used to generate the NURBS. Default is None, which uses the one defined at the class instantiation
        :param interp_order:            string, final interpolation of the curve. This is a 1D interpolation to further smooth out the trajectory. Default is 'cubic'.
        :return                         two m x 1 arrays, a position profile and time vector, interpolated with the desired timestepsize.
        """
        m = len(times)  # time values of each corresponding point in pos
        
        # Initialize array containing the time-parameterized curve
        p_vector = np.zeros((2, m))
        p_vector[0, :] = times  # first row is time
        p_vector[1, :] = pos    # second row is anchor points (or positions)

        # Evaluate NURBS at desired times with corresponding weight vector, given order and basis length
        pos_vs_time = self.evaluate(p_vector, weightvector, order=order, basis_length=basis_length)

        # Generate fine-grained time vector for interpolation
        timevec = np.arange(times[0], times[-1]+timestepsize, timestepsize)

        # Interpolation position vs time to have a smooth trajectory
        pos_vs_time[:, 0] = np.array([0.0, pos[0]])
        pos_interp = interp.interp1d(pos_vs_time[0, :], pos_vs_time[1, :], kind=interp_order, fill_value='extrapolate')(timevec)
        return pos_interp, timevec


    def generate_intermediate_points(self,):
        """
        This function takes the original set of curve points and generates a new set containing fictitious, intermediate points.
        The fictitious points help in generating a NURBS trajectory that passes through the desired (true) points according
        to the weight vector.

        The fictitious points are particularly useful for rotary-wing UAV trajectories, since the latter have very small minimum-radius
        turns and can do sharp turns. Without such fictitious points, the trajectory will look more like a fixed-wing trajectory, where the waypoints
        are approached, but they are not reached.

        :param points:             (n x 3) array, coordinate of points
        :param yaw:            (n,) array, yaw angle in between waypoints (ini yaw=0), [rad]
        :param eta:            (n,) array, estimated time of arrival at each waypoint, [s]

        :return points:                  (m x 3) array, coordinate including intermediate points,
        :return yaw_new:                 (m,) array, yaw values including intermediate points, [rad]
        :return eta_new:                 (m,) array, ETA at points, including intermediate points, [s]
        """
        # Get from class properties
        points = self.point_values()
        times = self.times
        yaw = self.yaw

        # ------- New array of waypoints -------
        intermediate_points = np.zeros((points.shape[0] - 1, 3))  # initialize array of fictitious waypoints
        for ii in range(1, points.shape[0]):
            intermediate_points[ii - 1, :] = (points[ii, :] - points[ii - 1, :]) / 2.0 + points[ii - 1, :]  # generate a fictitious waypoint as average of true waypoints

        # Generate the new waypoint matrix
        points_new = np.zeros((points.shape[0] + intermediate_points.shape[0], points.shape[1]))
        counter_1, counter_2 = 0, 0
        for ii in range(points_new.shape[0]):
            if ii == 0 or ii % 2 == 0:
                points_new[ii, :] = points[counter_1, :]
                counter_1 += 1
            else:
                points_new[ii, :] = intermediate_points[counter_2, :]
                counter_2 += 1

        # -------- New Weight Vector -------------
        weight_vector_new = np.zeros((points_new.shape[0],))  # initialize weight vector
        counter_1 = 0  # add a temporary counter for the true weight vector
        for jj in range(len(weight_vector_new)):
            if jj % 2 == 0:
                weight_vector_new[jj] = self.weights[counter_1]  # Assign pre-defined weight to the waypoint
                counter_1 += 1
            else:
                weight_vector_new[jj] = 1  # Assign standard weight of 1 to the fictitious waypoint

        # -------- Generate new ETA vector -------------
        intermediate_times = np.zeros((len(self.weights) - 1,))  # initialize fictitious ETA vector
        for ii in range(1, len(intermediate_times) + 1):
            intermediate_times[ii - 1] = (times[ii] - times[ii - 1]) / 2.0 + times[ii - 1]  # generate fictitious ETA as average of real ETAs

        # Generate new ETA vector
        times_new = np.zeros((weight_vector_new.shape[0],))
        counter_1, counter_2 = 0, 0
        for ii in range(len(times_new)):
            if ii == 0 or ii % 2 == 0:
                times_new[ii] = times[counter_1]  # Assign existing ETA
                counter_1 += 1
            else:
                times_new[ii] = intermediate_times[counter_2]  # Assign fictitious ETA
                counter_2 += 1

        # -------- Generate new Yaw vector ----------
        # The fictitious yaw is NOT the average of the true yaw, but it is equal to the yaw at the previous (true) waypoint.
        yaw_new = np.zeros((points_new.shape[0],))  # Initialize new yaw vector
        counter_1 = 1
        for jj in range(len(yaw)):
            yaw_new[counter_1:counter_1 + 2] = yaw[jj]  # Assign new yaw value
            counter_1 += 2  # this counter is
        return points_new, times_new, yaw_new, weight_vector_new
        
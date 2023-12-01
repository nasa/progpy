# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np


def lqr_calc_k(A, B, Q, R):
    """
    Calculate control matrix K using LQR

    The function estimates the gain matrix associated with a linear quadratic regulator(LQR) via Hamiltonian matrix.
    The function assumes a linearized system:

    dx / dt = Ax + Bu
    y = Cx + Du

    The closed loop uses - K * x as input dx / dt = (A - B * K) * x

    :param A:       Nx x Nx state matrix, linearized w.r.t. current attitude and thrust
    :param B:       Nu x Nx input matrix, linearized w.r.t. current attitude
    :param Q:       Nx x Nx state cost matrix, the higher the values of the (i,i)-th element, the more aggressive the controller is w.r.t. that variable
    :param R:       Nx x Nu input cost matrix, higher values imply higher cost of producing that input (used to limit the amount of power)
    :return K:      Nx x Nu, control gain matrix
    :return E:      Nx x 1, eigenvalues of the matrix A - B * K
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    B_tr = B.T

    # --------- Generate hamiltonian matrix ------------ #
    HM1 = A
    HM2 = - np.dot(np.dot(B, R_inv), B_tr)
    HM3 = - Q
    HM4 = - A.T

    HM12 = np.concatenate((HM1, HM2), axis=1)
    HM34 = np.concatenate((HM3, HM4), axis=1)
    HM = np.concatenate((HM12, HM34), axis=0)

    # --------- Extract eigenvectors whose eigenvalues have real part < 0 ------------ #
    eig_val, eig_vec = np.linalg.eig(HM)
    V_ = eig_vec[:, np.real(eig_val) < 0.0]

    # --------- Partition matrix of resulting eigenvectors ------------ #
    X = V_[:n, :n]  # first half (row-wise)
    Y = V_[n:, :n]  # second half (row-wise)

    # ----------- Estimate control gain ----------------- #
    K = np.dot(np.dot(np.dot(R_inv, B_tr), Y), np.linalg.inv(X))
    K = np.real(K)  # some spurious imaginary parts (1e-13) sometimes remain in the matrix. we manually remove them

    # Calculate the eigenvalues of the matrix A - B*K
    E, _ = np.linalg.eig(A - np.dot(B, K))
    return K, E


# CONTROLLERS: FULL STATE VECTOR CONTROL (i.e., no powertrain)
class LQR():
    """
    Linear Quadratic Regulator UAV Controller

    A Controller that calculates the vehicle control inputs for a UAV model (progpy.models.uav_model).

    args:
        x_ref (dict[str, np.ndarray]):
            dictionary of reference trajectories for each state variable (x, y, z, phi, theta, psi, ...)
        vehicle:
            UAV model object

    keyword args: 
        Q (np.ndarray):
            State error penalty matrix, size num_states x num_states (where num_states only includes dynamic states)
            Represents how 'bad' an error is in the state vector w.r.t. the reference state vector
        R (np.ndarray):
            Input penalty matrix, size num_inputs x num_inputs (where num_inputs only includes inputs that inform the system dynamics)
            Represents how 'hard' it is to produce the desired input (thrust and three moments along three axes)
        scheduled_var (str):
            Variable used to create the scheduled controller gains; must correspond to a state key
        index_scheduled_var (int):
            Index corresponding to the scheduled_var in the state vector

    """
    def __init__(self, x_ref, vehicle, **kwargs):

        # Check correct arguments:
        if not isinstance(x_ref, dict):
            raise TypeError("Reference trajectory must be a dictionary of numpy arrays for each state throughout time.")
        for vals in x_ref.values():
            if not isinstance(vals, np.ndarray):
                raise TypeError("Reference trajectory must be a dictionary of numpy arrays for each state throughout time.")

        self.states = vehicle.states         # state variables of the system to be controlled (x, y, z, phi, theta, psi)
        self.n_states = len(self.states) - 2   # number of states (minus two to remove time and mission_complete)
        self.inputs = vehicle.inputs         # input variables of the system to be controlled
        self.n_inputs = len(self.inputs) - 1   # number of inputs (minus one to remove mission_complete)
        self.ref_traj = x_ref                  # reference state to follow during simulation (x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref, ...)
        self.ss_input = vehicle.parameters['steadystate_input']
        self.vehicle_max_thrust = vehicle.parameters['dynamics']['max_thrust']
        
        # Default control parameters
        # --------------------------------
        self.parameters = dict(Q=np.diag([1000, 1000, 25000, 100.0, 100.0, 100.0, 1000, 1000, 5000, 1000, 1000, 1000]),  # state error penalty matrix: how 'bad' is an error in the state vector w.r.t. the reference state vector
                               R=np.diag([500, 4000, 4000, 4000]), # input penalty matrix: how 'hard' it is to produce the desired input (thrust and three moments along three axes)
                               scheduled_var='psi',     # variable used to create the scheduled controller gains (only psi allowed for now)
                               index_scheduled_var=5)   # index corresponding to the scheduled_var (psi) in the state vector x; i.e., x[5] = psi
        self.parameters.update(kwargs)                  # update control parameters according to user

        # Get scheduled variable index
        self.parameters['index_scheduled_var'] = self.states.index(self.parameters['scheduled_var'])

        # Initialize other controller-related variables
        self.dt = vehicle.parameters['dt']

        # Initialize control gain storage matrix
        self.control_gains = np.zeros((self.n_inputs, self.n_states, 1))

        # Build and store control matrices for scheduling
        self.build_scheduled_control(vehicle.linear_model, input_vector=[self.ss_input])

    def __call__(self, t, x=None):

        if x is None:
            x_k = np.zeros((self.n_states, 1))
        else:
            x_k = np.array([x.matrix[ii][0] for ii in range(self.n_states)])

        # Identify reference state (desired state) at t
        t_k = np.round(t + self.dt/2.0, 1)  # current time step
        time_ind = np.argmin(np.abs(t_k - self.ref_traj['t'].tolist()))  # get index of time value in ref_traj closest to t_k
        x_ref_k = []
        for state in self.states:
            if state != 'mission_complete':
                x_ref_k.append(self.ref_traj[state][time_ind])
        x_ref_k = np.asarray(x_ref_k[:-1])  # get rid of time index in state vector
        x_k = x_k.reshape(x_k.shape[0],)

        error = x_k - x_ref_k    # Error between current and reference state
        scheduled_var = x_k[self.parameters['index_scheduled_var']]    # get psi from current state vector (self.parameters = 'psi')
        k_idx = np.argmin(np.abs(self.scheduled_states[self.parameters['index_scheduled_var'], :] - scheduled_var))  # find the psi value stored in the controller closest to the current psi --> extract index
        K = self.control_gains[:, :, k_idx]  # extract gain corresponding to the current psi value
        u = self.compute_input(K, error)  # compute input u given the gain matrix K and the error between current and reference state
        u[0] += self.ss_input
        u[0] = min(max([0, u[0]]), self.vehicle_max_thrust)
        return {
            'T': u[0],
            'mx': u[1],
            'my': u[2],
            'mz': u[3],
            'mission_complete': t_k/self.ref_traj['t'][-1]}

    def compute_gain(self, A, B):
        """ Compute controller gain given state of the system described by linear model A, B"""
        self.K, self.E = lqr_calc_k(A, B, self.parameters['Q'], self.parameters['R'])
        return self.K, self.E

    def compute_input(self, gain, state_error):
        """ Compute system input given the controller gain 'gain' and the error w.r.t. the reference state 'error' """
        return - np.dot(gain, state_error)
    
    def build_scheduled_control(self, system_linear_model_fun, input_vector, state_vector_vals=None, index_scheduled_var=None):
        if state_vector_vals is None:
            # using psi (yaw angle) as scheduled variable as the LQR control cannot work with yaw=0 since it's in the inertial frame.
            n_schedule_grid = 360*2 + 1
            index_scheduled_var = 5
            state_vector_vals = np.zeros((self.n_states, n_schedule_grid))
            state_vector_vals[index_scheduled_var, :] = np.linspace(-2.0*np.pi, 2.0*np.pi, n_schedule_grid)

        n, m = state_vector_vals.shape
        assert n == self.n_states, "number of states set at initialization and size of state_vector_vals mismatch."
        self.control_gains = np.zeros((self.control_gains.shape[0], self.control_gains.shape[1], m))
        self.scheduled_states = state_vector_vals

        for j in range(m):
            phi, theta, psi = state_vector_vals[3:6, j]
            p, q, r = state_vector_vals[-3:, j]
            Aj, Bj = system_linear_model_fun(phi, theta, psi, p, q, r, input_vector[0])
            self.control_gains[:, :, j], _ = self.compute_gain(Aj, Bj)

        print('Control gain matrices complete.')


class LQR_I(LQR):
    """
    Linear Quadratic Regulator with Integral Effect
    
    args:
        x_ref (dict[str, np.ndarray]):
            dictionary of reference trajectories for each state variable (x, y, z, phi, theta, psi, ...)
        vehicle:
            UAV model object

    keyword args: 
        Q (np.ndarray):
            State error penalty matrix, size num_states x num_states (where num_states only includes dynamic states)
            Represents how 'bad' an error is in the state vector w.r.t. the reference state vector
        R (np.ndarray):
            Input penalty matrix, size num_inputs x num_inputs (where num_inputs only includes inputs that inform the system dynamics)
            Represents how 'hard' it is to produce the desired input (thrust and three moments along three axes)
        int_lag (int):
            Length of the time window, in number of simulation steps, to integrate the state position error. The time window is defined 
            as the last int_lag discrete steps up until the current discrete time step. The integral of the state position error adds to 
            the overall state error to compute the gain matrix, and helps compensate for constant offsets between the reference (desired) 
            position and the actual position of the vehicle.
        scheduled_var (str):
            Variable used to create the scheduled controller gains; must correspond to a state key
        index_scheduled_var (int):
            Index corresponding to the scheduled_var in the state vector
    """

    def __init__(self, x_ref, vehicle, **kwargs):
        # Check correct arguments:
        if not isinstance(x_ref, dict):
            raise TypeError("Reference trajectory must be a dictionary of numpy arrays for each state throughout time.")
        for vals in x_ref.values():
            if not isinstance(vals, np.ndarray):
                raise TypeError("Reference trajectory must be a dictionary of numpy arrays for each state throughout time.")

        self.states = vehicle.states         # state variables of the system to be controlled (x, y, z, phi, theta, psi)
        self.n_states = len(self.states) - 2  # number of states (minus two to remove time and mission_complete)
        self.inputs = vehicle.inputs         # input variables of the system to be controlled
        self.n_inputs = len(self.inputs) - 1  # number of inputs (minus one to remove mission_complete)
        self.ref_traj = x_ref                 # reference state to follow during simulation (x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref, ...)
        self.ss_input = vehicle.parameters['steadystate_input']
        self.vehicle_max_thrust = vehicle.parameters['dynamics']['max_thrust']

        self.outputs = vehicle.outputs[:3]    # output variables of the system to be controlled (x, y, z only)
        self.n_outputs = 3                    # number of outputs
        self.err_hist = []                    # error history (integral)

        # Default control parameters
        # --------------------------------
        self.parameters = dict(Q=np.diag([1.e3, 1.e3, 1.e6,  # x, y, z
                                          5.e5, 5.e5, 15.e3,  # phi, theta, psi,
                                          2.e2, 2.e2, 1.e2,  # vx, vy, vz,
                                          5.e4, 5.e4, 1.e4]),  # p, q, r
                               R=np.diag([200, 4e3, 4e3, 4e3]),  # T, Mx, My, Mz
                               qi=np.array([100, 100, 500]),
                               int_lag=200,
                               scheduled_var='psi',    # variable used to create the scheduled controller gains (only psi allowed for now)
                               index_scheduled_var=5)  # index corresponding to the scheduled_var (psi) in the state vector x; i.e., x[5] = psi
        self.parameters.update(kwargs)                 # update control parameters according to user

        # Get scheduled variable index
        self.parameters['index_scheduled_var'] = self.states.index(self.parameters['scheduled_var'])

        # Initialize other controller-related variables
        # ---------------------------------------------
        self.dt = vehicle.parameters['dt']
        
        # Initialize other controller-related variables
        self.C = np.zeros((self.n_outputs, self.n_states))
        self.C[:self.n_outputs, :self.n_outputs] = np.eye(self.n_outputs)

        # Initialize Augmented state space matrices for integral action
        self.Ai = np.zeros((self.n_states+self.n_outputs, self.n_states+self.n_outputs))
        self.Bi = np.zeros((self.n_states+self.n_outputs, self.n_inputs))
        self.Ai[self.n_states:, :self.n_states] = self.C

        # Generate augmented Q and R for integral term
        self.parameters['Qi'] = np.diag(np.concatenate((np.diag(self.parameters['Q']), self.parameters['qi']), axis=0))
        self.parameters['Ri'] = self.parameters['R']

        # Control gain storage matrix:
        self.control_gains = np.zeros((self.n_inputs, self.n_states + self.n_outputs, 1))

        # Build and store control matrices for scheduling
        self.build_scheduled_control(vehicle.linear_model, input_vector=[self.ss_input])

    def compute_gain(self, A, B):
        """Compute controller gain given state of the system described by linear model A, B"""
        self.Ai[:self.n_states, :self.n_states] = A
        self.Bi[:self.n_states, :] = B
        self.K, self.E = lqr_calc_k(self.Ai, self.Bi, self.parameters['Qi'], self.parameters['Ri'])
        return self.K, self.E

    def compute_input(self, gain, error):
        """ Compute system input given the controller gain 'gain' and the error w.r.t. the reference state 'error' """
        self.err_hist.append(np.dot(self.C, error))
        err_hist = np.asarray(self.err_hist).T
        err_integral = np.sum(err_hist[:, max([0, len(self.err_hist) - self.parameters['int_lag']]):], axis=1) * self.dt
        return - np.dot(gain[:, :self.n_states], error) - np.dot(gain[:, self.n_states:], err_integral)

    def reset_controller(self):
        """ Reset Error history (from integral action, if exist) """
        if hasattr(self, 'err_hist'):
            self.err_hist = []

# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

from progpy.models.aircraft_model import AircraftModel
from progpy.models.aircraft_model.vehicles.aero import aerodynamics as aero
from progpy.models.aircraft_model.vehicles import vehicles
from progpy.utils.traj_gen import geometry as geom
from progpy.utils.traj_gen.trajectory import TrajectoryFigure


class SmallRotorcraft(AircraftModel):
    r"""
    .. versionadded:: 1.5.0
    
    Vectorized prognostics :term:`model` to generate a predicted trajectory for a small rotorcraft using a n=6 degrees-of-freedom dynamic model
    with feedback control loop. The model follows the form:
    
    .. math::
        u     = h(x, x_{ref})
        
        dx/dt = f(x, \theta, u)
    
    where:
      x is a 2n state vector containing position, attitude and corresponding derivatives
      :math:`\theta` is a vector of model parameters including rotorcraft mass, inertia moment, aerodynamic coefficients, etc.
      u is the input vector: thrust along the body vertical axis, and three moments along the UAV body axis to follow the desired trajectory.
      :math:`x_{ref}` is the desired state vector at that specific time step, with dimension 2n
      f(.) is growth rate function of all vehicle state
      h(.) is the feedback-loop control function that returns the necessary thrust and moments (u vector) to cover the error between desired state :math:`x_{ref}` and current state x
      dx/dt is the state-increment per unit time.

    Model generates cartesian positions and velocities, pitch, roll, and yaw, and angular velocities throughout time to satisfy some user-define waypoints.

    See [0]_ for modeling details.

    :term:`Events<event>`: (1)
        TrajectoryComplete: The final time of the reference trajectory has been reached
    
    :term:`Inputs/Loading<input>`: (0)
        | T: thrust
        | mx: moment in x
        | my: moment in y
        | mz: moment in z
        | mission_complete: progression throughout time to final time point in reference trajectory, where 0 is no progress and 1 is mission completed

    :term:`States<state>`: (14)
        | x: first position in cartesian reference frame East-North-Up (ENU), i.e., East in fixed inertia frame, center is at first waypoint
        | y: second position in cartesian reference frame East-North-Up (ENU), i.e., North in fixed inertia frame, center is at first waypoint
        | z: third position in cartesian reference frame East-North-Up (ENU), i.e., Up in fixed inertia frame, center is at first waypoint
        | phi: Euler's first attitude angle
        | theta: Euler's second attitude angle
        | psi: Euler's third attitude angle
        | vx: velocity along x-axis, i.e., velocity along East in fixed inertia frame
        | vy: velocity along y-axis, i.e., velocity along North in fixed inertia frame
        | vz: velocity along z-axis, i.e., velocity Up in fixed inertia frame
        | p: angular velocity around UAV body x-axis
        | q: angular velocity around UAV body y-axis
        | r: angular velocity around UAV body z-axis
        | t: time
        | mission_complete: progression throughout time to final time point in reference trajectory, where 0 is no progress and 1 is mission completed

    :term:`Outputs<output>`: (12)
        | x: first position in cartesian reference frame East-North-Up (ENU), i.e., East in fixed inertia frame, center is at first waypoint
        | y: second position in cartesian reference frame East-North-Up (ENU), i.e., North in fixed inertia frame, center is at first waypoint 
        | z: third position in cartesian reference frame East-North-Up (ENU), i.e., Up in fixed inertia frame, center is at first waypoint
        | phi: Euler's first attitude angle
        | theta: Euler's second attitude angle
        | psi: Euler's third attitude angle
        | vx: velocity along x-axis, i.e., velocity along East in fixed inertia frame
        | vy: velocity along y-axis, i.e., velocity along North in fixed inertia frame
        | vz: velocity along z-axis, i.e., velocity Up in fixed inertia frame
        | p: angular velocity around UAV body x-axis
        | q: angular velocity around UAV body y-axis
        | r: angular velocity around UAV body z-axis

    Keyword Args
    ------------
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state).
          Can be number (e.g., .2) applied to every state, a dictionary of values for each
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)
        dt : Optional, float
          Time step in seconds for trajectory generation
        gravity : Optional, float
          m/s^2, gravity magnitude
        air_density : Optional, float
          kg/m^3, atmospheric density
        steadystate_input : Optional, float
          Input vector to maintain the vehicle in a stable position that is used to build the linearized model for the controller.
        x0 : dict[str, float]
          Initial :term:`state`
        vehicle_model: Optional, str
          String to specify vehicle type. 'tarot18' and 'djis1000' are supported
        vehicle_payload: Optional, float
          kg, payload mass
        vehicle_max_speed : Optional, float
          m/s, maximum vehicle speed
        vehicle_max_roll : Optional, float
          rad, maximum roll angle
        vehicle_max_pitch : Optional, float
          rad, maximum pitch angle

    References
    ----------
    [0] M. Corbetta et al., "Real-time UAV trajectory prediction for safely monitoring in low-altitude airspace," AIAA Aviation 2019 Forum,  2019. https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3514
    """

    events = ['TrajectoryComplete']
    inputs = ['T', 'mx', 'my', 'mz', 'mission_complete']
    states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r', 't', 'mission_complete']
    outputs = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    
    is_vectorized = True

    default_parameters = {  # Set to defaults
        # Simulation parameters:
        'dt': 0.1,
        'gravity': 9.81,
        'air_density': 1.225,
        'steadystate_input': None,
        
        'x0': {key: 0.0 for key in states}, # Initial state

        # Vehicle parameters:
        'vehicle_model': 'tarot18',
        'vehicle_payload': 0.0,
        'vehicle_max_speed': 15.0,
        'vehicle_max_roll': 0.7853981633974483,
        'vehicle_max_pitch': 0.7853981633974483
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Select model
        # ------------
        if not isinstance(self.parameters['vehicle_model'], str):
            raise TypeError("Vehicle model must be defined as a string.")
        if self.parameters['vehicle_model'].lower() == 'djis1000':
            self.parameters['mass'], self.parameters['geom'], self.parameters['dynamics'] = vehicles.DJIS1000(self.parameters['vehicle_payload'], self.parameters['gravity'])
        elif self.parameters['vehicle_model'].lower() == 'tarot18':
            self.parameters['mass'], self.parameters['geom'], self.parameters['dynamics'] = vehicles.TAROT18(self.parameters['vehicle_payload'], self.parameters['gravity'])
        else:
            raise ValueError("Specified vehicle type is not supported. Only 'tarot18' and 'djis1000' are currently supported.")
        
        # Steady-state input value: hover, [weight, 0, 0, 0]
        if self.parameters['steadystate_input'] is None:
            self.parameters['steadystate_input'] = self.parameters['mass']['total'] * self.parameters['gravity']
        
        # Introduction of Aerodynamic effects:
        self.parameters['aero'] = {
          'drag': aero.DragModel(
            bodyarea=self.parameters['dynamics']['aero']['ad'],
            Cd=self.parameters['dynamics']['aero']['cd'],
            air_density=self.parameters['air_density']),
          'lift': None}

    def dx(self, x, u):
        # Extract useful values
        m = self.parameters['mass']['total']  # vehicle mass
        Ixx, Iyy, Izz = self.parameters['mass']['Ixx'], self.parameters['mass']['Iyy'], self.parameters['mass']['Izz']  # vehicle inertia

        # Input vector
        T = u['T']  # Thrust (along body z)
        tp = u['mx']  # Moment along body x
        tq = u['my']  # Moment along body y
        tr = u['mz']  # Moment along body z

        # Extract state variables from current state vector
        phi, theta, psi = x['phi'], x['theta'], x['psi']
        vx_a, vy_a, vz_a = x['vx'], x['vy'], x['vz']
        p, q, r = x['p'], x['q'], x['r']

        # Pre-compute Trigonometric values
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)

        # Compute drag forces
        v_earth = np.array([vx_a, vy_a, vz_a]).reshape((-1,))  # velocity in Earth-fixed frame
        v_body = np.dot(
          geom.rot_eart2body_fast(
            sin_phi,
            cos_phi,
            sin_theta,
            cos_theta,
            sin_psi,
            cos_psi),
          v_earth)  # Velocity in body-axis
        fb_drag = self.parameters['aero']['drag'](v_body)   # drag force in body axis
        fe_drag = np.dot(
          geom.rot_body2earth_fast(
            sin_phi,
            cos_phi,
            sin_theta,
            cos_theta,
            sin_psi,
            cos_psi),
          fb_drag)  # drag forces in Earth-fixed frame
        fe_drag[-1] = np.sign(v_earth[-1]) * np.abs(fe_drag[-1])  # adjust vertical (z=Up) force according to velocity in fixed frame

        # Update state vector
        # -------------------
        dxdt = np.zeros((len(x),))

        dxdt[0] = vx_a    # x-position increment (airspeed along x-direction)
        dxdt[1] = vy_a    # y-position increment (airspeed along y-direction)
        dxdt[2] = vz_a    # z-position increment (airspeed along z-direction)

        dxdt[3] = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta        # Euler's angle phi increment
        dxdt[4] = q * cos_phi - r * sin_phi                                    # Euler's angle theta increment
        dxdt[5] = q * sin_phi / cos_theta + r * cos_phi / cos_theta            # Euler's angle psi increment

        dxdt[6] = ((sin_theta * cos_psi * cos_phi + sin_phi * sin_psi) * T - fe_drag[0]) / m   # Acceleration along x-axis
        dxdt[7] = ((sin_theta * sin_psi * cos_phi - sin_phi * cos_psi) * T - fe_drag[1]) / m   # Acceleration along y-axis
        dxdt[8] = -self.parameters['gravity'] + (cos_phi * cos_theta * T - fe_drag[2]) / m   # Acceleration along z-axis

        dxdt[9] = ((Iyy - Izz) * q * r + tp * self.parameters['geom']['arm_length']) / Ixx  # Angular acceleration along body x-axis: roll rate
        dxdt[10] = ((Izz - Ixx) * p * r + tq * self.parameters['geom']['arm_length']) / Iyy  # Angular acceleration along body y-axis: pitch rate
        dxdt[11] = ((Ixx - Iyy) * p * q + tr * 1) / Izz  # Angular acceleration along body z-axis: yaw rate
        dxdt[12] = 1     # Auxiliary time variable
        dxdt[13] = (u['mission_complete'] - x['mission_complete']) / self.parameters['dt']    # Value to keep track of percentage of mission completed

        return self.StateContainer(np.array([np.atleast_1d(item) for item in dxdt]))

    def event_state(self, x) -> dict:
        # Based on percentage of reference trajectory completed
        return {'TrajectoryComplete': x['mission_complete']}
 
    def output(self, x):
        # Output is the same as the state vector, without time and mission_complete
        return self.OutputContainer(x.matrix[0:-2])

    def threshold_met(self, x) -> dict:
        # Progress through the reference trajectory is saved in the state 'mission_complete'
        return {'TrajectoryComplete': x['mission_complete'] >= 1}

    def simulate_to_threshold(self, future_loading_eqn, first_output=None, events=None, **kwargs):
        # Check for appropriately defined dt - must be same as vehicle model
        if 'dt' in kwargs and kwargs['dt'] != self.parameters['dt']:
          kwargs['dt'] = self.parameters['dt']
          warn(f"Simulation dt must be equal to dt defined for the vehicle model. dt = {self.parameters['dt']} is used.")
        elif 'dt' not in kwargs:
          kwargs['dt'] = self.parameters['dt']

        # Simulate to threshold
        sim_res = super().simulate_to_threshold(future_loading_eqn, first_output, events, **kwargs)

        return sim_res

    def linear_model(self, phi, theta, psi, p, q, r, T):
        """ 
        Linearized model of the small rotorcraft 6-dof.
        The function returns the state-transition matrix A and the input matrix B that forms the linearized model as:
        dx/dt = A x + B u
        where x is the state vector in state-space form (namely x, \dot{x}), u is the input vector containing the thrust and three moments around the vehicle's main body axes.
        To generate the linearized matrices, only the attitude angles, angular velocities, and thrust are necessary,
        since the model model ignores gyroscopic effect and wind rate of change. 
            
        :param phi:       rad, scalar, double, first Euler's angle
        :param theta:     rad, scalar, double, second Euler's angle
        :param psi:       rad, scalar, double, third Euler's angle
        :param p:         rad/s, scalar, double, roll rate of change
        :param q:         rad/s, scalar, double, pitch rate of change
        :param r:         rad/s, scalar, double, yaw rate of change
        :param T:         N, scalar, double, thrust
        :return:          Linearized state transition matrix A, n_states x n_states, and linearized input matrix B, n_states x n_inputs
        """
        m = self.parameters['mass']['total']
        Ixx, Iyy, Izz = self.parameters['mass']['Ixx'], self.parameters['mass']['Iyy'], self.parameters['mass']['Izz']
        length = self.parameters['geom']['arm_length']
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)


        A = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, q*cos_phi*tan_theta - r*sin_phi*tan_theta, q*(tan_theta**2 + 1)*sin_phi + r*(tan_theta**2 + 1)*cos_phi, 0, 0, 0, 0, 1, sin_phi*tan_theta, cos_phi*tan_theta],
                      [0, 0, 0, -q*sin_phi - r*cos_phi, 0, 0, 0, 0, 0, 0, cos_phi, -sin_phi],
                      [0, 0, 0, q*cos_phi/cos_theta - r*sin_phi/cos_theta, q*sin_phi*sin_theta/cos_theta**2 + r*sin_theta*cos_phi/cos_theta**2, 0, 0, 0, 0, 0, sin_phi/cos_theta, cos_phi/cos_theta],
                      [0, 0, 0, T*(-sin_phi*sin_theta*cos_psi + sin_psi*cos_phi)/m, T*cos_phi*cos_psi*cos_theta/m, T*(sin_phi*cos_psi - sin_psi*sin_theta*cos_phi)/m, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, T*(-sin_phi*sin_psi*sin_theta - cos_phi*cos_psi)/m, T*sin_psi*cos_phi*cos_theta/m, T*(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)/m, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -T*sin_phi*cos_theta/m, -T*sin_theta*cos_phi/m, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r*(Iyy - Izz)/Ixx, q*(Iyy - Izz)/Ixx],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, r*(-Ixx + Izz)/Iyy, 0, p*(-Ixx + Izz)/Iyy],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, q*(Ixx - Iyy)/Izz, p*(Ixx - Iyy)/Izz, 0]])

        B = np.array([ [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)/m, 0, 0, 0],
                       [(-sin_phi*cos_psi + sin_psi*sin_theta*cos_phi)/m, 0, 0, 0],
                       [cos_phi*cos_theta/m, 0, 0, 0],
                       [0, length/Ixx, 0, 0],
                       [0, 0, length/Iyy, 0],
                       [0, 0, 0, 1.0/Izz]])

        return A, B

    def visualize_traj(self, pred, ref=None, prefix='', fig=None, pred_cfg={'linewidth': 2.0, 'alpha': 0.6, 'color': 'tab:blue', 'linestyle':'-', 'label':'predicted'}, ref_cfg={'linewidth': 2.0, 'alpha': 0.6, 'color': 'tab:orange', 'linestyle':'--', 'label':'reference'}):
        """
        This method provides functionality to visualize a predicted trajectory generated, plotted with the reference trajectory. 

        Calling this returns a figure with two subplots: 1) x vs y, and 2) z vs time.

        Args
        ----------
        pred : SimulationResults
              Results from vehicle model simulation from simulate_to or simulate_to_threshold for a defined SmallRotorcraft class

        Keyword Args
        -------------
        ref : dict[str, np.ndarray], optional
              Reference trajectory - dict with keys for each state in the vehicle model and corresponding values as numpy arrays
        prefix : str, optional
              Prefix added to keys in predicted values. This is used to plot the trajectory using the results from a composite model
        pred_cfg : dict, optional
              Configuration for the prediction line on the graphs. See matplotlib.pyplot.plot documentation for more details
        ref_cfg : dict, optional
              Configuration for the reference line (if provided) on the graphs. See matplotlib.pyplot.plot documentation for more details
        fig : TrajectoryFigure, optional
              Figure where the additional diagrams are to be added. Creates a new figure if not provided

        Returns
        -------
        TrajectoryFigure : Visualization of trajectory generation results 
        """

        if fig is None:
            fig = plt.figure(FigureClass=TrajectoryFigure)
        elif not isinstance(fig, TrajectoryFigure):
            raise TypeError(f"fig must be a TrajectorFigure, was {type(fig)}")

        # Handle reference information
        if ref is not None:
          # Extract reference trajectory information
          time        = ref['t'].tolist() 
          ref_x       = ref['x'].tolist() 
          ref_y       = ref['y'].tolist() 
          ref_z       = ref['z'].tolist()

          # Plot reference trajectories
          fig.plot_traj(ref_x, ref_y, **ref_cfg)
          fig.plot_alt(time, ref_z, **ref_cfg)

        # Extract predicted trajectory information
        pred_x = [pred.outputs[iter][prefix+'x'] for iter in range(len(pred.times))]
        pred_y = [pred.outputs[iter][prefix+'y'] for iter in range(len(pred.times))]
        pred_z = [pred.outputs[iter][prefix+'z'] for iter in range(len(pred.times))]
        
        # Plot predictions
        fig.plot_traj(pred_x, pred_y, **pred_cfg)
        fig.plot_alt(pred.times, pred_z, **pred_cfg)

        # Final formatting
        fig.get_axes()[0].legend(fontsize=14)

        return fig
        
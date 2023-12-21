# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Auxiliary functions for trajectories and aircraft routes
"""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from warnings import warn

from progpy.utils.traj_gen import geometry as geom
from progpy.utils.traj_gen.nurbs import NURBS

DEG2RAD: float = (np.pi/180.0)


class TrajectoryFigure(Figure):
    """
    Figure visualizing a trajectory
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (ax_traj, ax_alt) = self.subplots(2)
        ax_traj.set_xlabel('x', fontsize=14)
        ax_traj.set_ylabel('y', fontsize=14)
        
        ax_alt.set_xlabel('time stamp, -', fontsize=14)
        ax_alt.set_ylabel('z', fontsize=14)
        
    def plot_traj(self, x, y, **kwargs) -> None:
        """
        Plot the trajectory in 2d space (upper graph)

        Args:
            x (list[float]): x location
            y (list[float]): y location
        """
        ax_traj = self.axes[0]
        ax_traj.plot(x, y, **kwargs)

    def plot_alt(self, t, z, **kwargs) -> None:
        """
        Plot the altitude vs time (lower graph)

        Args:
            t (list[float]): Times
            z (list[float]): z location
        """
        ax_alt = self.axes[1]
        ax_alt.plot(t, z, **kwargs)

def compute_derivatives(position_profile, timevec):
    # Compute derivatives of position: velocity and acceleration
    # (optional: jerk, not needed)
    dim_keys = list(position_profile.keys())
    vel_interp = {dim_key: None for dim_key in dim_keys}
    acc_interp = {dim_key: None for dim_key in dim_keys}
    for key in dim_keys:
        vel_interp[key], acc_interp[key], _ = derivate_position(
            position_profile[key],
            timevec[1]-timevec[0])
    
    return {'velocity': vel_interp, 'acceleration': acc_interp}


def linearinterp_t(t0, x0, t1, x1, xp):
    """
    Linear interpolation of input xp given two points (t0, x0), (t1, x1)
    :param t0:          independent variable of first point
    :param x0:          dependent variable of first point
    :param t1:          independent variable of second point
    :param x1:          dependent variable of second point
    :param xp:          independent variable of query point
    """
    dx = x1 - x0
    dt = t1 - t0
    der = dx / dt
    t_land = 1.0/der * (xp + der * t0 - x0)
    return t_land


def reshape_route_attribute(x, dim=None, msk=None):
    """
    Reshape route attribute to ensure correct length

    :param x:           attribute to reshape
    :param dim:         scalar, int or None, desired length of attribute. Used only if x has no attribute "len"
    :param msk:         list or array of int, or None, mask to reshape attribute x. The function return x by repeating x[msk] at msk points.
    :return:            reshaped attribute
    """
    if not hasattr(x, "__len__"):
        return x * np.ones((dim, ))
    elif msk:
        return np.insert(x, msk, x[msk])
    

def angular_vel_from_attitude(phi, theta, psi, delta_t=1):
    """
    Compute angular velocities from attitudes. The function computes the body angular rates p, q, r given the 
    desired Euler's angles parameterized by time: phi(t), theta(t), psi(t), and their corresponding rates.

    :param phi:         double, n x 1, first Euler's angle as a function of time.
    :param theta:       double, n x 1, second Euler's angle as a function of time.
    :param psi:         double, n x 1, third Euler's angle as a function of time.
    :param delta_t:     double, scalar, time step size, (default=1).
    :return p:          double, n x 1, body roll rate as a function of time
    :return q:          double, n x 1, body pitch rate as a function of time
    :return r:          double, n x 1, body yaw rate as a function of time
    """
    phidot = np.insert(np.diff(phi) / delta_t, 0, 0.0)
    thetadot = np.insert(np.diff(theta) / delta_t, 0, 0.0)
    psidot = np.insert(np.diff(psi) / delta_t, 0, 0.0)
    p = np.zeros_like(phidot)
    q = p.copy()
    r = p.copy()
    for ii in range(len(phi)):
        des_angular_vel = geom.body_ang_vel_from_eulers(
            phi[ii], theta[ii], psi[ii], phidot[ii], thetadot[ii], psidot[ii])
        p[ii], q[ii], r[ii] = des_angular_vel[0], des_angular_vel[1], des_angular_vel[2]
    return p, q, r


def derivate_position(p, dt):
    """
    Given a position profile as a function of time, p = p(t), derivate it up to 3 times to obtain velocity, acceleration, and jerk

    For simplicity of computation as part of a trajectory profile, the derivatives are force to start at 0, e.g.,  velocity[0] = 0.0

    :param p:           n x 1 array, doubles, position profile as a function of time
    :param dt:          scalar, double, time step size used to generate the position profile, which also serves as derivative increment
    :return:            three n x 1 arrays of doubles in a list, corresponding to velocity, acceleration and jerk, respectively.
    """
    v = np.zeros_like(p)
    a = v.copy()
    j = v.copy()

    v = np.gradient(p, dt)
    v[0] = 0.
    a = np.gradient(v, dt)
    a[0] = 0.
    j = np.gradient(j, dt)
    j[0] = 0.
    return v, a, j


def gen_attitude(psi, ax, ay, az, max_phi, max_theta, gravity):
    """Calculate angular kinematics based on acceleration and yaw"""
    # linearized angular kinematics
    phi = 1.0 / (gravity + az) * (ax * np.sin(psi) - ay * np.cos(psi))
    theta = 1.0 / (gravity + az) * (ax * np.cos(psi) + ay * np.sin(psi))
    
    # Introduce limits on attitude angles
    phi = np.fmax(np.fmin(phi, max_phi), -max_phi)
    theta = np.fmax(np.fmin(theta, max_theta), -max_theta)
    return phi, theta, psi


class Trajectory():
    """
    Trajectory class.

    This class generates a trajectory object that is used to create a smooth profile including
    position, velocity, acceleration, Euler's angles, and attitude rates.
    The profiles are used to generate the desired state of a vehicle that needs to be followed to complete the trajectory.
    
    args:
            lat (np.ndarray):
                rad, n x 1 array, doubles, latitude coordinates of waypoints, deg
            lon (np.ndarray):
                rad, n x 1 array, doubles, longitude coordinates of waypoints, deg
            alt (np.ndarray):
                m, n x 1 array, doubles, altitude coordinates of waypoints, m
            takeoff_time (float, optional):
                take off time of the trajectory. Default is None (starting at current time).
            etas (list[float], optional):
                ETAs of each waypoints. Default is None. In that case, the ETAs are calculated based on the desired speed between waypoints.

    kwargs:
            gravity (float, optional):
                Magnitude of force of gravity. Default 9.81 m/s^2
            max_phi (float, optional):
                Maximum Euler's angle (rad). Default 45/180.0*np.pi
            max_theta (float, optional):
                maximum Euler's angle (rad). Default 45/180.0*np.pi
            max_iter (int, optional):
                maximum number of iterations to adjust the trajectory according to the maximum average jerk. Default: 10
            max_avgjerk (float, optional):
                maximum average Jerk allowed (m/s^3), if the generated trajectory has a higher value, new, more forgiving ETAs are generated to satisfy this constraint. Default: 20
            nurbs_order (int, optional):
                order of the NURBS used to generate the position profile. Default: 4
            waypoint_weight (int, optional):
                default weight value associated to all the waypoints. Default: 20
            weight_vector (np.ndarray, optional):
                Waypoint weight vector. Default is None (all waypoints will have same value from 'waypoint_weight'). If passed, the user can define different weight for each waypoints, such that some waypoints will be approached more closely and some will be approached more smoothly.
            nurbs_basis_length (int, optional):
                default length of basis function used to generate the position profile with the NURBS.Default: 1000
            cruise_speed (float, optional):
                desired cruise speed between waypoints (m/s). If ETAs are provided, this value is ignored. Default = 6.0.
            ascent_speed (float, optional):
                desired ascent speed (m/s) between waypoints. If ETAs are provided, this value is ignored. Defualt: 3
            descent_speed (float, optional):
                desired descent speed (m/s) between waypoints. If ETAs are provided, this value is ignored. Default: 3
            landing_speed (float, optional):
                desired landing speed (m/s) between waypoints. This speed is used when the vehicle's altitude is lower than 'landing_altitude' parameter. If ETAs are provided, this value is ignored. Default: 1.5
            landing_altitude (float, optional):
                landing altitude (m) below which the vehicle is supposed to move at 'landing_speed'. Default: 10.5
    """
    def __init__(self,
                 lat,
                 lon,
                 alt,
                 takeoff_time=None,
                 etas=None,
                 **kwargs):

        # Check waypoint types:
        # ---------------------
        if not isinstance(lat, np.ndarray) or not isinstance(lon, np.ndarray) or not isinstance(alt, np.ndarray):
            raise TypeError("Latitudes, longitudes, and altitudes must be provided as n x 1 arrays.")
        if lat.shape != lon.shape or lon.shape != alt.shape:
            raise ValueError("Provided latitude, longitude, and altitude arrays must be the same length.")
        if lat.shape[0] <= 1 or lon.shape[0] <= 1 or alt.shape[0] <= 1:
            raise ValueError("Latitudes, longitudes, and altitudes must be provided as n x 1 arrays, with n > 1.")
        if etas is not None:
            if not isinstance(etas, list):
                raise TypeError("ETAs must be provided as a list of floats.")
            if len(etas) != 1 and len(etas) != lat.shape[0]:
                raise ValueError("ETA must be either a take off time (one value), or a vector array with same length as lat, lon and alt.")
            for eta in etas:
                if not isinstance(eta, (float, int)):
                    raise TypeError("ETAs must be provided as a list of floats.")
        if takeoff_time is not None and not isinstance(takeoff_time, (float, int)):
            raise TypeError("Takeoff time must be provided as a float.")

        self.trajectory = {}
        self.waypoints = {
            'lat': [elem*DEG2RAD for elem in lat],
            'lon': [elem*DEG2RAD for elem in lon],
            'alt': alt,
            'takeoff_time': takeoff_time,
            'eta': etas,
            'x': None,
            'y': None,
            'z': None,
            'heading': None}
        
        # Assign takeoff time
        if self.waypoints['takeoff_time'] is None:
            if self.waypoints['eta'] is not None:
                self.waypoints['takeoff_time'] = self.waypoints['eta'][0]
            else:
                self.waypoints['takeoff_time'] = 0

        self.waypoints['heading'] = geom.gen_heading_angle(
            self.waypoints['lat'],
            self.waypoints['lon'],
            self.waypoints['alt'])

        # Set up coordinate system conversion between Geodetic,
        # Earth-Centric Earth-Fixed (ECF), and Cartesian (East-North-Up, ENU)
        self.coordinate_system = geom.Coord(
            lat[0],
            lon[0],
            alt[0])

        # Define speed parameters - only necessary if ETAs are not defined
        if etas is not None and ('cruise_speed' in kwargs or 'ascent_speed' in kwargs or 'descent_speed' in kwargs or 'landing_speed' in kwargs):
            warn("Speed values are ignored since ETAs were specified. To define speeds (cruise, ascent, descent, landing) instead, do not specify ETAs.")
        if etas is None and ('cruise_speed' not in kwargs or 'ascent_speed' not in kwargs or 'descent_speed' not in kwargs or 'landing_speed' not in kwargs):
            warn("Neither ETAs nor speeds were defined. Default speeds will be used.")
        self.speed_parameters = {
            'cruise_speed': 6.0,
            'ascent_speed': 3.0,
            'descent_speed': 3.0,
            'landing_speed': 1.5,
            'landing_altitude': 10.5}
        self.speed_parameters.update(**kwargs)

        # Set landing waypoints dimensions
        idx_land_pos = self.set_landing_waypoints()

        # Set ETAs for waypoints
        self.set_eta(idx_land_pos=idx_land_pos)

        # Covert to cartesian coordinates
        self.waypoints['x'], \
            self.waypoints['y'], \
            self.waypoints['z'] = self.coordinate_system.geodetic2enu(
                self.waypoints['lat'],
                self.waypoints['lon'],
                self.waypoints['alt'])
                
        # Interpolation properties
        self.parameters = {'gravity': 9.81,
                           'max_phi': 0.25*np.pi,
                           'max_theta': 0.25*np.pi,
                           'max_iter': 10,
                           'max_avgjerk': 20.0,
                           'nurbs_order': 4,
                           'waypoint_weight': 20,
                           'weight_vector': None,
                           'nurbs_basis_length': 1000}
        self.parameters.update(**kwargs)
        if self.parameters['weight_vector'] is None:
            self.parameters['weight_vector'] = np.asarray([self.parameters['waypoint_weight'],] * len(self.waypoints['x']))

    @property
    def ref_traj(self,):
        return {
            'x': self.trajectory['position'][:, 0],
            'y': self.trajectory['position'][:, 1],
            'z': self.trajectory['position'][:, 2],
        
            'phi': self.trajectory['attitude'][:, 0],
            'theta': self.trajectory['attitude'][:, 1],
            'psi': self.trajectory['attitude'][:, 2],
        
            'vx': self.trajectory['velocity'][:, 0],
            'vy': self.trajectory['velocity'][:, 1],
            'vz': self.trajectory['velocity'][:, 2],
        
            'p': self.trajectory['angVel'][:, 0],
            'q': self.trajectory['angVel'][:, 1],
            'r': self.trajectory['angVel'][:, 2],
            
            't': self.trajectory['time']}

    def plot(self, fig=None, **kwargs) -> TrajectoryFigure:
        """
        Plot the reference trajectory

        Args:
            fig (TrajectoryFigure, optional): Figure where the additional diagrams are to be added. Creates a new figure if not provided

        Raises:
            TypeError: if Figure is not a TrajectoryFigure

        Returns:
            TrajectoryFigure
        """
        params = {'figsize': (13, 9), 'linewidth': 2.0, 'alpha_preds': 0.6}
        params.update(kwargs)
        if fig is None:
            fig = plt.figure(FigureClass=TrajectoryFigure)
        elif not isinstance(fig, TrajectoryFigure):
            raise TypeError(f"fig must be a TrajectorFigure, was {type(fig)}")

        x = self.trajectory['position'][:, 0].tolist()
        y = self.trajectory['position'][:, 1].tolist()
        z = self.trajectory['position'][:, 2].tolist()
        t = self.trajectory['time'].tolist()

        fig.plot_traj(x, y, **params)
        fig.plot_alt(t, z, **params)
    
    def compute_attitude(self, heading_profile, acceleration_profile, timestep_size):
        """
        Compute attitude defined by Euler's angles as a function of time given heading and acceleration profiles.
        Phi and theta are limited by max_phi, max_theta, and the angular velocities p, q, r are calculated
        from phi, theta, psi given the time step size.

        :param heading_profile:             n x 1 array, double, heading angle as a function of time
        :param acceleration_profile:        n x 3 array, double, acceleration along the three cartesian directions as a function of time
        :param timestep_size:               scalar, double, time step size dt
        :param max_phi:                     rad, scalar, maximum phi angle possible
        :param max_theta:                   rad, scalar, maximum theta angle possible
        :return:                            dictionary containing attitude phi, theta, psi and angular velocity p, q, r, as a function of time.
        """
        # get names of cartesian directions
        dim_keys = list(acceleration_profile.keys())

        # Compute attitude
        # ---------------
        phi, theta, psi = gen_attitude(heading_profile,
                                       acceleration_profile[dim_keys[0]],
                                       acceleration_profile[dim_keys[1]],
                                       acceleration_profile[dim_keys[2]],
                                       self.parameters['max_phi'],
                                       self.parameters['max_theta'],
                                       self.parameters['gravity'])
        # Compute angular velocity
        # -------------------------
        p, q, r = angular_vel_from_attitude(phi, theta, psi, timestep_size)

        return {
            'attitude': np.array([phi, theta, psi]).T,
            'angVel': np.array([p, q, r]).T}

    def compute_trajectory_nurbs(self, dt) -> None:
        """
        Compute position and yaw profiles with NURBS

        Args:
            dt (float): time step
        """
        # Instantiate NURBS class to generate trajectory
        points = {
            'x': self.waypoints['x'],
            'y': self.waypoints['y'],
            'z': self.waypoints['z']}
        nurbs_alg = NURBS(
            points=points,
            weights=self.parameters['weight_vector'],
            times=self.waypoints['eta'] - self.waypoints['eta'][0],
            yaw=self.waypoints['heading'],
            order=self.parameters['nurbs_order'],
            basis_length=self.parameters['nurbs_basis_length'])
        
        # Generate position and yaw interpolated given the timestep
        pos_interp, yaw_interp, time_interp = nurbs_alg.generate(timestep_size=dt)
        
        # Generate velocity, acceleration, and jerk (optional) profile
        # from position profile
        linear_profiles = compute_derivatives(pos_interp, time_interp)
        
        # Generate angular profiles:
        # attitude and angular velocities from heading and acceleration
        angular_profiles = self.compute_attitude(
            heading_profile=yaw_interp,
            acceleration_profile=linear_profiles['acceleration'],
            timestep_size=dt)
        
        # Store as attribute
        self.trajectory = {**{'position': np.vstack(list(pos_interp.values())).T},
                           **{'velocity': np.vstack(list(linear_profiles['velocity'].values())).T},
                           **{'acceleration': np.vstack(list(linear_profiles['acceleration'].values())).T},
                           **angular_profiles,
                           **{'time': time_interp}}

    def generate(self, dt, **kwargs):
        """
        Generate trajectory given the waypoints and desired velocity or ETAs.
        The function requires the time step size dt, in seconds, used to interpolate the waypoints and generate the trajectory.

        :param dt:          s, scalar, time step size used to interpolate the waypoints and generate the trajectory
        :return:            dictionary of state variables describing the trajectory as a function of time
        """
        self.parameters.update(**kwargs)  # Override NURBS parameters
        assert len(self.parameters['weight_vector']) == len(self.waypoints['x']), "Length of waypoint weight vector and number of waypoints must coincide."

        self.compute_trajectory_nurbs(dt)
        self.__adjust_eta_given_max_acceleration(dt)

        # Convert trajectory into geodetic coordinates
        # --------------------------------------------
        self.trajectory['geodetic_pos'] = np.zeros_like(self.trajectory['position'])
        self.trajectory['geodetic_pos'][:, 0], \
            self.trajectory['geodetic_pos'][:, 1], \
                self.trajectory['geodetic_pos'][:, 2] = self.coordinate_system.enu2geodetic(
                    self.trajectory['position'][:, 0],
                    self.trajectory['position'][:, 1],
                    self.trajectory['position'][:, 2])
        return self.ref_traj

    def __adjust_eta_given_max_acceleration(self, dt) -> None:
        """
        Adjusting the trajectory computed by the NURBS algorithm according to whether the maximum 
        jerk exceeds the limit allowed. This prevents the vehicle to crash because of high accelerations during the flight
        """
        maxiter = self.parameters['max_iter']
        
        keep_adjusting = True
        counter = 0
        positions = {'x': self.waypoints['x'], 'y': self.waypoints['y'], 'z': self.waypoints['z']}
        etas_rel = self.waypoints['eta'] - self.waypoints['eta'][0]

        while keep_adjusting:
            m = len(self.trajectory['time'])
            counter += 1
            new_eta_rel = etas_rel.copy()    # store eta to be updated, (relative eta)
            keep_adjusting = False
            position_values = np.vstack(list(positions.values())).T
            n = position_values.shape[0]
            for i in range(n-1):
                
                dist1 = geom.euclidean_distance_point_vector(position_values[i, :], self.trajectory['position'])
                dist2 = geom.euclidean_distance_point_vector(position_values[i+1, :], self.trajectory['position'])
                
                delta_etas = etas_rel[i+1] - etas_rel[i]
                dtime1 = 100.0 * np.abs(etas_rel[i] - self.trajectory['time'])**3.0  # time is fundamental and more important than distance, so I'm using a higher power to take that into account
                dtime2 = 100.0 * np.abs(etas_rel[i+1] - self.trajectory['time'])**3.0

                accelerations_abs = np.abs(self.trajectory['acceleration'][max(np.argmin(dist1 * dtime1), 0):min(np.argmin(dist2 * dtime2), m), :])

                if accelerations_abs.size > 0:
                    acc_max = np.amax(accelerations_abs)
                    if acc_max/delta_etas > self.parameters['max_avgjerk']:
                        extra_time = acc_max / self.parameters['max_avgjerk'] + 2.0
                        new_eta_rel[i+1:] += extra_time
                        keep_adjusting = True

            if any(etas_rel != new_eta_rel):
                # Generate new curve and new_eta becomes the reference eta
                self.waypoints['eta'] = new_eta_rel + self.waypoints['eta'][0]
                self.compute_trajectory_nurbs(dt)
                etas_rel = new_eta_rel.copy()

            if counter == maxiter:
                # Exit the while loop and give up if in maxiter iterations the trajectory acceleration has not been pushed under the limit
                warn("max number of iterations reached, the trajectory still contains accelerations beyond the limit.")
                break

    def set_landing_waypoints(self,):
        """
        Set waypoints at altitude defined by landing_altitude.
        By so doing, the trajectory can split the ascending and descending phases into segments, and assign the landing_speed to all segments that
        fall below the landing_alt mark.
        :return:                        int, n x 1 array, indices of waypoints that define the landing.
        """
        
        # get boolean flag where altitude is below the landing altitude, and get the corresponding indices
        idx_land = np.asarray(self.waypoints['alt'] < self.speed_parameters['landing_altitude'])
        idx_land_pos = np.where(idx_land)[0]    
        
        # if there are waypoints below landing altitude: append a landing point (with altitude self.speed_parameters['landing_altitude']) accordingly.
        if idx_land_pos.size != 0:
            n_waypoints = len(self.waypoints['lat'])
            if hasattr(self.waypoints['eta'], '__len__'):
                n_times = len(self.waypoints['eta'])
            else:
                n_times = 0
            counter = 0
            for item in idx_land_pos:
                if item == 0:   # if first element is below, just add a landing waypoint
                    self.waypoints['lat'] = np.insert(self.waypoints['lat'], item + 1, self.waypoints['lat'][item])
                    self.waypoints['lon'] = np.insert(self.waypoints['lon'], item + 1, self.waypoints['lon'][item])
                    if n_times > 1:
                        t_land = linearinterp_t(self.waypoints['eta'][item], 
                                                self.waypoints['alt'][item], 
                                                self.waypoints['eta'][item+1], 
                                                self.waypoints['alt'][item+1], # just added the landing altitude, so I need to take the next point
                                                self.speed_parameters['landing_altitude'])
                        self.waypoints['eta'] = np.insert(self.waypoints['eta'], item + 1, t_land)
                    self.waypoints['alt'] = np.insert(self.waypoints['alt'], item + 1, self.speed_parameters['landing_altitude']*1.0)
                    counter += 1

                elif item == n_waypoints-1:  # if one before the last element, add landing waypoints right before landing.
                    if self.waypoints['alt'][item+counter-1] > self.speed_parameters['landing_altitude']:
                        self.waypoints['lat'] = np.insert(self.waypoints['lat'], -1, self.waypoints['lat'][item+counter-1])
                        self.waypoints['lon'] = np.insert(self.waypoints['lon'], -1, self.waypoints['lon'][item+counter-1])
                        if n_times > 1:
                            t_land = linearinterp_t(self.waypoints['eta'][item+counter-1], 
                                                    self.waypoints['alt'][item+counter-1], 
                                                    self.waypoints['eta'][-1], 
                                                    self.waypoints['alt'][-1], 
                                                    self.speed_parameters['landing_altitude'])
                            self.waypoints['eta'] = np.insert(self.waypoints['eta'], -1, t_land)
                        self.waypoints['alt'] = np.insert(self.waypoints['alt'], -1, self.speed_parameters['landing_altitude']*1.0)
                        counter += 1
                else:
                    if self.waypoints['alt'][item+counter] - self.waypoints['alt'][item+counter-1] < 0:  
                        # descending
                        idx_delta = 0
                    else:
                        # ascending
                        idx_delta = 1
                    self.waypoints['lat'] = np.insert(self.waypoints['lat'], item+counter + idx_delta, self.waypoints['lat'][item+counter])
                    self.waypoints['lon'] = np.insert(self.waypoints['lon'], item+counter + idx_delta, self.waypoints['lon'][item+counter])
                    if n_times > 1:
                        t_land = linearinterp_t(self.waypoints['eta'][item+counter+idx_delta-1], 
                                                self.waypoints['alt'][item+counter+idx_delta-1], 
                                                self.waypoints['eta'][item+counter+idx_delta], 
                                                self.waypoints['alt'][item+counter+idx_delta], 
                                                self.speed_parameters['landing_altitude'])
                        self.waypoints['eta'] = np.insert(self.waypoints['eta'], item+counter+idx_delta, t_land)
                    self.waypoints['alt'] = np.insert(self.waypoints['alt'], item+counter+idx_delta, self.speed_parameters['landing_altitude']*1.0)
                    counter += 1
                    if idx_delta == 0:
                        # if descended, needs to go back up
                        if self.waypoints['alt'][item+counter+1] > self.speed_parameters['landing_altitude']:
                            self.waypoints['lat'] = np.insert(self.waypoints['lat'], item+counter+1, self.waypoints['lat'][item+counter+1])
                            self.waypoints['lon'] = np.insert(self.waypoints['lon'], item+counter+1, self.waypoints['lon'][item + counter+1])
                            self.waypoints['alt'] = np.insert(self.waypoints['alt'], item+counter+1, self.speed_parameters['landing_altitude']*1.0)
                            if self.waypoints['eta'] is not None:
                                t_land = linearinterp_t(self.waypoints['eta'][item+counter], 
                                                        self.waypoints['alt'][item+counter], 
                                                        self.waypoints['eta'][item+counter+1], 
                                                        self.waypoints['alt'][item+counter+1], 
                                                        self.speed_parameters['landing_altitude'])
                                self.waypoints['eta'] = np.insert(self.waypoints['eta'], item+counter+1, t_land)
                                
                            counter += 1
        
        # Recalculate landing positions with new waypoints:
        idx_land = np.asarray(self.waypoints['alt'] < self.speed_parameters['landing_altitude'])
        idx_land_pos = np.where(idx_land)[0]
        
        return idx_land_pos
    
    def set_eta(self, idx_land_pos, hovering=0) -> None:
        """
        Assign ETAs to waypoints
        If ETAS are provided (i.e., eta is not None), assign them.
        If they are not provided, compute them from desired speeds.
        index of the landing waypoints is necessary to reshape speed values.

        :param idx_land_pos:            -, m x 1 array. Index of the added landing waypoints.
        :param hovering:                s, scalar or n x 1 array. Default = 0. Hovering condition to add to the waypoints.
        """
        # Set speed dimensions
        n = len(self.waypoints['lat'])
        self.speed_parameters['cruise_speed'] = reshape_route_attribute(self.speed_parameters['cruise_speed'], dim=n-1, msk=idx_land_pos)
        self.speed_parameters['ascent_speed'] = reshape_route_attribute(self.speed_parameters['ascent_speed'], dim=n-1, msk=idx_land_pos)
        self.speed_parameters['descent_speed'] = reshape_route_attribute(self.speed_parameters['descent_speed'], dim=n-1, msk=idx_land_pos)
        self.speed_parameters['landing_speed'] = reshape_route_attribute(self.speed_parameters['landing_speed'], dim=n-1, msk=idx_land_pos)
        hovering = reshape_route_attribute(hovering, dim=n-1, msk=idx_land_pos)

        if self.waypoints['eta'] is None or len(self.waypoints['eta']) == 1:
            etas = None
        else:
            etas = self.waypoints['eta']
        
        # Compute ETAs
        if len(self.waypoints['alt']) <= 2:
            raise ValueError("At least 3 waypoints are required to compute ETAS from speed. Only {} were given.".format(len(self.lat)))
        
        # define margin on cruise speed
        # If calculated ETA produces a speed that is larger than desired speed, we can accommodate it as long as is within this margin (%)
        cruise_speed_margin = 0.1  # %, 'extra' speed we can tolerate on cruise.
        vert_speed_margin = 0.05   # %, 'extra' speed we can tolerate on ascent/descent

        # Compute relative ETAs
        alt_for_land = self.waypoints['alt'][1:]
        n = len(self.waypoints['alt'])-1
        
        if etas is None:
            d_eta = np.zeros((n,))
        else:
            d_eta = np.diff(etas)

        for point in range(n):  # this for loop could be improved by calculating distances all at once.
            dh, dv = geom.geodetic_distance([self.waypoints['lat'][point], self.waypoints['lat'][point+1]], 
                                            [self.waypoints['lon'][point], self.waypoints['lon'][point+1]], 
                                            [self.waypoints['alt'][point], self.waypoints['alt'][point+1]], 
                                            method='greatcircle', return_surf_vert=True)
            dv = dv[0]
            
            # Identify correct vertical speed
            if dv > 0 and alt_for_land[point] > self.speed_parameters['landing_altitude']:    
                vert_speed = self.speed_parameters['ascent_speed'][point]
            elif dv > 0 and alt_for_land[point] <= self.speed_parameters['landing_altitude']:   
                vert_speed = self.speed_parameters['landing_speed'][point]
            elif dv < 0 and alt_for_land[point] >= self.speed_parameters['landing_altitude']:   
                vert_speed = self.speed_parameters['descent_speed'][point]
            elif dv < 0 and alt_for_land[point] < self.speed_parameters['landing_altitude']:    
                vert_speed = self.speed_parameters['landing_speed'][point]
            else:                                                           
                vert_speed = 0.  # not moving vertically.
                
            if etas is None:
                # Define the correct speed:
                if np.isclose(dh + dv, 0.0):
                    d_eta[point] = 2.0  # if there's no vertical / horizontal speed (waypoints are identical) add a default hovering value of 2 s to avoid extreme accelerations.
                else:
                    if np.isclose(dh, 0.):
                        speed_sq = vert_speed*vert_speed
                    elif np.isclose(dv, 0.):
                        speed_sq = self.speed_parameters['cruise_speed'][point]**2.0
                    else:
                        speed_sq = self.speed_parameters['cruise_speed'][point]**2.0 + vert_speed*vert_speed
                    d_eta[point] = np.sqrt( (dh**2.0 + dv**2.0) / speed_sq ) * 1.3  # adding some %
                    
                    # If speed is larger than desired (possible when both dh, dv>0), increment d_eta to reduce until desired (consider margin)
                    while dh/d_eta[point] > (self.speed_parameters['cruise_speed'][point]*(1.+cruise_speed_margin)) or dv/d_eta[point] > (vert_speed*(1.+vert_speed_margin)):
                        d_eta[point] += 2.0
            else:
                # If speed is larger than maximum (possible when both dh, dv>0), increment d_eta to reduce until desired (consider margin)
                while dh / d_eta[point] > (self.speed_parameters['cruise_speed'][point]*(1.+cruise_speed_margin)) or dv / d_eta[point] > (vert_speed*(1.+vert_speed_margin)):
                    d_eta[point] += 2.0

            # Add hovering if desired
            if hovering[point] != 0:    
                d_eta[point] += hovering[point]

        eta_array = np.asarray(np.cumsum(np.insert(d_eta, 0, 0.0)))
        self.waypoints['eta'] = [eta + self.waypoints['takeoff_time'] for eta in eta_array]
        
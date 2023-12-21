# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest
import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
import warnings

from progpy.utils.traj_gen import Trajectory
from progpy.models.aircraft_model import SmallRotorcraft
from progpy.loading.controllers import LQR_I, LQR
from progpy.utils.traj_gen import geometry as geom

class TestUAVGen(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_reference_trajectory_generation(self):

        # Set warnings to temporarily act as exceptions
        warnings.simplefilter("error", category=UserWarning)

        # Define vehicle, necessary to pass to ref_traj
        vehicle = SmallRotorcraft()

        # Define waypoints 
        waypoints_dict = {}
        waypoints_dict['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719])
        waypoints_dict['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628])
        waypoints_dict['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints_dict['time_unix'] = [1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539]
        
        alt_in = waypoints_dict['alt_ft'] * 0.3048
        etas_in = waypoints_dict['time_unix']
        takeoff_time = etas_in[0]
        etas_wrong = [dt.datetime.fromtimestamp(waypoints_dict['time_unix'][ii]) for ii in range(len(waypoints_dict['time_unix']))]

        lat_small = np.array([waypoints_dict['lat_deg'][0]])
        lon_small = np.array([waypoints_dict['lon_deg'][0]])
        alt_small = np.array([alt_in[0]])
        etas_small = [etas_in[0]]

        # Incorrect number of input arguments to Trajectory
        with self.assertRaises(TypeError):
            # No waypoint information provided
            ref_traj = Trajectory()
        with self.assertRaises(TypeError):
            # Only subset of waypoint information provided
            ref_traj = Trajectory(lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(TypeError):
            # Only subset of waypoint information provided
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(TypeError):
            # Only subset of waypoint information provided
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], takeoff_time=takeoff_time, etas=etas_in)

        # Waypoints defined incorrectly 
        # Wrong type for waypoints 
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be numpy arrays
            ref_traj = Trajectory(lat='a', lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be numpy arrays
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=[1, 2, 3], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be numpy arrays
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=1, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be numpy arrays
            ref_traj = Trajectory(lat={'lat': 1}, lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; takeoff_time must be float/int
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=np.array([1]), etas=etas_in)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; takeoff_time must be float/int
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time='abc', etas=etas_in)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; etas must be list of float/int
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_wrong)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; etas must be list of float/int
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=np.array([1, 2, 3]))

        # Wrong lengths for waypoints
        with self.assertRaises(ValueError):
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'][:5], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(ValueError):
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'][2:4], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(ValueError):
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in[1:7], takeoff_time=takeoff_time, etas=etas_in)
        with self.assertRaises(ValueError):
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in[:5])
        with self.assertRaises(ValueError):
            ref_traj = Trajectory(lat=lat_small, lon=lon_small, alt=alt_small, etas=etas_small, takeoff=takeoff_time)

        # Checking correct combination of ETAs and speeds
        with self.assertRaises(UserWarning):
            # No ETAs or speeds provided, warning is thrown
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time)
        with self.assertRaises(UserWarning):
            # Both ETAs and speeds provided, warning is thrown
            params = {'cruise_speed': 1, 'descent_speed': 1, 'ascent_speed': 1, 'landing_speed': 1}
            ref_traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, takeoff_time=takeoff_time, etas=etas_in, **params)

        # Test trajectory generation functionality is generating an accurate result
        # Convert waypoints to Cartesian
        FEET2MET = 0.3048
        coord = geom.Coord(lat0=waypoints_dict['lat_deg'][0], lon0=waypoints_dict['lon_deg'][0], alt0=waypoints_dict['alt_ft'][0]*FEET2MET)
        x_ref, y_ref, z_ref = coord.geodetic2enu(waypoints_dict['lat_deg']*np.pi/180.0, waypoints_dict['lon_deg']*np.pi/180.0, waypoints_dict['alt_ft']*FEET2MET)
        time_ref = [waypoints_dict['time_unix'][iter] - waypoints_dict['time_unix'][0] for iter in range(len(waypoints_dict['time_unix']))]

        # Generate trajectory
        vehicle.parameters['dt'] = 1
        traj = Trajectory(lat=waypoints_dict['lat_deg'], lon=waypoints_dict['lon_deg'], alt=alt_in, etas=etas_in, takeoff_time=takeoff_time)
        ref_traj_test = traj.generate(dt=vehicle.parameters['dt'])

        # Check that generated trajectory is close to waypoints
        for ind, val in enumerate(time_ref):
            # Since dt is 1, the time values in ref_traj_test['t'] correspond to their index number
            self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)
            self.assertAlmostEqual(y_ref[ind], ref_traj_test['y'][val], delta=45)
            self.assertAlmostEqual(z_ref[ind], ref_traj_test['z'][val], delta=45)

        # Reset warnings
        warnings.simplefilter("default", category=UserWarning)

    def test_controllers_and_vehicle(self):

        # Controller and vehicle tests are combined so we only have to
        # generate the reference trajectory once

        # Set warnings to temporarily act as exceptions
        warnings.simplefilter("error", category=UserWarning)

        # Instantiate vehicles - one for each configuration
        vehicle = SmallRotorcraft(**{'dt': 0.1, 'process_noise': 0})
        vehicle_djis = SmallRotorcraft(**{'dt': 0.1, 'vehicle_model': 'djis1000', 'process_noise': 0})

        # Define waypoints 
        waypoints_dict = {}
        waypoints_dict['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719])
        waypoints_dict['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628])
        waypoints_dict['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints_dict['time_unix'] = [1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539]
        
        lat_in = waypoints_dict['lat_deg']
        lon_in = waypoints_dict['lon_deg']
        alt_in = waypoints_dict['alt_ft'] * 0.3048
        etas_in = waypoints_dict['time_unix']

        # Generate reference trajectory
        ref_traj_temp = Trajectory(lat=lat_in, lon=lon_in, alt=alt_in, etas=etas_in)
        ref_traj = ref_traj_temp.generate(dt=vehicle.parameters['dt'])

        # Controller tests: LQR
        # -----------------
        # Testing incorrect arguments:
        with self.assertRaises(TypeError):
            # Reference trajectory not defined as dict
            ctrl = LQR(x_ref='abc', vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Reference trajectory not defined as dict
            ctrl = LQR(x_ref=[1, 2, 3], vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect type for values in reference trajectory dict
            ctrl = LQR(x_ref={'x': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect type for values in reference trajectory dict
            ctrl = LQR(x_ref={'x': np.array([1, 2, 3]), 'y': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # No vehicle model given
            ctrl = LQR(x_ref=ref_traj)
        with self.assertRaises(TypeError):
            # No reference trajectory given
            ctrl = LQR(vehicle=vehicle)

        # Generate Controller
        ctrl = LQR(x_ref=ref_traj, vehicle=vehicle)

        # Controller tests: LQR_I
        # -----------------
        # Testing incorrect arguments:
        with self.assertRaises(TypeError):
            # Reference trajectory not defined as dict
            ctrl = LQR_I(x_ref='abc', vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Reference trajectory not defined as dict
            ctrl = LQR_I(x_ref=[1, 2, 3], vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect type for values in reference trajectory dict
            ctrl = LQR_I(x_ref={'x': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect type for values in reference trajectory dict
            ctrl = LQR_I(x_ref={'x': np.array([1, 2, 3]), 'y': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # No vehicle model given
            ctrl = LQR_I(x_ref=ref_traj)
        with self.assertRaises(TypeError):
            # No reference trajectory given
            ctrl = LQR_I(vehicle=vehicle)

        # Generate Controller
        ctrl_I = LQR_I(x_ref=ref_traj, vehicle=vehicle)

        # Testing appropriate input parameters: 
        with self.assertRaises(ValueError):
            vehicle_wrong = SmallRotorcraft(**{'vehicle_model': 'fakemodel'})
        with self.assertRaises(TypeError):
            vehicle_wrong = SmallRotorcraft(**{'vehicle_model': 1})
        with self.assertRaises(TypeError):
            vehicle_wrong = SmallRotorcraft(**{'vehicle_model': [1, 2, 3]})
        with self.assertRaises(TypeError):
            vehicle_wrong = SmallRotorcraft(**{'vehicle_model': np.array([1, 2, 3])})

        # Test simulation parameters:
        with self.assertRaises(UserWarning):
            # dt value for simulation must match dt provided to vehicle
            sim = vehicle.simulate_to(100, ctrl, **{'dt': 2})

        # Run simulation and verify accuracy of simulated trajectory:
        # Case 1: ETAs provided + LQR controller + tarot18 vehicle 
        # Use above ref_traj, ctrl
        # Simulate and compare
        sim = vehicle.simulate_to(ref_traj['t'][-1], ctrl, **{'dt': vehicle.parameters['dt'], 'save_freq': vehicle.parameters['dt']})
        x_temp = [sim.outputs[iter]['x'] for iter in range(len(sim.times))]
        y_temp = [sim.outputs[iter]['y'] for iter in range(len(sim.times))]
        z_temp = [sim.outputs[iter]['z'] for iter in range(len(sim.times))]
        x_sim_interp = interp1d(sim.times, x_temp)(ref_traj['t'])
        y_sim_interp = interp1d(sim.times, y_temp)(ref_traj['t'])
        z_sim_interp = interp1d(sim.times, z_temp)(ref_traj['t'])
        for iter in range(len(ref_traj['t'])):
            self.assertAlmostEqual(ref_traj['x'][iter], x_sim_interp[iter], delta=5)
            self.assertAlmostEqual(ref_traj['y'][iter], y_sim_interp[iter], delta=5)
            self.assertAlmostEqual(ref_traj['z'][iter], z_sim_interp[iter], delta=5)

        # Case 2: Speeds provided, no ETAs + LQR_I controller + djis1000 vehicle
        # Define speeds:
        ref_params = {
            'nurbs_order': 4,
            'cruise_speed': 8.0,
            'ascent_speed': 2.0,
            'descent_speed': 3.0,
            'landing_speed': 2,
        }

        # Generate reference trajectory
        ref_traj_speeds_temp = Trajectory(lat=lat_in, lon=lon_in, alt=alt_in, **ref_params)
        ref_traj_speeds = ref_traj_speeds_temp.generate(dt=vehicle_djis.parameters['dt'])

        # Build controller
        ctrl_speeds = LQR_I(x_ref=ref_traj_speeds, vehicle=vehicle_djis)

        # Simulate and compare:
        sim_speeds = vehicle_djis.simulate_to(ref_traj_speeds['t'][-1], ctrl_speeds, **{'save_freq': vehicle_djis.parameters['dt']})
        x_speeds_temp = [sim_speeds.outputs[iter]['x'] for iter in range(len(sim_speeds.times))]
        y_speeds_temp = [sim_speeds.outputs[iter]['y'] for iter in range(len(sim_speeds.times))]
        z_speeds_temp = [sim_speeds.outputs[iter]['z'] for iter in range(len(sim_speeds.times))]
        x_speeds_sim_interp = interp1d(sim_speeds.times, x_speeds_temp)(ref_traj_speeds['t'])
        y_speeds_sim_interp = interp1d(sim_speeds.times, y_speeds_temp)(ref_traj_speeds['t'])
        z_speeds_sim_interp = interp1d(sim_speeds.times, z_speeds_temp)(ref_traj_speeds['t'])
        for iter in range(len(ref_traj_speeds['t'])):
            self.assertAlmostEqual(ref_traj_speeds['x'][iter], x_speeds_sim_interp[iter], delta=8)
            self.assertAlmostEqual(ref_traj_speeds['y'][iter], y_speeds_sim_interp[iter], delta=8)
            self.assertAlmostEqual(ref_traj_speeds['z'][iter], z_speeds_sim_interp[iter], delta=8)

        # Reset warnings
        warnings.simplefilter("default", category=UserWarning)
    
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Trajectory Generation model")
    result = runner.run(load_test.loadTestsFromTestCase(TestUAVGen)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()


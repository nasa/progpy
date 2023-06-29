# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Vehicle Models - originally developed by Matteo Corbetta (matteo.corbetta@nasa.gov) for the SWS project
"""

import numpy as np
from warnings import warn

from progpy.models.aircraft_model.vehicles.control import allocation_functions as caf

# VEHICLE MODELS
def TAROT18(payload=0.0, gravity=9.81):
    """
    Tarot T18 Octocopter mass, geometry and dynamic properties.
    :param payload:         float, payload for the flight.
    :param gravity:         float, gravity value according to location
    :return:                dictionaries containing mass, geometry and dynamic properties of the vehicle
    """
    mass = dict(body_empty=3.2,             # kg, total empty mass
                max_payload=5.0,            # kg, admissible payload
                max_tow=11.2,               # kg, maximum takeoff weight
                arm=0.23+0.11+0.0373+0.03,  # kg, arm (comprehensive of motors). 230g motor, 110g ESC, 37.3g propeller. 
                total=None,                 # kg, total mass (to be calculated using payload, num rotors and weight of all arms)
                body=None,                  # kg, mass of central body
                payload=payload,            # kg, payload mass
                Ixx=None,                   # kg*m^2, inertia moments along local axis
                Iyy=None,                   # kg*m^2, inertia moments along local axis
                Izz=None)                   # kg*m^2, inertia moments along local axis
    
    geom = dict(num_rotors=8,  #, number of rotors
                body_type='thickdisk',  # -, should be either sphere, flatdisk, thickdisk.
                body_radius=((0.301+0.209)/2.0)/2.0,  # m, body radius. Internal plate is kind of oval, took average of main axes
                body_height=0.025*2.0,  # m, height of the body
                arm_length=1.270/2.0 - 0.1275)  # m, arm length

    dynamics = dict(num_states=12,          # -, number of states in state vector: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
                    num_inputs=4,           # -, number of inputs: thrust, moments along three axes
                    num_outputs=3,          # -, number of output measures (position coordinates)
                    C=None,               # observation matrix (constant)
                    thrust2weight=(45.0/gravity * geom['num_rotors']) / (mass['body_empty'] + mass['arm']*geom['num_rotors']),  # -, thrust over weight ratio. Tarot 18: 50 N per motor divided by weight
                    max_speed=15.0,         # m/s, max speed, see: https://www.nasa.gov/offices/amd/nasa_aircraft/small_unmanned_aircraft_systems/larc
                    max_wind_speed=8.0,     # m/s, max wind speed for safe flight (from DJIS1000 forum)
                    max_acceleration=None,  # m/s^2, max acceleration to be calculated using thrust2weight ratio
                    max_thrust=None,        # N, max thrust the power system can deliver
                    min_thrust=0.0,         # N, min thrust associated with rotors at minimum speed (25% throttle for KDE motors). Default is 0 but assigned after the propulsion system is assigned to AircraftModel.
                    state_vars=['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz'],
                    max_roll=45/180.0*np.pi,      # rad, maximum roll during flight
                    max_pitch=45/180.0*np.pi,     # rad, maximum pitch during flight
                    aero=dict(cd=1.0,   # drag coefficient of airframe (including rotors), not reliable
                              ad=0.25,  # apparent face of the rotorcraft facing air for drag force
                              ),
                    kt=5.25e-5,    # Approximation derived from: max omega of motor+rotor couple and max thrust for each rotor
                    kq=5e-5,       # Rotor torque constant
                    Gamma=None,     # Thrust allocation matrix (constant, depending on kt, kq, and the UAV rotor configuration)
                    Gamma_inv=None,     # Inverse of thrust allocation matrix (constant, depending on kt, kq, and the UAV rotor configuration)
                    )

    mass = rotorcraft_masses(mass, geom)   # compute total and body mass
    mass, geom = rotorcraft_inertia(mass, geom)   # Build rotorcraft inertia properties
    if payload > mass['max_payload']:
       warn("Payload for TAROT 18 exceeds its maximum recommended payload.")

    dynamics = rotorcraft_performance(dynamics, mass, gravity)
    dynamics['C'] = observation_matrix(dynamics['num_states'], dynamics['num_outputs'])

    # Set control allocation matrix and its inverse
    dynamics['Gamma'], dynamics['Gamma_inv'], _ = caf.rotorcraft_cam(
        n=geom['num_rotors'],
        length=geom['arm_length'],
        b=dynamics['kt'],
        d=dynamics['kq']
    )
    
    return mass, geom, dynamics


def DJIS1000(payload=0.0, gravity=9.81):
    """
    DJI S 1000 / Octocopter
    :param payload:         float, payload for the flight.
    :param gravity:         float, gravity value according to location
    :return:                dictionaries containing mass, geometry and dynamic properties of the vehicle
    """

    mass = dict(body_empty=1.8,         # kg, total empty mass
                max_payload=6.6,        # kg, admissible payload    
                max_tow=11.0,           # kg, max takeoff weight
                arm=0.325,              # kg, arm mass (comprehensive of motors)
                total=None,             # kg, total mass (to be calculated using payload, num rotors and weight of all arms)
                body=None,              # kg, mass of central body
                payload=payload,        # kg, payload mass
                Ixx=None,               # kg*m^2, inertia moments along local axis
                Iyy=None,               # kg*m^2, inertia moments along local axis
                Izz=None)               # kg*m^2, inertia moments along local axis

    geom = dict(num_rotors=8,           # #, number of rotors
                body_type='thickdisk',  # -, should be either sphere, flatdisk, thickdisk.
                body_radius=0.3775/2.0, # m, body radius
                body_height=0.1,        # m, height of the body
                arm_length=0.386)       # m, arm length
    
    dynamics = dict(num_states=12,      # -, number of states in state vector: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
                    num_inputs=4,       # -, number of inputs: thrust, moments along three axes
                    num_outputs=3,      # -, number of output measures (position coordinates)
                    C=None,             # observation matrix (constant)
                    thrust2weight=5.0,  # -, thrust over weight ratio
                    max_speed=16.0,     # m/s, max speed, from DJI website, not recommended
                    max_wind_speed=8.0, # m/s, max wind speed for safe flight
                    max_acceleration=None,  # m/s^2, max acceleration to be calculated using thrust2weight ratio
                    max_thrust=None,    # N, max thrust the power system can deliver
                    min_thrust=0.0,     # N, min thrust delivered at rotor minimum speed (25% of throttle with KDE motor)
                    state_vars=['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz'],
                    max_roll=45/180.0*np.pi,      # rad, maximum roll during flight
                    max_pitch=45/180.0*np.pi,     # rad, maximum pitch during flight
                    aero=dict(cd=1.0,  # drag coefficient of airframe (including rotors), not reliable
                              ad=0.25, # apparent face of the rotorcraft facing air for drag force
                              ),
                    kt=5.41e-5,         # Approximation derived from: max omega of motor+rotor couple and max thrust for each rotor
                    kq=5e-5,            # Rotor torque constant
                    Gamma=None,         # Thrust allocation matrix (constant, depending on kt, kq, and the UAV rotor configuration)
                    Gamma_inv=None,     # Inverse of thrust allocation matrix (constant, depending on kt, kq, and the UAV rotor configuration)
                    )
    
    mass = rotorcraft_masses(mass, geom)
    mass, geom = rotorcraft_inertia(mass, geom)
    if payload > mass['max_payload']:
        warn("Payload for DJIS1000 exceeds its maximum recommended payload.")

    dynamics = rotorcraft_performance(dynamics, mass, gravity)

    # Generate observation matrix
    dynamics['C'] = observation_matrix(dynamics['num_states'], dynamics['num_outputs'])
    
    # Set control allocation matrix and its inverse
    dynamics['Gamma'], dynamics['Gamma_inv'], _ = caf.rotorcraft_cam(n=geom['num_rotors'],
                                                                      length=geom['arm_length'],
                                                                      b=dynamics['kt'], 
                                                                      d=dynamics['kq'])
    
    return mass, geom, dynamics

# Vehicle-agnostic functions
def rotor_angles(n):
    # n: number of rotors
    arm_angle = 2.0 * np.pi / n
    nhalf = int(n/2)  # half of number of rotors
    return [arm_angle/2.0 * (2*ri + 1) for ri in range(nhalf)]


def sphere_inertia(mass, radius):
    Ix = 0.4 * mass * radius*radius
    Iz = Ix.copy()
    return Ix, Iz


def flatdisk_inertia(mass, radius):
    Ix = 0.25 * mass * radius*radius
    Iz = Ix * 2.0
    return Ix, Iz


def thickdisk_inertia(mass, radius, height):
    Ix = 0.25 * mass * radius*radius + 1.0/12.0 * mass * height*height
    Iz = 0.25 * mass * radius*radius
    return Ix, Iz


def rotorcraft_inertia(m, g):
    n_rotors = g['num_rotors']
    m['body'] = m['body_empty'] + n_rotors * m['arm']
    m['total'] = m['body'] + m['payload']

    # Define rotor positions on the 360 degree circle
    # ------------------------------------------------
    angular_vector = rotor_angles(n_rotors)

    motor_distance_from_xaxis = g['arm_length'] * np.sin(angular_vector)
    if g['body_type'].lower() == 'sphere':
        Ix0, Iz0 = sphere_inertia(m['body'], g['body_radius'])
    elif g['body_type'].lower() == 'flatdisk':
        Ix0, Iz0 = flatdisk_inertia(m['body'], g['body_radius'])
    elif g['body_type'].lower() == 'thickdisk':
        Ix0, Iz0 = thickdisk_inertia(m['body'], g['body_radius'], g['body_height'])
    else:
        raise Exception("Body geometry not implemented. Please choose among: sphere, flatdisk, thickdisk.")
    
    m['Ixx'] = Ix0 + 2.0 * sum(m['arm'] * motor_distance_from_xaxis**2.0)  # [kg m^2], inertia along x
    m['Iyy'] = m['Ixx']                                                    # [kg m^2], inertia along y (symmetric uav)
    m['Izz'] = Iz0 + g['num_rotors'] * (g['arm_length']**2.0 * m['arm'])   # [kg m^2], inertia along z
    return m, g


def rotorcraft_masses(mass_dict, geom_dict):
    """
    Compute mass of a small rotorcraft given the mass and geometry dictionary defined by the vehicle
    :param mass_dict:           dictionary of mass properties of the vehicle
    :param geom_dict:           dictionary of geometry properties of the vehicle
    :return:                    updated dictionary of mass properties of the vehicle
    """
    mass_dict['body'] = mass_dict['body_empty'] + geom_dict['num_rotors'] * mass_dict['arm']
    mass_dict['total'] = mass_dict['body'] + mass_dict['payload'] 
    return mass_dict


def rotorcraft_performance(dyn_dict, mass_dict, g):
    """
    Update dynamic performance dictionary of vehicle with maximum thrust and acceleration.

    :param dyn_dict:            dictionary of dynamic properties of the vehicle
    :param mass_dict:           dictionary of mass properties of the vehicle
    :param g:                   m/s^2, double, scalar, gravity magnitude
    :return:                    updated dictionary of dynamic properties of the vehicle
    """
    dyn_dict['max_thrust']       = dyn_dict['thrust2weight'] * (mass_dict['body'] * g)
    dyn_dict['max_acceleration'] = dyn_dict['max_thrust'] / mass_dict['total']
    return dyn_dict


def observation_matrix(num_states, num_outputs):
    """
    Build observation matrix C of a dynamic system following:

    dx/dt = f_{\theta}(x, u)
    y = C x

    where x is the state vector, u is the input vector, \theta is the vector of model parameters, and y is the output vector.
    """
    c = np.zeros((num_outputs, num_states))
    for ii in range(num_outputs):
        c[ii, ii] = 1.0
    return c

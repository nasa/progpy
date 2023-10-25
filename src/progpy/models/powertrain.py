# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from progpy import PrognosticsModel


def update_Cq(params):
    return {
        'C_q': params['c_q'] * params['rho'] * pow(params['D'], 5)
    }


class Powertrain(PrognosticsModel):
    """
    .. versionadded:: 1.3.0

    Powertrain :term:`model` composed of ESC and DC motor (with the option to add propeller load using parameter Cq).
    The powertrain model is used to simulate the motor dynamics including the effect of the speed controller and pulse-width modulation.

    Parameters for a standard propeller for commercial UAV are also added to the powertrain model, so that the load torque 
    acting on the motor can be computed. At this stage, the propeller is modeled simply as a load torque proportional to the square of the rotor speed. 
    When simulating the full PWM signal, this model needs a very small time step size (e.g., dt=1e-5) to show the full dynamics.
    Faster simulations can be achieved by ignoring the PWM square wave and acting directing on the input voltage. For example, modulating
    the input voltage to replicate the behavior of a throttle.

    References:
    Matteo Corbetta, Chetan S. Kulkarni. An approach for uncertainty quantification and management of unmanned aerial vehicle health. 
    Annual Conference of the PHM Society, Scottsdale, AZ, 2019. http://papers.phmsociety.org/index.php/phmconf/article/view/847

    George E. Gorospe Jr, Chetan S. Kulkarni, Edward Hogge, Andrew Hsu, and Natalie Ownby. A Study of the Degradation of Electronic Speed Controllers forBrushless DC Motors.
    Asia Pacific Conference of the Prognostics and Health Management Society, 2017. https://ntrs.nasa.gov/citations/20200000579

    R.P. Palanisamy C. Kulkarni, M. Corbetta, P. Banerjee “Fault Detection and Performance Monitoring of Propellers in Electric UAV", 2022 IEEE Aerospace

    This model was developed by NASA's System Wide Safety (SWS) Project. https://www.nasa.gov/aeroresearch/programs/aosp/sws/

    :term:`Events<event>`: (0)
        | None

    :term:`Inputs/Loading<input>`: (2)
        | duty :        Duty cycle [-], percentage the input is "on" (i.e., voltage is supplied). 0 = no voltage supply (always closed), 1 = 100% voltage supply (always open).
        | v :           voltage [V], voltage input from Battery (after DC converter, should be constant).

    :term:`States<state>`: (5)
        | v_a :         3-phase voltage value, first phase, [V], input to the motor
        | v_b :         3-phase voltage value, second phase, [V], input to the motor
        | v_c :         3-phase voltage value, third phase [V], input to the motor
        | t :           time value [s].
        | i_a :         3-phase current value, first phase [A], motor state
        | i_b :         3-phase current value, second phase [A], motor state
        | i_c :         3-phase current value, third phase [A], motor state
        | v_rot :       Motor angular velocity [rad/s]
        | theta :       Motor rotor position [rad]

    :term:`Outputs<output>`: (2)
        | v_rot :       Motor angular velocity [rad/s]
        | theta :       Motor rotor position [rad]

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
          distribution for :term:`measurement noise` (e.g., normal, uniform, 
        c_q : float
            Dimensionless coefficient of torque of the propeller [-], (APC data, derived).
        rho : float
            Air density [Kg/m^3].
        D: float
            Propeller diameter [m].

    Note:
        This model is known to be sensitive to noise. The process noise and measurement noise should be set to low values.
        
    Note: 
        Powertrain is added on top of any noise in the underlying esc and motor models. To update the esc or motor parameters, access m.esc.parameters and m.motor.paramters, respectively.
    """
    inputs = ['duty', 'v']
    states = ['v_a', 'v_b', 'v_c', 't', 'i_a', 'i_b', 'i_c', 'v_rot', 'theta']
    outputs = ['v_rot', 'theta']
    param_callbacks = {
        'c_q': [update_Cq],
        'rho': [update_Cq],
        'D': [update_Cq],
    }

    default_parameters = {
        # Load parameters
        'c_q': 5.42e-7,  # coefficient of torque (APC data, derived) [dimensionless]
        'rho': 1.225,  # (Kg/m^3)
        'D': 0.381,  # (m)
    }

    def __init__(self, esc, motor, **kwargs):
        super().__init__(**kwargs)
        self.esc = esc
        self.motor = motor

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            self.esc == other.esc and
            self.motor == other.motor
        )

    def initialize(self, u=None, z=None):
        x0 = self.esc.initialize(u=u, z=z)
        x0.update(self.motor.initialize(u=u, z=z))

        return self.StateContainer(x0)

    def next_state(self, x, u, dt: float):
        x_esc = self.esc.StateContainer(x)
        u_esc = {
            'duty': u['duty'],
            'theta': x['theta'],
            'v': u['v']}
        u_esc = self.esc.InputContainer(u_esc)
        x_esc = self.esc.next_state(x_esc, u_esc, dt)

        x_motor = self.motor.StateContainer(x)
        u_motor = {
            'v_a': x_esc['v_a'],
            'v_b': x_esc['v_b'],
            'v_c': x_esc['v_c'],
            't_l': self.parameters['C_q']*x['v_rot']**2
        }
        u_motor = self.motor.InputContainer(u_motor)
        x_motor = self.motor.next_state(x_motor, u_motor, dt)

        x_esc.update(x_motor)

        return self.StateContainer(x_esc)

    def output(self, x):
        return self.OutputContainer(
            {
                'v_rot': x['v_rot'], 
                'theta': x['theta']})

# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from math import floor
import numpy as np
import scipy.signal as signal

from progpy import PrognosticsModel

RAD_TO_DEG = 180/np.pi
PI2 = 2 * np.pi
CL = [[1, -1, 0],
      [1, 0, -1],
      [0, 1, -1],
      [-1, 1, 0],
      [-1, 0, 1],
      [0, -1, 1]]


def commutation(theta):
    """selection of switching pattern based on rotor position (theta)"""
    theta *= RAD_TO_DEG  # convert rad to deg
    return CL[floor(theta/60) % 6]


class ESC(PrognosticsModel):
    """
    .. versionadded:: 1.3.0

    Simple Electronic-Speed Controller (ESC) :term:`model` for powertrain modeling.
    This model replicates the behavior of the speed controller with pulse-width modulation (PWM) and commutation matrix.
    Duty cycle simulated with a square wave using scipy signal.square function.

    References: [0]_, [1]_.

    This model was developed by NASA's System Wide Safety (SWS) Project. https://www.nasa.gov/aeroresearch/programs/aosp/sws/

    :term:`Events<event>`: (0)
        | None
    
    :term:`Inputs/Loading<input>`: (3)
        | duty :        Duty cycle (unitless), percentage the input is "on" (i.e., voltage is supplied). 0 = no voltage supply (always closed), 1 = 100% voltage supply (always open).
        | theta :       rotor position (rad).
        | v :           voltage (V), voltage input from Battery (after DC converter, should be constant).

    :term:`States<state>`: (4)
        | v_a :         3-phase voltage value, first phase, (V), motor input
        | v_b :         3-phase voltage value, second phase, (V), motor input
        | v_c :         3-phase voltage value, third phase (V), motor input
        | t :           time value (s).

    :term:`Outputs<output>`: (4)
        | v_a :         3-phase voltage value, first phase, (V), motor input
        | v_b :         3-phase voltage value, second phase, (V), motor input
        | v_c :         3-phase voltage value, third phase (V), motor input
        | t :           time value (s).

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
        sawtooth_freq : float
            Frequency of PWM signal [Hz], default value in default_parameters.
        x0 : dict[str, float]
            Initial :term:`state`

    References
    ----------
    .. [0] Matteo Corbetta, Chetan S. Kulkarni. An approach for uncertainty quantification and management of unmanned aerial vehicle health. 
    Annual Conference of the PHM Society, Scottsdale, AZ, 2019. http://papers.phmsociety.org/index.php/phmconf/article/view/847
    .. [1] George E. Gorospe Jr, Chetan S. Kulkarni, Edward Hogge, Andrew Hsu, and Natalie Ownby. A Study of the Degradation of Electronic Speed Controllers forBrushless DC Motors.
    Asia Pacific Conference of the Prognostics and Health Management Society, 2017. https://ntrs.nasa.gov/citations/20200000579
    """
    default_parameters = {
        'sawtooth_freq': 16000,  # Hz

        # Motor Parameters
        'x0': {
            'v_a': 0,
            'v_b': 0,
            'v_c': 0,
            't': 0
        }
    }

    states = ['v_a', 'v_b', 'v_c', 't']
    inputs = ['duty', 'theta', 'v']
    outputs = states

    def next_state(self, x, u, dt: float):
        pw = np.maximum(signal.square(PI2 * self.parameters['sawtooth_freq'] * x['t'], duty=u['duty']), 0)
        V = pw*u['v']
        SP = commutation(u['theta'])
        VP = [V * sp_i for sp_i in SP]
        return self.StateContainer(np.array([
            np.atleast_1d(VP[0]),
            np.atleast_1d(VP[1]),
            np.atleast_1d(VP[2]),
            np.atleast_1d(x['t'] + dt)]))

    def output(self, x):
        return self.OutputContainer(x)

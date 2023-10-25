# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from math import inf
import numpy as np

from progpy import PrognosticsModel


class BatteryCircuit(PrognosticsModel):
    """
    Vectorized prognostics :term:`model` for a battery, represented by an equivilant circuit model as described in [DaigleSankararaman2013]_

    :term:`Events<event>`: (1)
        EOD: End of Discharge

    :term:`Inputs/Loading<input>`: (1)
        i: Current draw on the battery

    :term:`States<state>`: (4)
        | tb : Battery Temperature (K)
        | qb : Charge stored in Capacitor Cb of the equivalent circuit model
        | qcp : Charge stored in Capacitor Ccp of the equivalent circuit model
        | qcs : Charge stored in Capacitor Ccs of the equivalent circuit model

    :term:`Outputs<output>`: (2)
        | t: Temperature of battery (K)
        | v: Voltage supplied by battery

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
        V0 : float
          Nominal Battery Voltage
        Rp : float
          Battery Parasitic Resistance
        qMax : float
          Maximum Charge
        CMax : float
          Maximum Capacity
        VEOD : float
          End of Discharge Voltage Threshold
        Cb0 : float 
          Battery Capacity Parameter
        Cbp0 : float
          Battery Capacity Parameter
        Cbp1 : float
          Battery Capacity Parameter
        Cbp2 : float
          Battery Capacity Parameter
        Cbp3 : float
          Battery Capacity Parameter
        Rs : float
          R-C Pair Parameter
        Cs : float
          R-C Pair Parameter
        Rcp0 : float
          R-C Pair Parameter
        Rcp1 : float
          R-C Pair Parameter
        Rcp2 : float
          R-C Pair Parameter
        Ccp : float
          R-C Pair Parameter
        Ta : float
          Ambient Temperature (K)
        Jt : float
          Temperature parameter
        ha : float
          Heat transfer coefficient, ambient
        hcp : float
          Heat transfer coefficient parameter
        hcs : float
          Heat transfer coefficient - surface
        x0 : StateContianer
          Initial :term:`state`
    
    Note
    ----
        This is quicker but also less accurate than the electrochemistry :term:`model` (:py:class:`progpy.models.BatteryElectroChemEOD`). We recommend using the electrochemistry model, when possible.

    References
    -----------
    .. [DaigleSankararaman2013] M. Daigle and S. Sankararaman, "Advanced Methods for Determining Prediction Uncertainty in Model-Based Prognostics with Application to Planetary Rovers," Annual Conference of the Prognostics and Health Management Society 2013, pp. 262-274, New Orleans, LA, October 2013. https://papers.phmsociety.org/index.php/phmconf/article/view/2253
    """
    events = ['EOD']
    inputs = ['i']
    states = ['tb', 'qb', 'qcp', 'qcs']
    outputs = ['t', 'v']
    is_vectorized = True

    default_parameters = {  # Set to defaults
        'V0': 4.183,
        'Rp': 1e4,
        'qMax': 7856.3254,
        'CMax': 7777,
        'VEOD': 3.0,
        # Voltage above EOD after which voltage will be considered in SOC calculation
        'VDropoff': 0.1,
        # Capacitance 
        'Cb0': 1878.155726,
        'Cbp0': -230,
        'Cbp1': 1.2,
        'Cbp2': 2079.9,
        'Cbp3': 27.055726,
        # R-C Pairs
        'Rs': 0.0538926,
        'Cs': 234.387,
        'Rcp0': 0.0697776,
        'Rcp1': 1.50528e-17,
        'Rcp2': 37.223,
        'Ccp': 14.8223,
        # Temperature Parameters
        'Ta': 292.1,
        'Jt': 800,
        'ha': 0.5,
        'hcp': 19,
        'hcs': 1,
        'x0': {
            'tb': 292.1,
            'qb': 7856.3254,
            'qcp': 0,
            'qcs': 0
        }
    }

    state_limits = {
        'tb': (0, inf),  # Limited by absolute zero. Note thermal runaway temperature is ~130°C, so the model is not valid after that temperature.
        'qb': (0, inf)
    }

    def dx(self, x, u):
        # Keep this here- accessing member can be expensive in python
        # this optimization reduces runtime by almost half!
        parameters = self.parameters
        Rs = parameters['Rs']
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] +
               x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1'] * \
            SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Rcp = parameters['Rcp0'] + parameters['Rcp1'] * \
            np.exp(parameters['Rcp2']*(-SOC + 1))
        Vb = x['qb']/Cb
        Tbdot = (Rcp*Rs*parameters['ha']*(parameters['Ta'] - x['tb']) + Rcp*Vcs**2*parameters['hcs'] + Rs*Vcp**2*parameters['hcp']) \
            / (parameters['Jt']*Rcp*Rs)
        Vp = Vb - Vcp - Vcs
        ip = Vp/parameters['Rp']
        ib = u['i'] + ip
        icp = ib - Vcp/Rcp
        ics = ib - Vcs/Rs

        return self.StateContainer(np.array([
            np.atleast_1d(Tbdot),  # tb
            np.atleast_1d(-ib),  # qb
            np.atleast_1d(icp),  # qcp
            np.atleast_1d(ics)  # qcs
        ]))
    
    def event_state(self, x) -> dict:
        parameters = self.parameters
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1']*SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Vb = x['qb']/Cb
        v = Vb - Vcp - Vcs
        charge_EOD = (parameters['CMax'] -
                      parameters['qMax'] + x['qb'])/parameters['CMax']
        voltage_EOD = (v - self.parameters['VEOD']) / \
            self.parameters['VDropoff']
        return {
            'EOD': np.minimum(charge_EOD, voltage_EOD)
        }

    def output(self, x):
        parameters = self.parameters
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1']*SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Vb = x['qb']/Cb

        return self.OutputContainer(np.array([
            np.atleast_1d(x['tb']),            # t
            np.atleast_1d(Vb - Vcp - Vcs)]))   # v

    def threshold_met(self, x) -> dict:
        parameters = self.parameters
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1']*SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Vb = x['qb']/Cb
        V = Vb - Vcp - Vcs

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': V < parameters['VEOD']
        }

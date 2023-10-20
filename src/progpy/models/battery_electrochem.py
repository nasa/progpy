# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy
import numpy as np
from scipy.optimize import fsolve
import warnings

from progpy import PrognosticsModel

# Constants of nature
R = 8.3144621  # universal gas constant, J/K/mol
F = 96487      # Faraday's constant, C/mol
R_F = R / F    # Optimization - R / F
mC = 37.04     # kg/m2/(K-s^2)
tau = 100


def update_qmax(params: dict) -> dict:
    # note qMax = qn+qp
    return {
        'qMax': params['qMobile']/(params['xnMax']-params['xnMin'])
    }

def update_vols(params: dict) -> dict:
    # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
    # same and the surface/bulk split is the same for both electrodes
    return {
        'VolS': params['VolSFraction']*params['Vol'],
        'VolB': params['Vol']*(1.0-params['VolSFraction'])
    }

# set up charges (Li ions)
def update_qnmin(params: dict) -> dict:
    # min charge at negative electrode
    return {
        'qnMin': params['qMax']*params['xnMin']
    }

def update_qnmax(params: dict) -> dict:
    # max charge at negative electrode
    return {
        'qnMax': params['qMax']*params['xnMax']
    }

def update_qpSBmin(params: dict) -> dict:
    # min charge at surface and bulk pos electrode
    return {
        'x0': {
            **params['x0'],
            'qpS': params['qMax']*params['xpMin']*params['VolSFraction'],
            'qpB': params['qMax']*params['xpMin']*(1.0-params['VolSFraction'])
        }
    }

def update_xpMin(params: dict) -> dict:
    return {
        'xpMin': 1.0-params['xnMax']
    }

def update_xnMin(params: dict) -> dict:
    return {
        'xnMin': 1.0-params['xpMax']
    }

def update_qnSBmax(params: dict) -> dict:
    # max charge at surface and pos electrode
    return {
        'x0': {
            **params['x0'],
            'qnS': params['qMax']*params['xnMax']*params['VolSFraction'],
            'qnB': params['qMax']*params['xnMax']*(1.0-params['VolSFraction'])
        }
    }

def update_v0(params: dict) -> dict:
    # update the initial voltage

    if 'qnS' not in params['x0']:
        # qnS not yet set
        return {}

    An = params['An']
    # Negative Surface
    xnS = params['x0']['qnS']/params['qSMax']
    xnS2 = xnS+xnS  # Note: in python x+x is more efficient than 2*x
    one_minus_xnS = 1 - xnS
    xnS2_minus_1 = xnS2 - 1
    VenParts = [
        An[0] *xnS2_minus_1/F,  # Ven0
        An[1] *(xnS2_minus_1**2  - (xnS2*one_minus_xnS))/F,  # Ven1
        An[2] *(xnS2_minus_1**3  - (4 *xnS*one_minus_xnS)*xnS2_minus_1)/F,  #Ven2
        An[3] *(xnS2_minus_1**4  - (6 *xnS*one_minus_xnS)*xnS2_minus_1**2) /F,  #Ven3
        An[4] *(xnS2_minus_1**5  - (8 *xnS*one_minus_xnS)*xnS2_minus_1**3) /F,  #Ven4
        An[5] *(xnS2_minus_1**6  - (10*xnS*one_minus_xnS)*xnS2_minus_1**4) /F,  #Ven5
        An[6] *(xnS2_minus_1**7  - (12*xnS*one_minus_xnS)*xnS2_minus_1**5) /F,  #Ven6
        An[7] *(xnS2_minus_1**8  - (14*xnS*one_minus_xnS)*xnS2_minus_1**6) /F,  #Ven7
        An[8] *(xnS2_minus_1**9  - (16*xnS*one_minus_xnS)*xnS2_minus_1**7) /F,  #Ven8
        An[9] *(xnS2_minus_1**10 - (18*xnS*one_minus_xnS)*xnS2_minus_1**8) /F,  #Ven9
        An[10]*(xnS2_minus_1**11 - (20*xnS*one_minus_xnS)*xnS2_minus_1**9) /F,  #Ven10
        An[11]*(xnS2_minus_1**12 - (22*xnS*one_minus_xnS)*xnS2_minus_1**10)/F,  #Ven11
        An[12]*(xnS2_minus_1**13 - (24*xnS*one_minus_xnS)*xnS2_minus_1**11)/F   #Ven12
    ]
    Ven = params['U0n'] + R*params['x0']['tb']/F*np.log(one_minus_xnS/xnS) + sum(VenParts)

    # Positive Surface
    Ap = params['Ap']
    xpS = params['x0']['qpS']/params['qSMax']
    one_minus_xpS = (1 - xpS)
    xpS2 = xpS + xpS
    xpS2_minus_1 = xpS2 - 1
    VepParts = [
        Ap[0] *(xpS2_minus_1)/F,  #Vep0
        Ap[1] *((xpS2_minus_1)**2  - (xpS2*one_minus_xpS))/F,  #Vep1 
        Ap[2] *((xpS2_minus_1)**3  - (4 *xpS*one_minus_xpS)*(xpS2_minus_1)) /F,  #Vep2
        Ap[3] *((xpS2_minus_1)**4  - (6 *xpS*one_minus_xpS)*(xpS2_minus_1)**(2)) /F,  #Vep3
        Ap[4] *((xpS2_minus_1)**5  - (8 *xpS*one_minus_xpS)*(xpS2_minus_1)**(3)) /F,  #Vep4
        Ap[5] *((xpS2_minus_1)**6  - (10*xpS*one_minus_xpS)*(xpS2_minus_1)**(4)) /F,  #Vep5
        Ap[6] *((xpS2_minus_1)**7  - (12*xpS*one_minus_xpS)*(xpS2_minus_1)**(5)) /F,  #Vep6
        Ap[7] *((xpS2_minus_1)**8  - (14*xpS*one_minus_xpS)*(xpS2_minus_1)**(6)) /F,  #Vep7
        Ap[8] *((xpS2_minus_1)**9  - (16*xpS*one_minus_xpS)*(xpS2_minus_1)**(7)) /F,  #Vep8
        Ap[9] *((xpS2_minus_1)**10 - (18*xpS*one_minus_xpS)*(xpS2_minus_1)**(8)) /F,  #Vep9
        Ap[10]*((xpS2_minus_1)**11 - (20*xpS*one_minus_xpS)*(xpS2_minus_1)**(9)) /F,  #Vep10
        Ap[11]*((xpS2_minus_1)**12 - (22*xpS*one_minus_xpS)*(xpS2_minus_1)**(10))/F,  #Vep11
        Ap[12]*((xpS2_minus_1)**13 - (24*xpS*one_minus_xpS)*(xpS2_minus_1)**(11))/F   #Vep12
    ]
    Vep = params['U0p'] + R*params['x0']['tb']/F*np.log(one_minus_xpS/xpS) + sum(VepParts)

    return {
        'v0': Vep - Ven - params['x0']['Vo'] - params['x0']['Vsn'] - params['x0']['Vsp']
    }

def update_qSBmax(params: dict) -> dict:
    # max charge at surface, bulk (pos and neg)
    return {
        'qSMax': params['qMax']*params['VolSFraction'],
        'qBMax': params['qMax']*(1.0-params['VolSFraction']),
    }


class BatteryElectroChemEOD(PrognosticsModel):
    """
    Vectorized prognostics :term:`model` for a battery, represented by an electrochemical equations as described in [Daigle2013]_. This model predicts the end of discharge event. 

    The default model parameters included are for Li-ion batteries, specifically 18650-type cells. Experimental discharge curves for these cells can be downloaded from the Prognostics Center of Excellence Data Repository [DataRepo]_.

    :term:`Events<event>`: (1)
        EOD: End of Discharge

    :term:`Inputs/Loading<input>`: (1)
        i: Current draw on the battery

    :term:`States<state>`: (8)
        | tb: Battery temperature (K)
        | Vo: Voltage Drops due to Solid-Phase Ohmic Resistances
        | Vsn: Negative Surface Voltage (V)
        | Vsp: Positive Surface Voltage (V)
        | qnB: Amount of Negative Ions at the Battery Bulk
        | qnS: Amount of Negative Ions at the Battery Surface
        | qpB: Amount of Positive Ions at the Battery Bulk
        | qpS: Amount of Positive Ions at the Battery Surface

    :term:`Outputs<output>`: (2)
        | t: Temperature of battery (°C) 
        | v: Voltage supplied by battery

    :term:`Performance Metrics<performance metric>`: (1)
        | max_i : The maximum current (amps) that can be sustained before steady-state voltage falls below VEOD. Decreases with discharge 

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
        qMobile : float
        xnMax : float
            Maximum mole fraction (neg electrode)
        xpMax : float
            Maximum mole fraction (pos electrode). Typically 1.
        Ro : float
            for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        alpha : float
            anodic/cathodic electrochemical transfer coefficient
        Sn : float
            Surface area (- electrode) 
        Sp : float
            Surface area (+ electrode)
        kn : float
            lumped constant for BV (- electrode)
        kp : float
            lumped constant for BV (+ electrode)
        Vol : float
            total interior battery volume/2 (for computing concentrations)
        VolSFraction : float
            fraction of total volume occupied by surface volume
        tDiffusion : float
            diffusion time constant (increasing this causes decrease in diffusion rate)
        to : float
            for Ohmic voltage
        tsn : float
            for surface overpotential (neg)
        tsp : float
            for surface overpotential (pos)
        U0p : float
            Redlich-Kister parameter (+ electrode)
        Ap : float
            Redlich-Kister parameter (+ electrode)
        U0n : float
            Redlich-Kister parameter (- electrode)
        An : float
            Redlich-Kister parameter (- electrode)
        VEOD : float
            End of Discharge Voltage Threshold
        x0 : dict[str, float]
            Initial :term:`state`

    See Also
    --------
    BatteryElectroChemEOL, BatteryElectroChem, BatteryElectroChemEODEOL

    References 
    -------------
     .. [Daigle2013] M. Daigle and C. Kulkarni, "Electrochemistry-based Battery Modeling for Prognostics," Annual Conference of the Prognostics and Health Management Society 2013, pp. 249-261, New Orleans, LA, October 2013. https://papers.phmsociety.org/index.php/phmconf/article/view/2252
     .. [DataRepo] Prognostics Center of Excellence Data Repository https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/.
    """
    events = ['EOD']
    inputs = ['i']
    states = ['tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
    outputs = ['t', 'v']
    performance_metric_keys = ['max_i']
    is_vectorized = True

    default_parameters = {  # Set to defaults
        'qMobile': 7600,
        'xnMax': 0.6,
        'xnMin': 0.0,
        'xpMax': 1.0,
        'xpMin': 0.4,
        'Ro': 0.117215,
        
        # Li-ion parameters
        'alpha': 0.5,
        'Sn': 0.000437545,
        'Sp': 0.00030962,
        'kn': 2120.96,
        'kp': 248898,
        'Vol': 2e-5,
        'VolSFraction': 0.1,

        # time constants
        'tDiffusion': 7e6,
        'to': 6.08671,
        'tsn': 1001.38,
        'tsp': 46.4311,

        # Redlich-Kister parameters (+ electrode)
        'U0p': 4.03,
        'Ap': [-31593.7, 0.106747, 24606.4, -78561.9, 13317.9, 307387, 84916.1, -1.07469e+06, 2285.04, 990894, 283920, -161513, -469218],

        # Redlich-Kister parameters (- electrode)
        'U0n': 0.01,
        'An': [86.19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        'x0': {
            'Vo': 0,
            'Vsn': 0,
            'Vsp': 0,
            'tb': 292.1  # in K, about 18.95 C
        },

        # End of discharge voltage threshold
        'VEOD': 3.0, 
        'VDropoff': 0.1 # Voltage above EOD after which voltage will be considered in SOC calculation
    }

    state_limits = {
        'tb': (0, np.inf),  # Limited by Absolute Zero (0 K)
        'qnB': (0, np.inf),
        'qnS': (0, np.inf),
        'qpB': (0, np.inf),
        'qpS': (0, np.inf)
    }

    param_callbacks = {  # Callbacks for derived parameters
        'qMobile': [update_qmax],
        'VolSFraction': [update_vols, update_qpSBmin, update_qSBmax],
        'Vol': [update_vols],
        'qMax': [update_qpSBmin, update_qnmin, update_qnmax, update_qpSBmin, update_qSBmax],
        'xpMin': [update_qpSBmin],
        'xpMax': [update_xnMin],
        'xnMin': [update_qmax, update_qnmin],
        'xnMax': [update_xpMin, update_qmax, update_qnmax, update_qnSBmax]
    }

    def dx(self, x, u):
        params = self.parameters
        # Negative Surface
        CnBulk = x['qnB']/params['VolB']
        CnSurface = x['qnS']/params['VolS']
        xnS = x['qnS']/params['qSMax']

        qdotDiffusionBSn = (CnBulk-CnSurface)/params['tDiffusion']
        qnBdot = -qdotDiffusionBSn
        qnSdot = qdotDiffusionBSn - u["i"]

        Jn = u['i']/params['Sn']
        Jn0 = params['kn']*((1-xnS)*xnS)**params['alpha']

        v_part = R_F*x['tb']/params['alpha']

        VsnNominal = v_part*np.arcsinh(Jn/(Jn0 + Jn0))
        Vsndot = (VsnNominal-x['Vsn'])/params['tsn']

        # Positive Surface
        CpBulk = x['qpB']/params['VolB']
        CpSurface = x['qpS']/params['VolS']
        xpS = x['qpS']/params['qSMax']
        
        qdotDiffusionBSp = (CpBulk-CpSurface)/params['tDiffusion']
        qpBdot = -qdotDiffusionBSp
        qpSdot = u['i'] + qdotDiffusionBSp

        Jp = u['i']/params['Sp']
        Jp0 = params['kp']*((1-xpS)*xpS)**params['alpha']

        VspNominal = v_part*np.arcsinh(Jp/(Jp0+Jp0))
        Vspdot = (VspNominal-x['Vsp'])/params['tsp']

        # Combined
        VoNominal = u['i']*params['Ro']
        Vodot = (VoNominal-x['Vo'])/params['to']

        # Thermal Effects
        voltage_eta = x['Vo'] + x['Vsn'] + x['Vsp'] # (Vep - Ven) - V;

        Tbdot = voltage_eta*u['i']/mC + (params['x0']['tb'] - x['tb'])/tau # Newman

        return self.StateContainer(np.array([
            np.atleast_1d(Tbdot),
            np.atleast_1d(Vodot),
            np.atleast_1d(Vsndot),
            np.atleast_1d(Vspdot),
            np.atleast_1d(qnBdot),
            np.atleast_1d(qnSdot),
            np.atleast_1d(qpBdot),
            np.atleast_1d(qpSdot)
        ]))

    def performance_metrics(self, x):
        params = self.parameters
        An = params['An']
        # Negative Surface
        xnS = x['qnS']/params['qSMax']
        xnS2 = xnS+xnS  # Note: in python x+x is more efficient than 2*x
        one_minus_xnS = 1 - xnS
        xnS2_minus_1 = xnS2 - 1
        VenParts = [
            An[0] *xnS2_minus_1/F,  # Ven0
            An[1] *(xnS2_minus_1**2  - (xnS2*one_minus_xnS))/F,  # Ven1
            An[2] *(xnS2_minus_1**3  - (4 *xnS*one_minus_xnS)*xnS2_minus_1)/F,  #Ven2
            An[3] *(xnS2_minus_1**4  - (6 *xnS*one_minus_xnS)*xnS2_minus_1**2) /F,  #Ven3
            An[4] *(xnS2_minus_1**5  - (8 *xnS*one_minus_xnS)*xnS2_minus_1**3) /F,  #Ven4
            An[5] *(xnS2_minus_1**6  - (10*xnS*one_minus_xnS)*xnS2_minus_1**4) /F,  #Ven5
            An[6] *(xnS2_minus_1**7  - (12*xnS*one_minus_xnS)*xnS2_minus_1**5) /F,  #Ven6
            An[7] *(xnS2_minus_1**8  - (14*xnS*one_minus_xnS)*xnS2_minus_1**6) /F,  #Ven7
            An[8] *(xnS2_minus_1**9  - (16*xnS*one_minus_xnS)*xnS2_minus_1**7) /F,  #Ven8
            An[9] *(xnS2_minus_1**10 - (18*xnS*one_minus_xnS)*xnS2_minus_1**8) /F,  #Ven9
            An[10]*(xnS2_minus_1**11 - (20*xnS*one_minus_xnS)*xnS2_minus_1**9) /F,  #Ven10
            An[11]*(xnS2_minus_1**12 - (22*xnS*one_minus_xnS)*xnS2_minus_1**10)/F,  #Ven11
            An[12]*(xnS2_minus_1**13 - (24*xnS*one_minus_xnS)*xnS2_minus_1**11)/F   #Ven12
        ]
        Ven = params['U0n'] + R*x['tb']/F*np.log(one_minus_xnS/xnS) + sum(VenParts)

        # Positive Surface
        Ap = params['Ap']
        xpS = x['qpS']/params['qSMax']
        one_minus_xpS = 1 - xpS
        xpS2 = xpS + xpS
        xpS2_minus_1 = xpS2 - 1
        VepParts = [
            Ap[0] *(xpS2_minus_1)/F,  #Vep0
            Ap[1] *((xpS2_minus_1)**2  - xpS2*one_minus_xpS)/F,  #Vep1 
            Ap[2] *((xpS2_minus_1)**3  - 4 *xpS*one_minus_xpS*xpS2_minus_1)/F,  #Vep2
            Ap[3] *((xpS2_minus_1)**4  - 6 *xpS*one_minus_xpS*xpS2_minus_1**2) /F,  #Vep3
            Ap[4] *((xpS2_minus_1)**5  - 8 *xpS*one_minus_xpS*xpS2_minus_1**3) /F,  #Vep4
            Ap[5] *((xpS2_minus_1)**6  - 10*xpS*one_minus_xpS*xpS2_minus_1**4) /F,  #Vep5
            Ap[6] *((xpS2_minus_1)**7  - 12*xpS*one_minus_xpS*xpS2_minus_1**5) /F,  #Vep6
            Ap[7] *((xpS2_minus_1)**8  - 14*xpS*one_minus_xpS*xpS2_minus_1**6) /F,  #Vep7
            Ap[8] *((xpS2_minus_1)**9  - 16*xpS*one_minus_xpS*xpS2_minus_1**7) /F,  #Vep8
            Ap[9] *((xpS2_minus_1)**10 - 18*xpS*one_minus_xpS*xpS2_minus_1**8) /F,  #Vep9
            Ap[10]*((xpS2_minus_1)**11 - 20*xpS*one_minus_xpS*xpS2_minus_1**9) /F,  #Vep10
            Ap[11]*((xpS2_minus_1)**12 - 22*xpS*one_minus_xpS*xpS2_minus_1**10)/F,  #Vep11
            Ap[12]*((xpS2_minus_1)**13 - 24*xpS*one_minus_xpS*xpS2_minus_1**11)/F   #Vep12
        ]
        Vep = params['U0p'] + R*x['tb']/F*np.log(one_minus_xpS/xpS) + sum(VepParts)

        v_part = R_F*x['tb']/params['alpha']
        Jp0 = params['kp']*(one_minus_xpS*xpS)**params['alpha']
        Jn0 = params['kn']*(one_minus_xnS*xnS)**params['alpha']

        C1 = params['Sn']*(2*Jn0)
        C2 = params['Sp']*(2*Jp0)
        
        # Solve for the current that would cause the steady state voltage to hit VEOD
        def f(i):
            return Vep - Ven - i*params['Ro'] - v_part*(np.arcsinh(i/C1) + np.arcsinh(i/C2)) - params['VEOD']
        
        return {'max_i': fsolve(f, [3])}
        
    def event_state(self, x) -> dict:
        # The most "correct" indication of SOC is based on charge (charge_EOD), 
        # since voltage decreases non-linearally. 
        # However, as voltage approaches VEOD, the charge-based approach no 
        # longer accurately captures this behavior, so voltage_EOD takes over as 
        # the driving factor. 
        params = self.parameters
        An = params['An']
        # Negative Surface
        xnS = x['qnS']/params['qSMax']
        xnS2 = xnS+xnS  # Note: in python x+x is more efficient than 2*x
        one_minus_xnS = 1 - xnS
        xnS2_minus_1 = xnS2 - 1
        VenParts = [
            An[0] *xnS2_minus_1/F,  # Ven0
            An[1] *(xnS2_minus_1**2  - (xnS2*one_minus_xnS))/F,  # Ven1
            An[2] *(xnS2_minus_1**3  - (4 *xnS*one_minus_xnS)*xnS2_minus_1)/F,  #Ven2
            An[3] *(xnS2_minus_1**4  - (6 *xnS*one_minus_xnS)*xnS2_minus_1**2) /F,  #Ven3
            An[4] *(xnS2_minus_1**5  - (8 *xnS*one_minus_xnS)*xnS2_minus_1**3) /F,  #Ven4
            An[5] *(xnS2_minus_1**6  - (10*xnS*one_minus_xnS)*xnS2_minus_1**4) /F,  #Ven5
            An[6] *(xnS2_minus_1**7  - (12*xnS*one_minus_xnS)*xnS2_minus_1**5) /F,  #Ven6
            An[7] *(xnS2_minus_1**8  - (14*xnS*one_minus_xnS)*xnS2_minus_1**6) /F,  #Ven7
            An[8] *(xnS2_minus_1**9  - (16*xnS*one_minus_xnS)*xnS2_minus_1**7) /F,  #Ven8
            An[9] *(xnS2_minus_1**10 - (18*xnS*one_minus_xnS)*xnS2_minus_1**8) /F,  #Ven9
            An[10]*(xnS2_minus_1**11 - (20*xnS*one_minus_xnS)*xnS2_minus_1**9) /F,  #Ven10
            An[11]*(xnS2_minus_1**12 - (22*xnS*one_minus_xnS)*xnS2_minus_1**10)/F,  #Ven11
            An[12]*(xnS2_minus_1**13 - (24*xnS*one_minus_xnS)*xnS2_minus_1**11)/F   #Ven12
        ]
        Ven = params['U0n'] + R*x['tb']/F*np.log(one_minus_xnS/xnS) + sum(VenParts)

        # Positive Surface
        Ap = params['Ap']
        xpS = x['qpS']/params['qSMax']
        one_minus_xpS = 1 - xpS
        xpS2 = xpS + xpS
        xpS2_minus_1 = xpS2 - 1
        VepParts = [
            Ap[0] *(xpS2_minus_1)/F,  #Vep0
            Ap[1] *((xpS2_minus_1)**2  - (xpS2*one_minus_xpS))/F,  #Vep1 
            Ap[2] *((xpS2_minus_1)**3  - (4 *xpS*one_minus_xpS)*(xpS2_minus_1)) /F,  #Vep2
            Ap[3] *((xpS2_minus_1)**4  - (6 *xpS*one_minus_xpS)*(xpS2_minus_1)**(2)) /F,  #Vep3
            Ap[4] *((xpS2_minus_1)**5  - (8 *xpS*one_minus_xpS)*(xpS2_minus_1)**(3)) /F,  #Vep4
            Ap[5] *((xpS2_minus_1)**6  - (10*xpS*one_minus_xpS)*(xpS2_minus_1)**(4)) /F,  #Vep5
            Ap[6] *((xpS2_minus_1)**7  - (12*xpS*one_minus_xpS)*(xpS2_minus_1)**(5)) /F,  #Vep6
            Ap[7] *((xpS2_minus_1)**8  - (14*xpS*one_minus_xpS)*(xpS2_minus_1)**(6)) /F,  #Vep7
            Ap[8] *((xpS2_minus_1)**9  - (16*xpS*one_minus_xpS)*(xpS2_minus_1)**(7)) /F,  #Vep8
            Ap[9] *((xpS2_minus_1)**10 - (18*xpS*one_minus_xpS)*(xpS2_minus_1)**(8)) /F,  #Vep9
            Ap[10]*((xpS2_minus_1)**11 - (20*xpS*one_minus_xpS)*(xpS2_minus_1)**(9)) /F,  #Vep10
            Ap[11]*((xpS2_minus_1)**12 - (22*xpS*one_minus_xpS)*(xpS2_minus_1)**(10))/F,  #Vep11
            Ap[12]*((xpS2_minus_1)**13 - (24*xpS*one_minus_xpS)*(xpS2_minus_1)**(11))/F   #Vep12
        ]
        Vep = params['U0p'] + R*x['tb']/F*np.log(one_minus_xpS/xpS) + sum(VepParts)
        v = Vep - Ven - x['Vo'] - x['Vsn'] - x['Vsp']

        charge_EOD = (x['qnS'] + x['qnB'])/self.parameters['qnMax']
        voltage_EOD = (v - self.parameters['VEOD'])/self.parameters['VDropoff'] 
        return {
            'EOD': np.clip(min(charge_EOD, voltage_EOD), 0, 1)
        }

    def output(self, x):
        params = self.parameters
        An = params['An']
        # Negative Surface
        xnS = x['qnS']/params['qSMax']
        xnS2 = xnS+xnS  # Note: in python x+x is more efficient than 2*x

        one_minus_xnS = 1 - xnS
        xnS2_minus_1 = xnS2 - 1
        VenParts = [
            An[0] *xnS2_minus_1/F,  # Ven0
            An[1] *(xnS2_minus_1**2  - (xnS2*one_minus_xnS))/F,  # Ven1
            An[2] *(xnS2_minus_1**3  - (4 *xnS*one_minus_xnS)*xnS2_minus_1)/F,  #Ven2
            An[3] *(xnS2_minus_1**4  - (6 *xnS*one_minus_xnS)*xnS2_minus_1**2) /F,  #Ven3
            An[4] *(xnS2_minus_1**5  - (8 *xnS*one_minus_xnS)*xnS2_minus_1**3) /F,  #Ven4
            An[5] *(xnS2_minus_1**6  - (10*xnS*one_minus_xnS)*xnS2_minus_1**4) /F,  #Ven5
            An[6] *(xnS2_minus_1**7  - (12*xnS*one_minus_xnS)*xnS2_minus_1**5) /F,  #Ven6
            An[7] *(xnS2_minus_1**8  - (14*xnS*one_minus_xnS)*xnS2_minus_1**6) /F,  #Ven7
            An[8] *(xnS2_minus_1**9  - (16*xnS*one_minus_xnS)*xnS2_minus_1**7) /F,  #Ven8
            An[9] *(xnS2_minus_1**10 - (18*xnS*one_minus_xnS)*xnS2_minus_1**8) /F,  #Ven9
            An[10]*(xnS2_minus_1**11 - (20*xnS*one_minus_xnS)*xnS2_minus_1**9) /F,  #Ven10
            An[11]*(xnS2_minus_1**12 - (22*xnS*one_minus_xnS)*xnS2_minus_1**10)/F,  #Ven11
            An[12]*(xnS2_minus_1**13 - (24*xnS*one_minus_xnS)*xnS2_minus_1**11)/F   #Ven12
        ]
        Ven = params['U0n'] + R*x['tb']/F*np.log(one_minus_xnS/xnS) + sum(VenParts)

        # Positive Surface
        Ap = params['Ap']
        xpS = x['qpS']/params['qSMax']
        one_minus_xpS = 1 - xpS
        xpS2 = xpS + xpS
        xpS2_minus_1 = xpS2 - 1
        VepParts = [
            Ap[0] *(xpS2_minus_1)/F,  #Vep0
            Ap[1] *((xpS2_minus_1)**2  - (xpS2*one_minus_xpS))/F,  #Vep1 
            Ap[2] *((xpS2_minus_1)**3  - (4 *xpS*one_minus_xpS)*(xpS2_minus_1)) /F,  #Vep2
            Ap[3] *((xpS2_minus_1)**4  - (6 *xpS*one_minus_xpS)*(xpS2_minus_1)**(2)) /F,  #Vep3
            Ap[4] *((xpS2_minus_1)**5  - (8 *xpS*one_minus_xpS)*(xpS2_minus_1)**(3)) /F,  #Vep4
            Ap[5] *((xpS2_minus_1)**6  - (10*xpS*one_minus_xpS)*(xpS2_minus_1)**(4)) /F,  #Vep5
            Ap[6] *((xpS2_minus_1)**7  - (12*xpS*one_minus_xpS)*(xpS2_minus_1)**(5)) /F,  #Vep6
            Ap[7] *((xpS2_minus_1)**8  - (14*xpS*one_minus_xpS)*(xpS2_minus_1)**(6)) /F,  #Vep7
            Ap[8] *((xpS2_minus_1)**9  - (16*xpS*one_minus_xpS)*(xpS2_minus_1)**(7)) /F,  #Vep8
            Ap[9] *((xpS2_minus_1)**10 - (18*xpS*one_minus_xpS)*(xpS2_minus_1)**(8)) /F,  #Vep9
            Ap[10]*((xpS2_minus_1)**11 - (20*xpS*one_minus_xpS)*(xpS2_minus_1)**(9)) /F,  #Vep10
            Ap[11]*((xpS2_minus_1)**12 - (22*xpS*one_minus_xpS)*(xpS2_minus_1)**(10))/F,  #Vep11
            Ap[12]*((xpS2_minus_1)**13 - (24*xpS*one_minus_xpS)*(xpS2_minus_1)**(11))/F   #Vep12
        ]
        Vep = params['U0p'] + R*x['tb']/F*np.log(one_minus_xpS/xpS) + sum(VepParts)

        return self.OutputContainer(np.array([
            np.atleast_1d(x['tb'] - 273.15),
            np.atleast_1d(Vep - Ven - x['Vo'] - x['Vsn'] - x['Vsp'])
        ]))

    def threshold_met(self, x) -> dict:
        z = self.output(x)

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': z['v'] < self.parameters['VEOD']
        }


class BatteryElectroChemEOL(PrognosticsModel):
    """
    Vectorized prognostics :term:`model` for a battery degredation, represented by an electrochemical model as described in [Daigle2016]_

    The default model parameters included are for Li-ion batteries, specifically 18650-type cells. Experimental discharge curves for these cells can be downloaded from the Prognostics Center of Excellence Data Repository [DataRepo]_.

    :term:`Events<event>`: (1)
        InsufficientCapacity: Insufficient battery capacity

    :term:`Inputs/Loading<input>`: (1)
        i: Current draw on the battery

    :term:`States<state>`: (3)
        | qMax: Maximum battery capacity
        | Ro : for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        | D : diffusion time constant (increasing this causes decrease in diffusion rate)

    :term:`Outputs<output>`: (0)

    Keyword Args
    ------------
        process_noise : Optional, float or dict[Srt, float]
          Process noise (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for process noise (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[Srt, float]
          Measurement noise (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for measurement noise (e.g., normal, uniform, triangular)
        qMaxThreshold : float
            Threshold for qMax (for threshold_met and event_state), after which the InsufficientCapacity event has occurred. Note: Battery manufacturers specify a threshold of 70-80% of qMax
        wq : float
            Wear rate for qMax
        wr : float
            Wear rate for Ro
        wd : float
            Wear rate for D
        x0 : dict[str, float]
            Initial :term:`state`
    
    See Also
    --------
    BatteryElectroChemEOD, BatteryElectroChem, BatteryElectroChemEODEOL

    References
    -----------
    .. [Daigle2016] M. Daigle and C. Kulkarni, "End-of-discharge and End-of-life Prediction in Lithium-ion Batteries with Electrochemistry-based Aging Models," AIAA SciTech Forum 2016, San Diego, CA. https://arc.aiaa.org/doi/pdf/10.2514/6.2016-2132
    """
    states = ['qMax', 'Ro', 'D']
    events = ['InsufficientCapacity']
    inputs = ['i']
    outputs = []

    default_parameters = {
        'x0': {
            'qMax': 7600,
            'Ro': 0.117215,
            'D': 7e6
        },
        'wq': -1e-2,
        'wr': 1e-6,
        'wd': 1e-2,
        'qMaxThreshold': 5320
    }

    state_limits = {
        'qMax': (0, np.inf)
    }

    def dx(self, _, u):
        params = self.parameters

        return self.StateContainer(np.array([
            np.atleast_1d(params['wq'] * abs(u['i'])),
            np.atleast_1d(params['wr'] * abs(u['i'])),
            np.atleast_1d(params['wd'] * abs(u['i']))
        ]))

    def event_state(self, x) -> dict:
        e_state = (x['qMax']-self.parameters['qMaxThreshold'])/(self.parameters['x0']['qMax']-self.parameters['qMaxThreshold'])
        return {'InsufficientCapacity': max(min(e_state, 1.0), 0.0)}

    def threshold_met(self, x) -> dict:
        return {'InsufficientCapacity': x['qMax'] < self.parameters['qMaxThreshold']}

    def output(self, _):
        return self.OutputContainer(np.array([]))

def merge_dicts(a: dict, b: dict) -> None:
    """Merge dict b into a"""
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]

def OverwrittenWarning(params):
    """
    Function to warn if overwritten changes
    """
    warnings.warn("Ro, qMobile, and tDiffusion will be overwritten within the model as part of battery aging modeling. Use BatteryElectroChemEOD to remove this behavior.")
    return {}


class BatteryElectroChemEODEOL(BatteryElectroChemEOL, BatteryElectroChemEOD):
    """
    Prognostics :term:`model` for a battery degredation and discharge, represented by an electrochemical model as described in [Daigle2013]_ and [Daigle2016]_

    The default model parameters included are for Li-ion batteries, specifically 18650-type cells. Experimental discharge curves for these cells can be downloaded from the Prognostics Center of Excellence Data Repository [DataRepo]_.

    :term:`Events<event>`: (2)
        | EOD: End of Discharge
        | InsufficientCapacity: Insufficient battery capacity

    :term:`Inputs/Loading<input>`: (1)
        i: Current draw on the battery

    :term:`States<state>`: (11)
        See BatteryElectroChemEOD, BatteryElectroChemEOL

    :term:`Outputs<output>` (2)
        | t: Temperature of battery (°C) 
        | v: Voltage supplied by battery

    See Also
    --------
    BatteryElectroChemEOL, BatteryElectroChemEOD, BatteryElectroChem

    Note
    ----
    For keyword arguments, see BatteryElectroChemEOD, BatteryElectroChemEOL
    """
    inputs = BatteryElectroChemEOD.inputs
    outputs = BatteryElectroChemEOD.outputs
    states = BatteryElectroChemEOD.states + BatteryElectroChemEOL.states
    events = BatteryElectroChemEOD.events + BatteryElectroChemEOL.events

    is_vectorized = False

    default_parameters = deepcopy(BatteryElectroChemEOD.default_parameters)
    merge_dicts(default_parameters,
        BatteryElectroChemEOL.default_parameters)

    state_limits = deepcopy(BatteryElectroChemEOD.state_limits)
    state_limits.update(BatteryElectroChemEOL.state_limits)

    def __init__(self, **kwargs):
        self.param_callbacks['qMobile'].append(OverwrittenWarning)
        self.param_callbacks['tDiffusion'] = [OverwrittenWarning]
        self.param_callbacks['Ro'] = [OverwrittenWarning]
        super().__init__(**kwargs)

    def dx(self, x, u):
        # Set EOD Parameters (corresponding to health)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.parameters['qMobile'] = x['qMax']
            self.parameters['Ro'] = x['Ro']
            self.parameters['tDiffusion'] = x['D']
        
        # Calculate 
        x_dot = BatteryElectroChemEOD.dx(self, x, u)
        x_dot2 = BatteryElectroChemEOL.dx(self, x, u)
        x_dot.matrix = np.vstack((x_dot.matrix, x_dot2.matrix))
        return x_dot

    def output(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.parameters['qMobile'] = x['qMax']
        return BatteryElectroChemEOD.output(self, x)

    def event_state(self, x) -> dict:
        e_state = BatteryElectroChemEOD.event_state(self, x)
        e_state.update(BatteryElectroChemEOL.event_state(self, x))
        return e_state

    def threshold_met(self, x) -> dict:
        t_met = BatteryElectroChemEOD.threshold_met(self, x)
        t_met.update(BatteryElectroChemEOL.threshold_met(self, x))
        return t_met

BatteryElectroChem = BatteryElectroChemEODEOL

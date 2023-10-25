# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy
import numpy as np
import warnings

from progpy import PrognosticsModel


def calc_x(x: float, forces: float, Ls: float, new_x: float) -> float:
    lower_wall = (x == 0 and forces < 0) or (new_x < 0)
    upper_wall = (x == Ls and forces > 0) or (new_x > Ls)
    if lower_wall:
        return 0
    if upper_wall:
        return Ls
    return new_x


def calc_v(x: float, v: float, dv: float, forces: float, Ls: float, new_x: float) -> float:
    lower_wall = (x == 0 and forces < 0) or (new_x < 0)
    upper_wall = (x == Ls and forces > 0) or (new_x > Ls)
    if lower_wall or upper_wall:
        return 0
    return v + dv


class PneumaticValveBase(PrognosticsModel):
    """
    Prognostics :term:`model` for a Pneumatic Valve model as described in [DaigleValve2011]_.
    
    :term:`Events<event>`: (5)
        | Bottom Leak: Failure due to a leak at the bottom pneumatic port
        | Top Leak: Failure due to a leak at the top pneumatic port
        | Internal Leak: Failure due to an internal leak at the seal surrounding the piston
        | Spring Failure: Failure due to spring weakening with use
        | Friction Failure: Failure due to increase in friction along the piston with wear

    :term:`Inputs/Loading<input>`: (4)
        | pL: Fluid pressure at the left side of the plug (Pa)
        | pR: Fluid pressure at the right side of the plug (Pa)
        | uBot: input pressure at the bottom pneumatic port (Pa)
        | uTop: input pressure at the botton pneumatic port (Pa)

    :term:`States<state>`: (10)
        | Aeb: Area of the leak at the bottom pneumatic port
        | Aet: Area of the leak at the top pneumatic port
        | Ai: Area of the internal leak
        | k: Spring Coefficient
        | mBot: Mass on bottom of piston (kg)
        | mTop: Mass on top of the piston (kg)
        | r: Friction Coefficient
        | v: Velocity of the piston (m/s)
        | x: Poisition of the piston (m)
        | pDiff: Difference in pressure between the left and the right

    :term:`Outputs<output>`: 6
        | Q: Flowrate 
        | iB: Is the piston at the bottom (bool)
        | iT: Is the piston at the top (bool)
        | pB: Pressure at the bottom (Pa)
        | pT: Pressure at the top (Pa)
        | x: Position of piston (m)

    Keyword Args
    ------------
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state).
          Can be number (e.g., .2) applied to every state, a dictionary of
          values for each state (e.g., {'x1': 0.2, 'x2': 0.3}),
          or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` 
          e.g., normal, uniform, triangular
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of
          values for each output (e.g., {'z1': 0.2, 'z2': 0.3}),
          or a function (z) -> z
        measurement_noise_dist : Optional, str
           distribution for :term:`measurement noise`
           e.g., normal, uniform, triangular
        g : float
            Acceleration due to gravity (m/s^2)
        pAtm : float
            Atmospheric pressure (Pa)
        m : float
            Plug mass (kg)
        offsetX : float
            Spring offset distance (m)
        Ls : float
            Stroke Length (m)
        Ap : float
            Surface area of piston for gas contact (m^2)
        Vbot0 : float
            Below piston "default" volume (m^3)
        Vtop0 : float
            Above piston "default" volume (m^3)
        indicatorTol : float
            tolerance bound for open/close indicators
        pSupply : float
            Supply Pressure (Pa)
        Av : float
            Surface area of plug end (m^2)
        Cv : float
            flow coefficient assuming Cv of 1300 GPM
        rhoL : float
            density of LH2 in kg/m^3
        gas_mass : float
            Molar mass of supply gas (kg/mol)
        gas_temp : float
            Temperature of supply gas (K)
        gas_gamma, gas_z, gas_R : float
            Supply gas parameters
        At : float
        Ct : float
        Ab : float
        Cb : float
        AbMax : float
            Max limit for state Aeb
        AtMax : float
            Max limit for state Aet
        AiMax : float
            Max limit for state Ai
        kMin : float
            Min limit for state k
        rMax : float
            Max limit for state r
        x0 : Dict[str, float]
            Initial state
        wb: float
            Wear parameter for bottom leak
        wi: float
            Wear parameter for internal leak
        wt: float
            Wear parameter for top leak
        wk: float
            Wear parameter for spring
        wr: float
            Wear parameter for friction

    Note
    ----
    Supply gas parameters (gas_mass, gas_temp, gas_gamme, gas_z, gas_R) are for Nitrogen by default

    References
    ----------
    .. [DaigleValve2014] `M. Daigle and K. Goebel, "A Model-based Prognostics Approach Applied to Pneumatic Valves," International Journal of Prognostics and Health Management, vol. 2, no. 2, August 2011. https://papers.phmsociety.org/index.php/ijphm/article/view/1359`
    """
    events = ["Bottom Leak", "Top Leak", "Internal Leak", "Spring Failure", "Friction Failure"]
    inputs = ["pL", "pR", "uBot", "uTop"]
    states = [
        "Aeb",
        "Aet",
        "Ai",
        "k",
        "mBot",
        "mTop",
        "r",
        "v",
        "x",
        "pDiff"  # pL-pR
    ]
    outputs = ["Q", "iB", "iT", "pB", "pT", "x"]
    is_vectorized = False
    default_parameters = {  # Set to defaults
        # Environmental Parameters
        'R': 8.314,  # Universal Gas Constant
        'g': 9.81,
        'pAtm': 101325,

        # Valve Parameters
        'm': 50,
        'offsetX': 0.254,
        'Ls': 0.0381,
        'Ap': 8.1073196655599634694e-3,
        'Vbot0': 8.107319665559963e-4,
        'Vtop0': 8.107319665559963e-4,
        'indicatorTol': 1e-3,

        # Flow Parameters
        'pSupply': 5.272420892278394995e6,
        'Av': 0.050670747909749769389,
        'Cv': 0.4358892767469993814,
        'rhoL': 70.99,

        # Supply gas params (Note: Default is nitrogen)
        'gas_mass': 28.01e-3,
        'gas_temp': 293,
        'gas_gamma': 1.4,
        'gas_z': 1,
        'gas_R': 296.8225633702249454,

        # Orifice params
        'At': 1e-5,
        'Ct': 0.62,
        'Ab': 1e-5,
        'Cb': 0.62,

        # Limits
        "AbMax": 4e-5,
        "AtMax": 4e-5,
        "AiMax": 1.7e-6,
        "kMin": 3.95e4,
        "rMax": 4e6,

        # Initial state
        'x0': {
            'x': 0,
            'v': 0,
            'mTop': 0.067876043046174843,
            'mBot': 9.4455962535380932526e-4,
            'Aeb': 1e-5,
            'Ai': 0,
            'Aet': 1e-5,
            'k': 48000,
            'r': 6000
        },

        # Wear Rates
        'wb': 0,
        'wi': 0,
        'wk': 0,
        'wr': 0,
        'wt': 0
    }

    state_limits = {
        'Aeb': (0, np.inf),
        'Aet': (0, np.inf),
        'Ai': (0, np.inf),
        'k': (0, np.inf),
        'mBot': (0, np.inf),
        'mTop': (0, np.inf),
        'r': (0, np.inf)
    }

    def initialize(self, u, z=None):
        x0 = self.parameters['x0']
        x0['pDiff'] = u['pL'] - u['pR']
        return self.StateContainer(x0)

    def gas_flow(self, pIn: float, pOut: float, C: float, A: float) -> float:
        # Step 1: If array- run for each element
        # Note: this is so complicated because it is run multiple times
        # with mixtures of scalars and arrays
        inputs = np.array([pIn, pOut, C, A])
        if np.any([not np.isscalar(i) for i in inputs]):
            # Handle case where one or more is array
            size = [np.shape(i) for i in inputs]
            size = max([i[0] if i else 0 for i in size])  # Size of array

            # Create Iterable Elements for scalars
            iter_inputs = [[i] * size if np.isscalar(i) else i for i in inputs]

            # Run each element through function
            return np.array([self.gas_flow(a, b, c, d) for a, b, c, d in zip(*iter_inputs)])

        k = self.parameters['gas_gamma']
        T = self.parameters['gas_temp']
        Z = self.parameters['gas_z']
        R = self.parameters['gas_R']
        threshold = ((k+1)/2)**(k/(k-1))

        if pIn/pOut >= threshold:
            return C*A*pIn*np.sqrt(k/Z/R/T*(2/(k+1))**((k+1)/(k-1)))
        if pIn >= pOut:
            if pIn == 0:
                pIn = 1e-99
            return C*A*pIn*np.sqrt(2/Z/R/T*k/(k-1)*abs((pOut/pIn)**(2/k)-(pOut/pIn)**((k+1)/k)))
        if pOut/pIn >= threshold:
            return -C*A*pOut*np.sqrt(k/Z/R/T*(2/(k+1))**((k+1)/(k-1)))
        # pOut>pIn but pOut/pIn < threshold - only remaining possibility 
        return -C*A*pOut*np.sqrt(2/Z/R/T*k/(k-1)*abs((pIn/pOut)**(2/k)-(pIn/pOut)**((k+1)/k)))
    
    def next_state(self, x, u, dt: float):
        params = self.parameters  # optimization
        pInTop = params['pSupply'] if u['uTop'] else params['pAtm']
        springForce = x['k']*(params['offsetX']+x['x'])
        friction = x['v']*x['r']
        fluidForce = (u['pL']-u['pR'])*params['Av']
        pInBot = params['pSupply'] if u['uBot'] else params['pAtm']
        volumeBot = params['Vbot0'] + params['Ap']*x['x']
        volumeTop = params['Vtop0'] + params['Ap']*(params['Ls']-x['x'])
        plugWeight = params['m']*params['g']
        kdot = -params['wk']*abs(x['v']*springForce)
        rdot = params['wr']*abs(x['v']*friction)
        Aidot = params['wi']*abs(x['v']*friction)
        pressureBot = x['mBot']*params['R']*params['gas_temp']/params['gas_mass']/volumeBot
        mBotDotn = self.gas_flow(pInBot, pressureBot, params['Cb'], params['Ab'])
        pressureTop = x['mTop']*params['R']*params['gas_temp']/params['gas_mass']/volumeTop
        leakBotToAtm = self.gas_flow(pressureBot, params['pAtm'], 1, x['Aeb'])
        gasForceTop = pressureTop*params['Ap']
        gasForceBot = pressureBot*params['Ap']
        leakTopToAtm = self.gas_flow(pressureTop, params['pAtm'], 1, x['Aet'])
        leakTopToBot = self.gas_flow(pressureTop, pressureBot, 1, x['Ai'])
        mBotdot = mBotDotn + leakTopToBot - leakBotToAtm
        mTopDotn = self.gas_flow(pInTop, pressureTop, params['Ct'], params['At'])
        pistonForces = gasForceBot - fluidForce - plugWeight - friction - springForce - gasForceTop
        mTopdot = mTopDotn - leakTopToBot - leakTopToAtm
        vdot = pistonForces / params['m']

        new_x = x['x'] + x['v'] * dt

        if np.isscalar(pistonForces):
            vel = calc_v(x['x'], x['v'], vdot*dt, pistonForces, params['Ls'], new_x)
            pos = calc_x(x['x'], pistonForces, params['Ls'], new_x)
            dp = u['pL'] - u['pR']
        else:
            # If array- run for each element
            vel = [calc_v(xi, vi, vdot_i*dt, force, params['Ls'], new_x_i) for xi, vi, vdot_i, force, new_x_i in zip(x['x'], x['v'], vdot, pistonForces, new_x)]
            pos = [calc_x(xi, force, params['Ls'], new_x_i) for xi, force, new_x_i in zip(x['x'], pistonForces, new_x)]
            dp = [u['pL'] - u['pR']] * len(x['x'])

        return self.StateContainer(np.array([
            np.atleast_1d(x['Aeb'] + params['wb'] * dt),     # Aeb
            np.atleast_1d(x['Aet'] + params['wt'] * dt),     # Aet
            np.atleast_1d(x['Ai'] + Aidot * dt),             # Ai
            np.atleast_1d(x['k'] + kdot * dt),               # k
            np.atleast_1d(x['mBot'] + mBotdot * dt),         # mBot
            np.atleast_1d(x['mTop'] + mTopdot * dt),         # mTop
            np.atleast_1d(x['r'] + rdot * dt),               # r
            np.atleast_1d(vel),                              # v
            np.atleast_1d(pos),                              # x
            np.atleast_1d(dp)                                # pL - pR
        ]))
    
    def output(self, x):
        params = self.parameters  # Optimization
        indicatorTopm = (x['x'] >= params['Ls']-params['indicatorTol'])
        indicatorBotm = (x['x'] <= params['indicatorTol'])
        maxFlow = params['Cv']*params['Av']*np.sqrt(2/params['rhoL']*abs(x['pDiff'])) * np.sign(x['pDiff'])
        volumeBot = params['Vbot0'] + params['Ap']*x['x']
        volumeTop = params['Vtop0'] + params['Ap']*(params['Ls']-x['x'])
        trueFlow = maxFlow * np.maximum(0, x['x'])/params['Ls']
        pressureTop = x['mTop']*params['R']*params['gas_temp']/params['gas_mass']/volumeTop
        pressureBot = x['mBot']*params['R']*params['gas_temp']/params['gas_mass']/volumeBot

        return self.OutputContainer(np.array([
            np.atleast_1d(trueFlow),             # Q
            np.atleast_1d(indicatorBotm),        # indicatorBotm
            np.atleast_1d(indicatorTopm),        # indicatorTopm
            np.atleast_1d(1e-6 * pressureBot),   # pBot
            np.atleast_1d(1e-6 * pressureTop),   # pTop
            np.atleast_1d(x['x'])                # x
        ]))

    def event_state(self, x) -> dict:
        params = self.parameters
        return {
            "Bottom Leak": (params['AbMax'] - x['Aeb'])/(params['AbMax'] - params['x0']['Aeb']), 
            "Top Leak": (params['AtMax'] - x['Aet'])/(params['AtMax'] - params['x0']['Aet']), 
            "Internal Leak": (params['AiMax'] - x['Ai'])/(params['AiMax'] - params['x0']['Ai']),
            "Spring Failure": (x['k'] - params['kMin'])/(params['x0']['k'] - params['kMin']),
            "Friction Failure": (params['rMax'] - x['r'])/(params['rMax'] - params['x0']['r'])
        }

    def threshold_met(self, x) -> dict:
        params = self.parameters
        return {
            "Bottom Leak": x['Aeb'] > params['AbMax'], 
            "Top Leak": x['Aet'] > params['AtMax'], 
            "Internal Leak": x['Ai'] > params['AiMax'],
            "Spring Failure": x['k'] < params['kMin'],
            "Friction Failure": x['r'] > params['rMax']
        }


def OverwrittenWarning(params):
    """
    Function to warn if overwritten changes
    """
    warnings.warn("wb, wi, wk, wr and wt will be overwritten within the model, since the wear rates are part of the state. Use PneumaticValveBase to remove this behavior.")
    return {}


class PneumaticValveWithWear(PneumaticValveBase):
    """
    Prognostics :term:`model` for a pneumatic valve with wear parameters as part of the model state. This is identical to PneumaticValveBase, only PneumaticValveBase has the wear params as parameters instead of states

    This class implements a Pneumatic Valve model as described in the following paper:
    `M. Daigle and K. Goebel, "A Model-based Prognostics Approach Applied to Pneumatic Valves," International Journal of Prognostics and Health Management, vol. 2, no. 2, August 2011. https://www.phmsociety.org/node/602`

    :term:`Events<event>`: (4)
        See PneumaticValveBase

    :term:`Inputs/Loading<input>`: (5)
        See PneumaticValveBase
    
    :term:`States<state>`: (12)
        States from PneumaticValveBase +  wb, wi, wk, wr, wt

    :term:`Outputs<output>`: (5)
        See PneumaticValveBase

    Model Configuration Parameters:
        See PneumaticValveBase
    """
    inputs = PneumaticValveBase.inputs
    outputs = PneumaticValveBase.outputs
    states = PneumaticValveBase.states + ['wb', 'wi', 'wk', 'wr', 'wt']
    events = PneumaticValveBase.events

    default_parameters = deepcopy(PneumaticValveBase.default_parameters)
    default_parameters['x0'].update(
        {'wb': 0,
        'wi': 0,
        'wk': 0,
        'wr': 0,
        'wt': 0})

    state_limits = deepcopy(PneumaticValveBase.state_limits)

    param_callbacks = {
        'wb': [OverwrittenWarning],
        'wi': [OverwrittenWarning],
        'wk': [OverwrittenWarning],
        'wr': [OverwrittenWarning],
        'wt': [OverwrittenWarning]
    }

    def next_state(self, x, u, dt: float):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.parameters['wb'] = x['wb']
            self.parameters['wi'] = x['wi']
            self.parameters['wk'] = x['wk']
            self.parameters['wr'] = x['wr']
            self.parameters['wt'] = x['wt']
        next_x = PneumaticValveBase.next_state(self, x, u, dt)

        # Append this way because the keys in the structure but the values are
        # missing - this is due to the behavior of subclassed models calling
        # their parent functions.

        next_x.matrix = np.vstack((next_x.matrix, np.array([
            np.atleast_1d(x['wb']),
            np.atleast_1d(x['wi']),
            np.atleast_1d(x['wk']),
            np.atleast_1d(x['wr']),
            np.atleast_1d(x['wt'])
        ])))
        return next_x


PneumaticValve = PneumaticValveWithWear

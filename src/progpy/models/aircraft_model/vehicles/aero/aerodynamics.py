# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Functions for aerodynamics effects
"""

import numpy as np


class DragModel():
    """
    .. versionadded:: 1.5.0
    
    Bluff body drag model of the form:

            F_D = 1/2 * rho * A * Cd * V^2

    where rho is the air density, typically 1.225 kg/m^3, A is the apparent face of the body, depending on geometry.
    Cd is the drag coefficient, also dependent on geometry.
    V is the body or fluid speed, and F_D is the drag force opposing motion wrt the direction of V.
    """
    def __init__(self, bodyarea=None, Cd=None, air_density=None):
        """
        DragModel initialization function.

        :param bodyarea:            m^2, scalar, apparent face of the body
        :param Cd:                  -, scalar, drag coefficient
        :param air_density:         kg/m^3, scalar, air density
        """
        self.area = bodyarea
        self.cd = Cd
        self.rho = air_density

    def __call__(self, air_v):
        """
        DragModel call

        :param air_v:       m/s^2, scalar or n x 1 array or 1D vector, airspeed
        :return:            N, drag force, opposing motion.
        """
        vsq = air_v * np.abs(air_v)
        return 0.5 * self.rho * self.area * self.cd * vsq
 
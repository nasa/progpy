# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Geometric functions
"""

import numpy as np

# EARTH-RELATED DISTANCE FUNCTIONS
def greatcircle_distance(lat1, lat2, lon1, lon2, R=6371e3):
    """
    Compute distance between two points on a sphere, using a circle passing for those two points given the sphere's radius.
    Typically used to approximate the distance between two points on Earth, ignoring Earth's ellipsoid shape.

    :param lat1:        scalar, rad, latitude of first point
    :param lat2:        scalar, rad, latitude of second point
    :param lon1:        scalar, rad, longitude of first point
    :param lon2:        scalar, rad, longitude of second point
    :param R:           scalar, m radius of the sphere, default is Earth's radius: 6371e3 m
    :return:            scalar, m, distance between the two points.
    """
    phi1 = lat1
    phi2 = lat2
    dphi = (lat2 - lat1)
    dlambda = (lon2 - lon1)

    a = np.sin(dphi / 2.0) * np.sin(dphi / 2.0) + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) * np.sin(dlambda / 2.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d = R * c  # distance in meters
    return d


def vincenty_distance(p1, p2, tol=1e-12, max_iter=200):
    """ 
    Compute distance between two geodetic coordinates p1=(lat1, lon1), p2=(lat2, lon2) using ellipsoid equations from Vincenty, 1975.
    T. Vincenty, 1975. Direct and inverse solution of geodesics on the ellipsoid with application of nested equations. Survey Review 23(176), 88-93. 

    This algorithm provides a more accurate estimate of the distance between two points on Earth compared to the Great Circle distance, since
    it accounts for the ellipsoid shape. 

    The algorithm does not converge for two nearly antipodal points, for example: (0.0, 0.0), (0.5, 179.5).

    :param p1:          tuple, list or 1D array, latitude and longitude of first point on Earth
    :param p2:          tuple, list or 1D array, latitude and longitude of second point on Earth
    :param tol:         scalar, tolerance to distance precision, default = 1e-12 typically refers to <1cm error
    :param max_iter:    int, number of maximum iterations before breaking algorithm. Default = 200
    :return:            distance between p1 and p2, in meters.
    """
    # Extract latitude and longitude values
    # ------------------------------------
    lat1, lon1 = p1
    lat2, lon2 = p2

    # If points are identical, return distance=0
    if lat1 == lat2 and lon1 == lon2:
        return 0.0
    
    # Define ellipsoid constants
    a = 6378137.0          # semi-major Earth axis (radius at Equator), meters, according to WGS84
    f = 1/298.257223563    # flat parameter of the ellipsoid, according to WGS84
    b = (1.0 - f) * a      # semi-minor axis of the ellipsoid (radius at the poles), meters, according to WGS84 = 6356752.314245 
    
    # Define coordinate-dependent values
    U1 = np.arctan((1.0 - f) * np.tan(lat1))    # reduced latitude (latitude on auxiliary sphere, lat1)
    U2 = np.arctan((1.0 - f) * np.tan(lat2))    # reduced latitude (latitude on auxiliary sphere, lat2)
    L = lon2 - lon1                        # difference over longitude of the two points
    # Compute trigonometric values for U1, U2
    sin_U1 = np.sin(U1)
    cos_U1 = np.cos(U1)
    sin_U2 = np.sin(U2)
    cos_U2 = np.cos(U2)

    lam = L    # initialize longitude difference between p1 and p2 on auxiliary sphere. It should asymptotically converge to 0
    iter = 0    # initialize iterator
    while iter < max_iter:
        # Trigonometry of lambda
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)

        # Trigonometry of sigma
        sin_sigma = np.sqrt((cos_U2 * sin_lam)**2.0 + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam)**2.0)
        if sin_sigma == 0.0:
            return 0.0  # coincident points
        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)
        sigma = np.arctan2(sin_sigma, cos_sigma)

        # Trigonometry of alpha
        sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
        sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
        cos_alpha2 = 1.0 - sin_alpha**2.0
        # Compute cos(2 \sigma_m)
        try:
            cos_2sigma_m = cos_sigma - 2.0 * sin_U1 * sin_U2 / cos_alpha2
        except ZeroDivisionError:
            cos_2sigma_m = 0.0
        try:
            cos_2sigma_m = cos_sigma - 2.0 * sin_U1 * sin_U2 / cos_alpha2
        except ZeroDivisionError:
            cos_2sigma_m = 0.0
        
        # Compute new lambda
        C = f / 16.0 * cos_alpha2 * (4.0 + f * (4.0 - 3.0 * cos_alpha2))
        lam_old = lam
        lam = L + (1.0 - C) * f * sin_alpha * (sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)))
        
        # Evaluate difference
        d_lam = abs(lam - lam_old)
        if d_lam < tol:
            break
        iter += 1  # update iterator
        if d_lam < tol:
            break
        iter += 1  # update iterator
    
    # Return value
    # ----------
    if d_lam > tol or iter == max_iter:
        # Failure to converge
        return None
    else:
        # After lambda converged, compute the following:
        u2 = cos_alpha2 * (a*a - b*b) / b**2.0
        A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-786.0 + u2 * (320.0 - 175.0 * u2)))
        B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
        dsigma = B * sin_sigma * (cos_2sigma_m + 1.0/4.0 * B * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m) - B / 6.0 * cos_2sigma_m * (-3.0 + 4.0 * sin_sigma*sin_sigma) * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)))
        s = b * A * (sigma - dsigma)
    return np.round(s, 6)


def geodetic_distance(lats, lons, alts, method='greatcircle', return_surf_vert=False):
    """
    Compute geodetic distance between two points defined by latitude, longitude, and altitude
            dist                = geodetic_distance(lats, lons, alts, method='greatcircle', return_surf_vert=False)
            surf_dist, ver_dist = geodetic_distance(lats, lons, alts, method='greatcircle', return_surf_vert=True)

    Distance in 3D using: add altitude as cartesian coordinate
    dist = sqrt(surface_distance**2 + (alt_1 - alt_2)**2)

    :param lats:                    tuple, list or 1D array, latitudes of the two points (radians)
    :param lons:                    tuple, list or 1D array, longitudes of the two points (radians)
    :param alts:                    tuple, list or 1D array, altitudes of the two points (meters)
    :param method:                  string, measurement method to compute distance. Options: 'greatcircle' and 'vincenty'
    :param return_surf_vert:        Boolean, whether to return surface distance and vertical distance separately (default=False)
    :return:                        distance between two points in meters. Tuple of values (surface and alt distance) if return_surf_vert==True; absolute distance otherwise 
    """
    if len(lats) != 2 or len(lons) != 2 or len(alts) != 2:
        raise ValueError("Latitudes, longitudes, and altitude values must be 2-element lists or arrays.")
    
    if type(alts[0]) == np.ndarray:  # if altitudes are vectors, compute point-wise difference (must be the same length)
        if len(alts[0]) != len(alts[1]):
            raise ValueError("If altitudes are vectors, their length must coincide.")
        vert_dist = alts[1] - alts[0]
    else:  # if alts are two points, compute the difference between them
        vert_dist = np.diff(alts)   # compute the difference in altitude
    
    # Compute geodetic distance according to the method
    if method == 'greatcircle':
        surface_dist = greatcircle_distance(lats[0], lats[1], lons[0], lons[1])
    elif method == 'vincenty':
        surface_dist = vincenty_distance([lats[0], lons[0]], [lats[1], lons[1]])
    else:
        raise Exception("Geodetic distance method " + method + " not recognized.")
    
    # Return horizontal and vertical distance or total distance
    if return_surf_vert:
        return surface_dist, vert_dist
    else:
        return np.sqrt(surface_dist**2.0 + vert_dist**2.0)


def euclidean_distance_point_vector(point, vector):
    """
    Return the euclidean distance between multi-dimensional point and vector, the vector should be organized by column, i.e., 
    each row is a value in the vector, and each column corresponds to a dimension.
    Number of dimensions must match for point and vector:
    
    point : n x 1 array
    vector: m x n array

    :param point:   n x 1 array
    :param vector:  m x n array
    """
    point_rep = np.repeat(point.reshape((1, -1)), repeats=vector.shape[0], axis=0)
    d = np.sum(abs(point_rep - vector)**2.0, axis=1)
    return np.sqrt(d)

    
# REFERENCE FRAMES
def rot_eart2body_fast(sphi, cphi, stheta, ctheta, spsi, cpsi):
    """
    Return the rotation matrix R to transform coordinates from an Earth-fixed (inertial) reference frame to a body reference frame.
    The function is called "fast" because it takes the pre-computed sine, cosine, and other trigonometric functions of Euler's angles.

    :param sphi:     scalar, sine of phi
    :param cphi:     scalar, cosine of phi
    :param stheta:   scalar, sine of theta
    :param ctheta:   scalar, cosine of theta
    :param spsi:     scalar, sine of psi
    :param cpsi:     scalar, cosine of psi
    :return:         rotation matrix to transform coordinates from Earth-fixed frame to body frame
    """
    return np.array([[ctheta * cpsi, ctheta * spsi, -stheta],
                     [-cphi * spsi + sphi * stheta * cpsi, cphi * cpsi + sphi * stheta * spsi, sphi * ctheta],
                     [sphi * spsi + cphi * stheta * cpsi, -sphi * cpsi + cphi * stheta * spsi, cphi * ctheta]])



def rot_body2earth_fast(sphi, cphi, stheta, ctheta, spsi, cpsi):
    """
    Return the rotation matrix R to transform coordinates from a body (non-inertial) reference frame to an Earth-fixed (inertial) reference frame.
    The function is called "fast" because it takes the pre-computed sine, cosine, and other trigonometric functions of Euler's angles.

    :param sphi:     scalar, sine of phi
    :param cphi:     scalar, cosine of phi
    :param stheta:   scalar, sine of theta
    :param ctheta:   scalar, cosine of theta
    :param spsi:     scalar, sine of psi
    :param cpsi:     scalar, cosine of psi
    :return:         rotation matrix to transform coordinates from body frame to Earth-fixed frame
    """
    return np.array([[cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
                     [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
                     [-stheta, ctheta * sphi, ctheta * cphi]])


def body_ang_vel_from_eulers(phi, theta, psi, phidot, thetadot, psidot):
    """ 
    Compute the desired body angular velocities p, q, r given the desired Euler's angular velocities with respect to the inertial (Earth) reference frame.
    The desired body angular velocities will be used to compute the desired moments and total thrust given the rotor configuration. From the desired
    thrust and moments, it is possible to compute the desired rotor speed given the thrust allocation matrix (UAV dependent).

    :param phi:         first Euler's angle (roll)
    :param theta:       second Euler's angle (pitch)
    :param psi:         third Euler's angle (yaw)
    :param phidot:      first Euler's angle (roll) rate of change
    :param thetadot:    second Euler's angle (pitch) rate of change
    :param psidot:      third Euler's angle (yaw) rate of change
    :return:            body angular velocities organized in a column vector [p, q, r]^T
    """
    p = phidot - psidot * np.sin(theta)
    q = thetadot * np.cos(phi) + psidot * np.sin(phi) * np.cos(theta)
    r = -thetadot * np.sin(phi) + psidot * np.cos(phi) * np.cos(theta)
    return p, q, r

def gen_heading_angle(lat, lon, alt):
    """
    Function to generate heading angle to follow a set of points defined by latitude (lat), longitude (lon), and altitude (alt).
    The function returns the heading from North, positive clock-wise

    :param lat:             rad, n x 1, latitude points
    :param lon:             rad, n x 1, longitude points
    :param alt:             ft or m, n x 1, altitude points
    :return:                rad, n x 1, heading angle to follow the input points
    """
    # Compute heading using lat-lon coordinates
    # -----------------------------------------
    # This heading is calculated from North, while reference frame is ENU (East-North-Up), therefore, the first direction is EAST, not NORTH.
    # Need to adjust heading for ENU reference frame after calculating it.
    head = heading_compute_geodetic(lat, lon)
    # Adjust first heading based on altitude (avoid issues with fictitious waypoints later on)
    head = heading_adjust_first_nonzero(head, alt)

    # Adjust heading based on minimum rotation between current and next heading.
    # ---------------------------------------------------------------------------
    # Select the sign of the angle based on the shortest rotation the UAV is supposed to move
    head = heading_adjust_rotation(head)
    return head


def heading_adjust_first_nonzero(heading, altitude):
    """
    Adjust first non-zero heading angle based on altitude.
    :param heading:     rad, n x 1, heading angles
    :param altitude:    m, n x 1, altitude points
    :return:            adjusted heading angles
    """
    n = len(heading)
    for jj in range(n - 1):
        if altitude[jj] > 1.0 and heading[jj] == 0.0 and heading[jj + 1] != 0.0:
            heading[jj] = heading[jj + 1]
            break
    return heading


def heading_adjust_rotation(heading):
    """
    Adjust heading w.r.t. direction of travel. The sign of the heading should be decided based on the shortest rotation the UAV is supposed to move.
    Example:
        heading at i -th point = 60 deg
        heading at i+1 -th point = -270 deg
        change heading at i+1 -th point to 90 deg, so the vehicle will rotate only 30 deg from 60 to 90, instead of 60 to -270.
    
    :param heading:     rad, n x 1, heading angles.
    :return:            rad, n x 1, heading angles with the shorter rotation distance between point i and i+1
    """
    n = len(heading)
    for idx in range(n-1):
        curr_p = heading[idx]
        next_p = heading[idx+1]
        # If angles are identical move on
        if curr_p == next_p:
            continue
        else:
            next_p_2 = next_p - 2.0*np.pi   # angle in opposite direction
            # Select angle based on minimum rotation necessary
            if abs(curr_p - next_p) < abs(curr_p - next_p_2):
                heading[idx+1] = next_p
            else:
                heading[idx+1] = next_p_2
    return heading


def heading_compute_geodetic(lat, lon):
    """
    Compute heading angle to follow a set of points defined by geodetic coordinates lat, lon.

    :param lat:         rad, n x 1, latitude points
    :param lon:         rad, n x 1, longitude points
    :return:            rad, n x 1, heading angle
    """
    n = len(lat)
    heading = np.zeros((n,))
    for jj in range(1, n):
        dlon_ = lon[jj] - lon[jj - 1]  # compute difference in longitude
        dlat_ = lat[jj] - lat[jj - 1]  # compute difference in latitude
        X = np.cos(lat[jj]) * np.sin(dlon_)  # compute cartesian coordinate
        Y = np.cos(lat[jj - 1]) * np.sin(lat[jj]) - np.sin(lat[jj - 1]) * np.cos(lat[jj]) * np.cos(dlon_)  # compute cartesian coordinate
        head_temp = np.arctan2(X, Y)  # heading as arc-tangent of cartesian coordinates
        if Y != 0:  # adjust for 360 degrees
            head_temp -= np.pi / 2.0
            head_temp *= -1.0
        if jj < n - 1 and ((dlat_ != 0 or dlon_ != 0) and head_temp == 0):
            # if calculated heading is 0, keep the old one.
            head_temp = heading[jj - 2]
        heading[jj - 1] = head_temp
    heading[-1] = heading[-2]
    return heading


class Coord():
    """
    Coordinate class:
    transform coordinate values between geodetic frame and cartesian frames, and vice versa.
    """
    def __init__(self, lat0, lon0, alt0):
        """
        Initialize Coordinate frame class.

        :param lat0:            Latitude of origin of reference frame
        :param lon0:            Longitude of origin of reference frame
        :param alt0:            Altitude of origin of reference frame
        """
        self.a = 6378137.0  # [m], equatorial radius
        self.f = 1.0 / 298.257223563  # [-], ellipsoid flatness
        self.b = self.a * (1.0 - self.f)  # [m], polar radius
        self.e = np.sqrt(self.f * (2 - self.f))  # [-], eccentricity
        self.lat0 = lat0
        self.lon0 = lon0
        self.alt0 = alt0
        self.N0 = self.a / np.sqrt(1 - self.e**2.0 * np.sin(self.lat0)**2.0)  # [m], Radius of curvature on the Earth

    def ecef2enu(self, xecef, yecef, zecef):
        """
        Conversion of Earth-centered, Earth-fixed (ECEF) reference frame coordinates to East-North-UP (ENU) reference frame.

        :param xecef:           m, n x 1, X-coordinate in ECEF reference frame
        :param yecef:           m, n x 1, Y-coordinate in ECEF reference frame
        :param zecef:           m, n x 1, Z-coordinate in ECEF reference frame
        :return x:              m, n x 1, x-coordinate (East) in ENU reference frame
        :return y:              m, n x 1, y-coordinate (North) in ENU reference frame
        :return z:              m, n x 1, z-coordinate (Up) in ENU reference frame
        """
        # Compute location of the origin of ENU reference frame in the ECEF reference frame
        x0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.cos(self.lon0)
        y0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.sin(self.lon0)
        z0 = (self.alt0 + (1.0 - self.e**2.0) * self.N0) * np.sin(self.lat0)

        # Compute relative distance between the data points and the origin of the ENU reference frame
        xd = xecef - x0
        yd = yecef - y0
        zd = zecef - z0

        # Compute coordinates in ENU reference frame
        x  = -np.sin(self.lon0) * xd + np.cos(self.lon0) * yd
        y  = -np.cos(self.lon0) * np.sin(self.lat0) * xd - np.sin(self.lat0) * np.sin(self.lon0) * yd + np.cos(self.lat0) * zd
        z  = np.cos(self.lat0) * np.cos(self.lon0) * xd + np.cos(self.lat0) * np.sin(self.lon0) * yd + np.sin(self.lat0) * zd

        return x, y, z

    def enu2ecef(self, xenu, yenu, zenu):
        """
        Conversion of East-North-UP (ENU) reference frame coordinates into Earth-centered, Earth-fixed (ECEF) reference frame.

        :param xenu:             m, n x 1, x-coordinate (East) in ENU reference frame
        :param yenu:             m, n x 1, y-coordinate (North) in ENU reference frame
        :param zenu:             m, n x 1, z-coordinate (Up) in ENU reference frame
        :return xecef:           m, n x 1, X-coordinate in ECEF reference frame
        :return yecef:           m, n x 1, Y-coordinate in ECEF reference frame
        :return zecef:           m, n x 1, Z-coordinate in ECEF reference frame
        """
        # Compute coordinates of origin of ENU in the ECEF reference frame
        x0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.cos(self.lon0)
        y0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.sin(self.lon0)
        z0 = (self.alt0 + (1.0 - self.e**2.0) * self.N0) * np.sin(self.lat0)

        # Compute relative coordinates in ECEF reference frame
        xd = -np.sin(self.lon0) * xenu - np.cos(self.lon0) * np.sin(self.lat0) * yenu + np.cos(self.lat0) * np.cos(self.lon0) * zenu
        yd = np.cos(self.lon0) * xenu - np.sin(self.lat0) * np.sin(self.lon0) * yenu + np.cos(self.lat0) * np.sin(self.lon0) * zenu
        zd = np.cos(self.lat0) * yenu + np.sin(self.lat0) * zenu

        # Compute global coordinates in ECEF reference frame by adding the location of the ENU to the relative ECEF coordinates
        xecef = xd + x0
        yecef = yd + y0
        zecef = zd + z0
        return xecef, yecef, zecef


    def ecef2geodetic(self, X, Y, Z):
        """
        lat, lon, alt = Coord.ecef2geodetic(X, Y, Z)

        Convert ECEF coordinates into geodetic coordinates.

        The equations used in this function come from the report:
            Geomatics Guidance Note 7, part 2 Coordinate Conversions & Transformations including Formulas
            International Association of Oil & Gas producers, Report 373-7-2,
            April 2018.
            See Chapter 2, Subsection 2.2, Sub-subsection 2.2.1

        INPUT
            X       n x 1 array, real, [m], X coordinate in ECEF reference system
            Y       n x 1 array, real, [m], Y coordinate in ECEF reference system
            Z       n x 1 array, real, [m], Z coordinate in ECEF reference system

        OUTPUT
            lat     n x 1 array, real, [rad], latitude values
            lon     n x 1 array, real, [rad], longitude values
            alt     n x 1 array, real, [m], altitude values
        """
        epsilon = self.e**2.0 / (1.0 - self.e**2.0)
        b = self.a * (1.0 - self.f)
        p = np.sqrt(X**2.0 + Y**2.0)
        q = np.arctan2(Z * self.a, p * b)

        # Compute latitude, longitude and altitude.
        lat = np.arctan2((Z + epsilon * b * np.sin(q)**3.0), (p - self.e**2.0 * self.a * np.cos(q)**3.0))  # [rad], latitude
        lon = np.arctan2(Y, X)  # [rad], longitude
        N = self.a / np.sqrt(1.0 - self.e**2.0 * np.sin(lat)**2.0)  # [m], Radius of curvature on the Earth
        alt = (p / np.cos(lat)) - N
        return lat, lon, alt


    def geodetic2ecef(self, lat, lon, alt):
        """
        Conversion of geodetic reference frame coordinates into Earth-centered, Earth-fixed (ECEF) reference frame.

        :param lat:     rad, n x 1, latitude 
        :param lon:     rad, n x 1, longitude
        :param alt:     m, n x 1, altitude
        :return X:      m, n x 1, X-coordinate in ECEF reference frame
        :return Y:      m, n x 1, Y-coordinate in ECEF reference frame
        :return Z:      m, n x 1, Z-coordinate in ECEF reference frame
        """
        N = self.a / np.sqrt(1.0 - self.e**2.0 * np.sin(lat)**2.0)  # [m], Radius of curvature on the Earth, phi = latitude
        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (alt + (1.0 - self.e**2.0) * N) * np.sin(lat)
        return X, Y, Z


    def enu2geodetic(self, x, y, z):
        """
        Conversion of East-North-UP (ENU) reference frame coordinates into geodetic coordinates.
        This method calls first method ENU2ECEF, and then ECEF2GEODETIC.

        :param x:              m, n x 1, x-coordinate (East) in ENU reference frame
        :param y:              m, n x 1, y-coordinate (North) in ENU reference frame
        :param z:              m, n x 1, z-coordinate (Up) in ENU reference frame
        :return lat:           rad, n x 1, latitude in geodetic coordinates
        :return lon:           rad, n x 1, longitude in geodetic coordinates
        :return alt:           m, n x 1, altitude in geodetic coordinates
        """
        xecef_tmp, yecef_tmp, zecef_tmp = self.enu2ecef(x, y, z)
        return self.ecef2geodetic(xecef_tmp, yecef_tmp, zecef_tmp )


    def geodetic2enu(self, lat, lon, alt):
        """
        Conversion of coordinates (lat, lon, alt) in geodetic reference frame into East-North-UP (ENU) reference frame.
        This method calls first method GEODETIC2ECEF, and then ECEF2ENU.

        :param lat:           rad, n x 1, latitude in geodetic coordinates
        :param lon:           rad, n x 1, longitude in geodetic coordinates
        :param alt:           m, n x 1, altitude in geodetic coordinates
        :return x:            m, n x 1, x-coordinate (East) in ENU reference frame
        :return y:            m, n x 1, y-coordinate (North) in ENU reference frame
        :return z:            m, n x 1, z-coordinate (Up) in ENU reference frame
        """
        X, Y, Z = self.geodetic2ecef(lat, lon, alt)
        return self.ecef2enu(X, Y, Z)

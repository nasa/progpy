# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

# ---------------------------
# Measurement Noise Functions
# ---------------------------


def uniform_measurement_noise(self, z):
    noise_mat = self.parameters['measurement_noise'].matrix
    z.matrix = z.matrix + np.random.uniform(-1*noise_mat, noise_mat, size=z.matrix.shape)
    return z


def triangular_measurement_noise(self, z):
    noise_mat = self.parameters['measurement_noise'].matrix
    z.matrix = z.matrix + np.random.triangular(-1*noise_mat, 0, noise_mat, size=z.matrix.shape)
    return z


def normal_measurement_noise(self, z):
    noise_mat = self.parameters['measurement_noise'].matrix
    z.matrix = z.matrix + np.random.normal(0, noise_mat, size=z.matrix.shape)
    return z


def no_measurement_noise(self, z):
    return z


measurement_noise_functions = {
    'uniform': uniform_measurement_noise,
    'triangular': triangular_measurement_noise,
    'normal': normal_measurement_noise,
    'gaussian': normal_measurement_noise,
    'none': no_measurement_noise,
}

# ---------------------------
# Process Noise Functions
# ---------------------------


def triangular_process_noise(self, x, dt: float = 1):
    noise_mat = self.parameters['process_noise'].matrix
    noise = np.random.triangular(-1*noise_mat, 0, noise_mat, size=x.matrix.shape)
    x.matrix = x.matrix + dt*noise
    return x


def uniform_process_noise(self, x, dt: float = 1):
    noise_mat = self.parameters['process_noise'].matrix
    noise = np.random.uniform(-1*noise_mat, noise_mat, size=x.matrix.shape)
    x.matrix = x.matrix + dt*noise
    return x


def normal_process_noise(self, x, dt: float = 1):
    noise_mat = self.parameters['process_noise'].matrix
    noise = np.random.normal(0, noise_mat, size=x.matrix.shape)
    x.matrix = x.matrix + dt*noise
    return x


def no_process_noise(self, x, dt: float = 1) -> dict:
    return x


process_noise_functions = {
    'uniform': uniform_process_noise,
    'triangular': triangular_process_noise,
    'normal': normal_process_noise,
    'gaussian': normal_process_noise,
    'none': no_process_noise,
}

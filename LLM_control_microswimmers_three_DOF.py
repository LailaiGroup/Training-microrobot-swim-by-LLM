import openai
import math
import numpy as np
import os
from retrying import retry
import re
import random

# Replace 'YOUR_OPENAI_API_KEY' with your actual API key
openai.api_key = ' '

def save_and_notify(filename, data):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if os.path.exists(filename):
        pass
    np.savetxt(filename, data, fmt='%f')

class ThreeLink:
    def __init__(self):
        self.gamma = None
        self.theta_1 = None
        self.theta_2 = None
        self.theta_3 = None
        self.Phi_1 = None
        self.Phi_2 = None
        self.Phi_1_last = None
        self.Phi_2_last = None
        self.L = None
        self.b = None
        self.dt = None
        self.dt_ratio = None
        self.xy_1 = None
        self.xy_2 = None
        self.xy_3 = None
        self.xy_4 = None
        self.x_position = None
        self.y_position = None
        self.trajectory = None
        self.pos_diff = None
        self.range = None
        self.low_lim = None
        self.up_lim = None
        self.pos = None
        self.prev_pos = None
        self.pos_diff_history = None

    def reset_world(self):
        self.gamma = math.pi / 3
        self.theta_1 = -self.gamma
        self.Phi_1 = self.gamma
        self.Phi_2 = self.gamma
        self.Phi_1_last = self.gamma
        self.Phi_2_last = self.gamma
        self.theta_2 = self.Phi_1 + self.theta_1
        self.theta_3 = self.Phi_2 + self.theta_2
        self.L = 1
        self.b = 1
        self.dt = 2 * np.abs(self.gamma)
        self.dt_ratio = 300
        self.range = 1e-6
        self.low_lim = -np.abs(self.gamma)
        self.up_lim = np.abs(self.gamma)
        self.xy_1 = np.array([-math.cos(self.theta_1) - 1 / 2, -math.sin(self.theta_1)])
        self.xy_2 = np.array([0, 0.0])
        self.xy_3 = np.array([0, 0.0])
        self.xy_4 = np.array([0, 0.0])
        self.xy_2[0] = self.xy_1[0] + (self.L) * math.cos(self.theta_1)
        self.xy_2[1] = self.xy_1[1] + (self.L) * math.sin(self.theta_1)
        self.xy_3[0] = self.xy_2[0] + self.b * (self.L) * math.cos(self.theta_2)
        self.xy_3[1] = self.xy_2[1] + self.b * (self.L) * math.sin(self.theta_2)
        self.xy_4[0] = self.xy_3[0] + (self.L) * math.cos(self.theta_3)
        self.xy_4[1] = self.xy_3[1] + (self.L) * math.sin(self.theta_3)
        self.x_position = (self.xy_1[0] + self.xy_2[0] + self.xy_3[0] + self.xy_4[0]) / 4
        self.y_position = (self.xy_1[1] + self.xy_2[1] + self.xy_3[1] + self.xy_4[1]) / 4
        self.pos = np.array([self.x_position, self.y_position])
        self.prev_pos = self.pos
        self.pos_diff = self.pos - self.prev_pos
        self.trajectory = []
        self.pos_diff_history = []

    def observation_RK4(self, u, dt_ratio):
        def rk4_step(y, u, dt_num, dydt_func):
            k1 = dt_num * dydt_func(y, u)
            k2 = dt_num * dydt_func(y + 0.5 * k1, u)
            k3 = dt_num * dydt_func(y + 0.5 * k2, u)
            k4 = dt_num * dydt_func(y + k3, u)
            return y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        def dydt_angular_Phi(Phi, Omega):
            if Phi + self.range >= self.up_lim and Omega > 0:
                Omega = 0
            elif Phi - self.range <= self.low_lim and Omega < 0:
                Omega = 0
            return Omega

        dt_num = self.dt / dt_ratio
        steps = int(self.dt / dt_num)

        Omega_2 = u[0]
        Omega_3 = u[1]
        total = np.zeros(13)
        self.Phi_1_last = self.Phi_1
        self.Phi_2_last = self.Phi_2
        for _ in range(steps+1):
            self.Phi_1 = rk4_step(self.Phi_1, Omega_2, dt_num, dydt_angular_Phi)
            self.Phi_2 = rk4_step(self.Phi_2, Omega_3, dt_num, dydt_angular_Phi)

            if self.Phi_1 + self.range >= self.up_lim and self.Phi_2 + self.range >= self.up_lim:
                self.Phi_1 = self.up_lim
                self.Phi_2 = self.up_lim
            elif self.Phi_1 - self.range <= self.low_lim and self.Phi_2 + self.range >= self.up_lim:
                self.Phi_1 = self.low_lim
                self.Phi_2 = self.up_lim
            elif self.Phi_1 - self.range <= self.low_lim and self.Phi_2 - self.range <= self.low_lim:
                self.Phi_1 = self.low_lim
                self.Phi_2 = self.low_lim
            elif self.Phi_1 + self.range >= self.up_lim and self.Phi_2 - self.range <= self.low_lim:
                self.Phi_1 = self.up_lim
                self.Phi_2 = self.low_lim

            self.theta_2 = self.theta_1 + self.Phi_1
            self.theta_3 = self.theta_2 + self.Phi_2
            Omega_2 = dydt_angular_Phi(self.Phi_1, Omega_2)
            Omega_3 = dydt_angular_Phi(self.Phi_2, Omega_3)

            velocity_x = - (((8 * (
                    12 + 21 * self.b + 29 * self.b ** 2 + 33 * self.b ** 3 + 15 * self.b ** 4 + 2 * self.b ** 5) * math.sin(
                self.theta_1) + 2 * self.b * (-4 + 12 * self.b + 9 * self.b ** 2 + 2 * self.b ** 3) * math.sin(
                self.theta_1 - 2 * self.theta_2) -
                              8 * self.b * math.sin(3 * self.theta_1 - 2 * self.theta_2) + 24 * self.b ** 2 * math.sin(
                        3 * self.theta_1 - 2 * self.theta_2) + 18 * self.b ** 3 * math.sin(
                        3 * self.theta_1 - 2 * self.theta_2) + 4 * self.b ** 4 * math.sin(
                        3 * self.theta_1 - 2 * self.theta_2) +
                              72 * self.b * math.sin(2 * self.theta_1 - self.theta_2) + 88 * self.b ** 2 * math.sin(
                        2 * self.theta_1 - self.theta_2) + 24 * self.b ** 3 * math.sin(
                        2 * self.theta_1 - self.theta_2) - 16 * self.b * math.sin(self.theta_2) -
                              24 * self.b ** 2 * math.sin(self.theta_2) - 8 * self.b ** 3 * math.sin(
                        self.theta_2) + 4 * math.sin(self.theta_1 - 2 * self.theta_3) + self.b ** 3 * math.sin(
                        self.theta_1 - 2 * self.theta_3) + 4 * math.sin(3 * self.theta_1 - 2 * self.theta_3) +
                              self.b ** 3 * math.sin(3 * self.theta_1 - 2 * self.theta_3) + 2 * self.b ** 2 * math.sin(
                        self.theta_2 - 2 * self.theta_3) + 8 * self.b * math.sin(
                        2 * self.theta_1 + self.theta_2 - 2 * self.theta_3) + 6 * self.b ** 2 * math.sin(
                        2 * self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                              4 * self.b * math.sin(
                        self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) + 12 * self.b ** 2 * math.sin(
                        self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) + 12 * self.b ** 3 * math.sin(
                        self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) +
                              4 * self.b ** 4 * math.sin(
                        self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) + 2 * self.b ** 2 * math.sin(
                        3 * self.theta_2 - 2 * self.theta_3) + 40 * math.sin(
                        2 * self.theta_1 - self.theta_3) + 36 * self.b * math.sin(2 * self.theta_1 - self.theta_3) +
                              24 * self.b * math.sin(
                        self.theta_1 - self.theta_2 - self.theta_3) + 12 * self.b ** 2 * math.sin(
                        self.theta_1 - self.theta_2 - self.theta_3) + 24 * self.b * math.sin(
                        3 * self.theta_1 - self.theta_2 - self.theta_3) + 12 * self.b ** 2 * math.sin(
                        3 * self.theta_1 - self.theta_2 - self.theta_3) +
                              96 * self.b * math.sin(
                        self.theta_1 + self.theta_2 - self.theta_3) + 144 * self.b ** 2 * math.sin(
                        self.theta_1 + self.theta_2 - self.theta_3) + 48 * self.b ** 3 * math.sin(
                        self.theta_1 + self.theta_2 - self.theta_3) + 4 * self.b * math.sin(
                        2 * self.theta_2 - self.theta_3) -
                              8 * math.sin(self.theta_3) - 12 * self.b * math.sin(
                        self.theta_3) - 12 * self.b ** 2 * math.sin(
                        self.theta_1 - 3 * self.theta_2 + self.theta_3) - 12 * self.b ** 2 * math.sin(
                        3 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                              12 * self.b * math.sin(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                              96 * self.b * math.sin(self.theta_1 - self.theta_2 + self.theta_3) +
                              144 * self.b ** 2 * math.sin(self.theta_1 - self.theta_2 + self.theta_3) +
                              48 * self.b ** 3 * math.sin(self.theta_1 - self.theta_2 + self.theta_3) -
                              3 * self.b ** 3 * math.sin(self.theta_1 - 4 * self.theta_2 + 2 * self.theta_3) -
                              3 * self.b ** 3 * math.sin(3 * self.theta_1 - 4 * self.theta_2 + 2 * self.theta_3) -
                              6 * self.b ** 2 * math.sin(2 * self.theta_1 - 3 * self.theta_2 + 2 * self.theta_3) +
                              4 * self.b * math.sin(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) +
                              12 * self.b ** 2 * math.sin(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) +
                              12 * self.b ** 3 * math.sin(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) +
                              4 * self.b ** 4 * math.sin(
                        self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3)) * Omega_2 +
                             2 * (4 * (12 + 21 * self.b + 8 * self.b ** 2) * math.sin(self.theta_1) -
                                  4 * self.b * math.sin(self.theta_1 - 2 * self.theta_2) -
                                  4 * self.b * math.sin(3 * self.theta_1 - 2 * self.theta_2) -
                                  6 * self.b * math.sin(2 * self.theta_1 - self.theta_2) -
                                  8 * self.b ** 2 * math.sin(2 * self.theta_1 - self.theta_2) +
                                  26 * self.b * math.sin(self.theta_2) +
                                  48 * self.b ** 2 * math.sin(self.theta_2) +
                                  16 * self.b ** 3 * math.sin(self.theta_2) +
                                  2 * math.sin(self.theta_1 - 2 * self.theta_3) +
                                  2 * math.sin(3 * self.theta_1 - 2 * self.theta_3) +
                                  6 * self.b * math.sin(self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b ** 2 * math.sin(self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b * math.sin(2 * self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b * math.sin(self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b ** 2 * math.sin(3 * self.theta_2 - 2 * self.theta_3) +
                                  20 * math.sin(2 * self.theta_1 - self.theta_3) +
                                  18 * self.b * math.sin(2 * self.theta_1 - self.theta_3) +
                                  6 * self.b ** 2 * math.sin(2 * self.theta_1 - self.theta_3) +
                                  2 * self.b ** 3 * math.sin(2 * self.theta_1 - self.theta_3) +
                                  6 * self.b * math.sin(self.theta_1 - self.theta_2 - self.theta_3) +
                                  3 * self.b ** 2 * math.sin(self.theta_1 - self.theta_2 - self.theta_3) +
                                  6 * self.b * math.sin(3 * self.theta_1 - self.theta_2 - self.theta_3) +
                                  3 * self.b ** 2 * math.sin(3 * self.theta_1 - self.theta_2 - self.theta_3) +
                                  42 * self.b * math.sin(self.theta_1 + self.theta_2 - self.theta_3) +
                                  54 * self.b ** 2 * math.sin(self.theta_1 + self.theta_2 - self.theta_3) +
                                  12 * self.b ** 3 * math.sin(self.theta_1 + self.theta_2 - self.theta_3) +
                                  2 * self.b * math.sin(2 * self.theta_2 - self.theta_3) +
                                  18 * self.b ** 2 * math.sin(2 * self.theta_2 - self.theta_3) +
                                  30 * self.b ** 3 * math.sin(2 * self.theta_2 - self.theta_3) +
                                  8 * self.b ** 4 * math.sin(2 * self.theta_2 - self.theta_3) -
                                  4 * math.sin(self.theta_3) -
                                  6 * self.b * math.sin(self.theta_3) -
                                  18 * self.b ** 2 * math.sin(self.theta_3) -
                                  10 * self.b ** 3 * math.sin(self.theta_3) -
                                  3 * self.b ** 2 * math.sin(self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                  3 * self.b ** 2 * math.sin(3 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                  6 * self.b * math.sin(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) -
                                  6 * self.b ** 2 * math.sin(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) -
                                  6 * self.b ** 3 * math.sin(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                                  6 * self.b * math.sin(self.theta_1 - self.theta_2 + self.theta_3) +
                                  18 * self.b ** 2 * math.sin(self.theta_1 - self.theta_2 + self.theta_3) +
                                  12 * self.b ** 3 * math.sin(self.theta_1 - self.theta_2 + self.theta_3) +
                                  2 * self.b * math.sin(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3)
                                  ) * Omega_3) / (2 * (
                    152 + 216 * self.b + 208 * self.b ** 2 + 190 * self.b ** 3 + 72 * self.b ** 4 + 8 * self.b ** 5 +
                    24 * self.b * (7 + 8 * self.b + 2 * self.b ** 2) * math.cos(self.theta_1 - self.theta_2) +
                    2 * self.b * (-2 + 12 * self.b + 9 * self.b ** 2 + 2 * self.b ** 3) * math.cos(
                2 * (self.theta_1 - self.theta_2)) +
                    24 * self.b * math.cos(self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                    12 * self.b ** 2 * math.cos(self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                    96 * math.cos(self.theta_1 - self.theta_3) +
                    72 * self.b * math.cos(self.theta_1 - self.theta_3) +
                    8 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                    self.b ** 3 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                    24 * self.b * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                    12 * self.b ** 2 * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                    168 * self.b * math.cos(self.theta_2 - self.theta_3) +
                    192 * self.b ** 2 * math.cos(self.theta_2 - self.theta_3) +
                    48 * self.b ** 3 * math.cos(self.theta_2 - self.theta_3) -
                    4 * self.b * math.cos(2 * (self.theta_2 - self.theta_3)) +
                    24 * self.b ** 2 * math.cos(2 * (self.theta_2 - self.theta_3)) +
                    18 * self.b ** 3 * math.cos(2 * (self.theta_2 - self.theta_3)) +
                    4 * self.b ** 4 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                    12 * self.b ** 2 * math.cos(2 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                    24 * self.b * math.cos(self.theta_1 - 2 * self.theta_2 + self.theta_3) -
                    3 * self.b ** 3 * math.cos(2 * (self.theta_1 - 2 * self.theta_2 + self.theta_3)) -
                    12 * self.b ** 2 * math.cos(self.theta_1 - 3 * self.theta_2 + 2 * self.theta_3))))

            velocity_y = (0.5 * (
                    4 * math.cos(3 * self.theta_1 - 2 * self.theta_2) * self.b ** 4 +
                    4 * math.cos(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) * self.b ** 4 +
                    4 * math.cos(self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) * self.b ** 4 +
                    24 * math.cos(2 * self.theta_1 - self.theta_2) * self.b ** 3 -
                    8 * math.cos(self.theta_2) * self.b ** 3 +
                    18 * math.cos(3 * self.theta_1 - 2 * self.theta_2) * self.b ** 3 +
                    48 * math.cos(self.theta_1 + self.theta_2 - self.theta_3) * self.b ** 3 +
                    48 * math.cos(self.theta_1 - self.theta_2 + self.theta_3) * self.b ** 3 +
                    12 * math.cos(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) * self.b ** 3 +
                    3 * math.cos(self.theta_1 - 4 * self.theta_2 + 2 * self.theta_3) * self.b ** 3 -
                    3 * math.cos(3 * self.theta_1 - 4 * self.theta_2 + 2 * self.theta_3) * self.b ** 3 -
                    math.cos(self.theta_1 - 2 * self.theta_3) * self.b ** 3 +
                    math.cos(3 * self.theta_1 - 2 * self.theta_3) * self.b ** 3 +
                    12 * math.cos(self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) * self.b ** 3 +
                    88 * math.cos(2 * self.theta_1 - self.theta_2) * self.b ** 2 -
                    24 * math.cos(self.theta_2) * self.b ** 2 +
                    24 * math.cos(3 * self.theta_1 - 2 * self.theta_2) * self.b ** 2 -
                    12 * math.cos(self.theta_1 - self.theta_2 - self.theta_3) * self.b ** 2 +
                    12 * math.cos(3 * self.theta_1 - self.theta_2 - self.theta_3) * self.b ** 2 +
                    144 * math.cos(self.theta_1 + self.theta_2 - self.theta_3) * self.b ** 2 +
                    144 * math.cos(self.theta_1 - self.theta_2 + self.theta_3) * self.b ** 2 +
                    12 * math.cos(self.theta_1 - 3 * self.theta_2 + self.theta_3) * self.b ** 2 -
                    12 * math.cos(3 * self.theta_1 - 3 * self.theta_2 + self.theta_3) * self.b ** 2 +
                    12 * math.cos(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) * self.b ** 2 -
                    6 * math.cos(2 * self.theta_1 - 3 * self.theta_2 + 2 * self.theta_3) * self.b ** 2 -
                    2 * math.cos(self.theta_2 - 2 * self.theta_3) * self.b ** 2 +
                    6 * math.cos(2 * self.theta_1 + self.theta_2 - 2 * self.theta_3) * self.b ** 2 +
                    12 * math.cos(self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) * self.b ** 2 +
                    2 * math.cos(3 * self.theta_2 - 2 * self.theta_3) * self.b ** 2 +
                    72 * math.cos(2 * self.theta_1 - self.theta_2) * self.b -
                    16 * math.cos(self.theta_2) * self.b -
                    2 * (2 * self.b ** 3 + 9 * self.b ** 2 + 12 * self.b - 4) * math.cos(
                self.theta_1 - 2 * self.theta_2) * self.b -
                    8 * math.cos(3 * self.theta_1 - 2 * self.theta_2) * self.b +
                    36 * math.cos(2 * self.theta_1 - self.theta_3) * self.b -
                    24 * math.cos(self.theta_1 - self.theta_2 - self.theta_3) * self.b +
                    24 * math.cos(3 * self.theta_1 - self.theta_2 - self.theta_3) * self.b +
                    96 * math.cos(self.theta_1 + self.theta_2 - self.theta_3) * self.b +
                    4 * math.cos(2 * self.theta_2 - self.theta_3) * self.b -
                    12 * math.cos(self.theta_3) * self.b +
                    96 * math.cos(self.theta_1 - self.theta_2 + self.theta_3) * self.b -
                    12 * math.cos(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) * self.b +
                    4 * math.cos(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) * self.b +
                    8 * math.cos(2 * self.theta_1 + self.theta_2 - 2 * self.theta_3) * self.b +
                    4 * math.cos(self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) * self.b +
                    8 * (
                            2 * self.b ** 5 + 15 * self.b ** 4 + 33 * self.b ** 3 + 29 * self.b ** 2 + 21 * self.b + 12) * math.cos(
                self.theta_1) +
                    40 * math.cos(2 * self.theta_1 - self.theta_3) -
                    8 * math.cos(self.theta_3) -
                    4 * math.cos(self.theta_1 - 2 * self.theta_3) +
                    4 * math.cos(3 * self.theta_1 - 2 * self.theta_3)
            ) * Omega_2 + Omega_3 * (
                                  8 * self.b ** 4 * math.cos(2 * self.theta_2 - self.theta_3) +
                                  16 * self.b ** 3 * math.cos(self.theta_2) +
                                  2 * self.b ** 3 * math.cos(2 * self.theta_1 - self.theta_3) +
                                  12 * self.b ** 3 * math.cos(self.theta_1 + self.theta_2 - self.theta_3) +
                                  30 * self.b ** 3 * math.cos(2 * self.theta_2 - self.theta_3) -
                                  10 * self.b ** 3 * math.cos(self.theta_3) +
                                  12 * self.b ** 3 * math.cos(self.theta_1 - self.theta_2 + self.theta_3) -
                                  6 * self.b ** 3 * math.cos(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) -
                                  8 * self.b ** 2 * math.cos(2 * self.theta_1 - self.theta_2) +
                                  48 * self.b ** 2 * math.cos(self.theta_2) +
                                  6 * self.b ** 2 * math.cos(2 * self.theta_1 - self.theta_3) -
                                  3 * self.b ** 2 * math.cos(self.theta_1 - self.theta_2 - self.theta_3) +
                                  3 * self.b ** 2 * math.cos(3 * self.theta_1 - self.theta_2 - self.theta_3) +
                                  54 * self.b ** 2 * math.cos(self.theta_1 + self.theta_2 - self.theta_3) +
                                  18 * self.b ** 2 * math.cos(2 * self.theta_2 - self.theta_3) -
                                  18 * self.b ** 2 * math.cos(self.theta_3) +
                                  18 * self.b ** 2 * math.cos(self.theta_1 - self.theta_2 + self.theta_3) -
                                  6 * self.b ** 2 * math.cos(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                                  3 * self.b ** 2 * math.cos(self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                  3 * self.b ** 2 * math.cos(3 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                  2 * self.b ** 2 * math.cos(self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b ** 2 * math.cos(3 * self.theta_2 - 2 * self.theta_3) +
                                  4 * (8 * self.b ** 2 + 21 * self.b + 12) * math.cos(self.theta_1) -
                                  6 * self.b * math.cos(2 * self.theta_1 - self.theta_2) +
                                  26 * self.b * math.cos(self.theta_2) +
                                  4 * self.b * math.cos(self.theta_1 - 2 * self.theta_2) -
                                  4 * self.b * math.cos(3 * self.theta_1 - 2 * self.theta_2) +
                                  18 * self.b * math.cos(2 * self.theta_1 - self.theta_3) -
                                  6 * self.b * math.cos(self.theta_1 - self.theta_2 - self.theta_3) +
                                  6 * self.b * math.cos(3 * self.theta_1 - self.theta_2 - self.theta_3) +
                                  42 * self.b * math.cos(self.theta_1 + self.theta_2 - self.theta_3) +
                                  2 * self.b * math.cos(2 * self.theta_2 - self.theta_3) -
                                  6 * self.b * math.cos(self.theta_3) +
                                  6 * self.b * math.cos(self.theta_1 - self.theta_2 + self.theta_3) -
                                  6 * self.b * math.cos(2 * self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                                  2 * self.b * math.cos(self.theta_1 - 2 * self.theta_2 + 2 * self.theta_3) -
                                  6 * self.b * math.cos(self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b * math.cos(2 * self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                  2 * self.b * math.cos(self.theta_1 + 2 * self.theta_2 - 2 * self.theta_3) +
                                  20 * math.cos(2 * self.theta_1 - self.theta_3) -
                                  4 * math.cos(self.theta_3) -
                                  2 * math.cos(self.theta_1 - 2 * self.theta_3) +
                                  2 * math.cos(3 * self.theta_1 - 2 * self.theta_3)
                          )) / (
                                 8 * self.b ** 5 +
                                 4 * self.b ** 4 * math.cos(2 * (self.theta_2 - self.theta_3)) +
                                 72 * self.b ** 4 +
                                 self.b ** 3 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                                 48 * self.b ** 3 * math.cos(self.theta_2 - self.theta_3) +
                                 18 * self.b ** 3 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                 3 * self.b ** 3 * math.cos(2 * (self.theta_1 - 2 * self.theta_2 + self.theta_3)) +
                                 190 * self.b ** 3 +
                                 12 * self.b ** 2 * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                                 192 * self.b ** 2 * math.cos(self.theta_2 - self.theta_3) +
                                 24 * self.b ** 2 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                 12 * self.b ** 2 * math.cos(2 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                 12 * self.b ** 2 * math.cos(self.theta_1 - 3 * self.theta_2 + 2 * self.theta_3) +
                                 12 * self.b ** 2 * math.cos(self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                 24 * (2 * self.b ** 2 + 8 * self.b + 7) * self.b * math.cos(
                             self.theta_1 - self.theta_2) +
                                 208 * self.b ** 2 +
                                 2 * (2 * self.b ** 3 + 9 * self.b ** 2 + 12 * self.b - 2) * self.b * math.cos(
                             2 * (self.theta_1 - self.theta_2)) +
                                 72 * self.b * math.cos(self.theta_1 - self.theta_3) +
                                 24 * self.b * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                                 168 * self.b * math.cos(self.theta_2 - self.theta_3) -
                                 4 * self.b * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                 24 * self.b * math.cos(self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                                 24 * self.b * math.cos(self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                 216 * self.b +
                                 96 * math.cos(self.theta_1 - self.theta_3) +
                                 8 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                                 152
                         )

            angular_velocity = - ((2 * Omega_3 * (
                    12 * self.b ** 3 * math.cos(self.theta_2 - self.theta_3) +
                    3 * self.b ** 2 * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                    48 * self.b ** 2 * math.cos(self.theta_2 - self.theta_3) -
                    3 * self.b ** 2 * math.cos(2 * self.theta_1 - 3 * self.theta_2 + self.theta_3) +
                    16 * self.b ** 2 -
                    4 * self.b * math.cos(2 * (self.theta_1 - self.theta_2)) +
                    6 * self.b * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                    42 * self.b * math.cos(self.theta_2 - self.theta_3) +
                    2 * self.b * math.cos(2 * (self.theta_2 - self.theta_3)) -
                    6 * self.b * math.cos(self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                    6 * (3 * self.b + 4) * math.cos(self.theta_1 - self.theta_3) +
                    54 * self.b +
                    2 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                    38
            ) + Omega_2 * (
                                           8 * self.b ** 5 +
                                           4 * self.b ** 4 * math.cos(2 * (self.theta_2 - self.theta_3)) +
                                           72 * self.b ** 4 +
                                           self.b ** 3 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                                           48 * self.b ** 3 * math.cos(self.theta_2 - self.theta_3) +
                                           18 * self.b ** 3 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                           3 * self.b ** 3 * math.cos(
                                       2 * (self.theta_1 - 2 * self.theta_2 + self.theta_3)) +
                                           190 * self.b ** 3 +
                                           12 * self.b ** 2 * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                                           192 * self.b ** 2 * math.cos(self.theta_2 - self.theta_3) +
                                           24 * self.b ** 2 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                           12 * self.b ** 2 * math.cos(
                                       2 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                           6 * self.b ** 2 * math.cos(
                                       self.theta_1 - 3 * self.theta_2 + 2 * self.theta_3) +
                                           6 * self.b ** 2 * math.cos(self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                           12 * (2 * self.b ** 2 + 8 * self.b + 7) * self.b * math.cos(
                                       self.theta_1 - self.theta_2) +
                                           176 * self.b ** 2 +
                                           2 * (
                                                   2 * self.b ** 3 + 9 * self.b ** 2 + 12 * self.b - 4) * self.b * math.cos(
                                       2 * (self.theta_1 - self.theta_2)) +
                                           36 * self.b * math.cos(self.theta_1 - self.theta_3) +
                                           24 * self.b * math.cos(2 * self.theta_1 - self.theta_2 - self.theta_3) +
                                           168 * self.b * math.cos(self.theta_2 - self.theta_3) +
                                           4 * self.b * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                           12 * self.b * math.cos(self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                                           12 * self.b * math.cos(self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                           108 * self.b +
                                           48 * math.cos(self.theta_1 - self.theta_3) +
                                           4 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                                           76)) / (8 * self.b ** 5 +
                                                   4 * self.b ** 4 * math.cos(2 * (self.theta_2 - self.theta_3)) +
                                                   72 * self.b ** 4 +
                                                   self.b ** 3 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                                                   48 * self.b ** 3 * math.cos(self.theta_2 - self.theta_3) +
                                                   18 * self.b ** 3 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                                   3 * self.b ** 3 * math.cos(
                        2 * (self.theta_1 - 2 * self.theta_2 + self.theta_3)) +
                                                   190 * self.b ** 3 +
                                                   12 * self.b ** 2 * math.cos(
                        2 * self.theta_1 - self.theta_2 - self.theta_3) +
                                                   192 * self.b ** 2 * math.cos(self.theta_2 - self.theta_3) +
                                                   24 * self.b ** 2 * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                                   12 * self.b ** 2 * math.cos(
                        2 * self.theta_1 - 3 * self.theta_2 + self.theta_3) -
                                                   12 * self.b ** 2 * math.cos(
                        self.theta_1 - 3 * self.theta_2 + 2 * self.theta_3) +
                                                   12 * self.b ** 2 * math.cos(
                        self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                                   24 * (2 * self.b ** 2 + 8 * self.b + 7) * self.b * math.cos(
                        self.theta_1 - self.theta_2) +
                                                   208 * self.b ** 2 +
                                                   2 * (
                                                           2 * self.b ** 3 + 9 * self.b ** 2 + 12 * self.b - 2) * self.b * math.cos(
                        2 * (self.theta_1 - self.theta_2)) +
                                                   72 * self.b * math.cos(self.theta_1 - self.theta_3) +
                                                   24 * self.b * math.cos(
                        2 * self.theta_1 - self.theta_2 - self.theta_3) +
                                                   168 * self.b * math.cos(self.theta_2 - self.theta_3) -
                                                   4 * self.b * math.cos(2 * (self.theta_2 - self.theta_3)) -
                                                   24 * self.b * math.cos(
                        self.theta_1 - 2 * self.theta_2 + self.theta_3) +
                                                   24 * self.b * math.cos(
                        self.theta_1 + self.theta_2 - 2 * self.theta_3) +
                                                   216 * self.b +
                                                   96 * math.cos(self.theta_1 - self.theta_3) +
                                                   8 * math.cos(2 * (self.theta_1 - self.theta_3)) +
                                                   152))


            self.xy_1[0] += velocity_x * dt_num
            self.xy_1[1] += velocity_y * dt_num
            self.theta_1 += angular_velocity * dt_num

            self.xy_2[0] = self.xy_1[0] + (self.L) * math.cos(self.theta_1)
            self.xy_2[1] = self.xy_1[1] + (self.L) * math.sin(self.theta_1)
            self.xy_3[0] = self.xy_2[0] + self.b * (self.L) * math.cos(self.theta_2)
            self.xy_3[1] = self.xy_2[1] + self.b * (self.L) * math.sin(self.theta_2)
            self.xy_4[0] = self.xy_3[0] + (self.L) * math.cos(self.theta_3)
            self.xy_4[1] = self.xy_3[1] + (self.L) * math.sin(self.theta_3)

        total[0] = self.xy_1[0]
        total[1] = self.xy_1[1]
        total[2] = self.xy_2[0]
        total[3] = self.xy_2[1]
        total[4] = self.xy_3[0]
        total[5] = self.xy_3[1]
        total[6] = self.xy_4[0]
        total[7] = self.xy_4[1]
        total[8] = self.theta_1
        total[9] = self.Phi_1
        total[10] = self.Phi_2
        total[11] = u[0]
        total[12] = u[1]

        self.trajectory.append(total)

    def get_trajectory(self):
        return self.trajectory

    def get_DOFs(self):
        return self.Phi_1, self.Phi_2

    def get_lims(self):
        return self.low_lim, self.up_lim

    def sum_last_elements(self, lst):
        if len(lst) > 4:
            return sum(lst[-4:])
        else:
            return sum(lst)

    def get_history(self, agent_action, agent_action_output):
        self.prev_pos = self.pos
        self.observation_RK4(agent_action, self.dt_ratio)
        x_position = (self.xy_1[0] + self.xy_2[0] + self.xy_3[0] + self.xy_2[0] + self.xy_3[0] + self.xy_4[0]) / 6
        y_position = (self.xy_1[1] + self.xy_2[1] + self.xy_3[1] + self.xy_2[0] + self.xy_3[0] + self.xy_4[1]) / 6
        self.pos = np.array([x_position, y_position])
        self.pos_diff = self.pos - self.prev_pos

        if abs(self.pos_diff[0]) < self.range:
            stuck = True
        else:
            stuck = False

        self.pos_diff_history.append(self.pos_diff[0])
        self.sum_elements = self.sum_last_elements(self.pos_diff_history)

        results = {
            "last_elements": round(self.sum_elements, 3),
            "last_position": round(self.prev_pos[0], 3),
            "position": round(self.pos[0], 3),
            "stuck": stuck
        }

        history_entry = {
            "actions": agent_action_output,
            "states": {
                "self.DOF1_last": round(self.Phi_1_last, 3),
                "self.DOF2_last": round(self.Phi_2_last, 3),
                "self.DOF1": round(self.Phi_1, 3),
                "self.DOF2": round(self.Phi_2, 3),
            },
            "results": results
        }

        return self.Phi_1, self.Phi_2, history_entry

class ThreeSphere:
    def __init__(self):
        self.a = None
        self.L1 = None
        self.L2 = None
        self.L1_last = None
        self.L2_last = None
        self.dt = None
        self.dt_ratio = None
        self.pos = None
        self.pos_diff = None
        self.prev_pos = None
        self.range = None
        self.trajectory = None
        self.low_lim = None
        self.up_lim = None
        self.sum_elements = None
        self.pos_diff_history = None

    def reset_world(self):
        self.a = 1
        self.L1 = 10.0
        self.L2 = 6.0
        self.L1_last = 10.0
        self.L2_last = 6.0
        self.dt = 4
        self.pos = 0
        self.prev_pos = 0
        self.pos_diff = self.pos - self.prev_pos
        self.range = 1e-6
        self.low_lim = 6.0
        self.up_lim = 10.0
        self.dt_ratio = 300
        self.sum_elements = 0
        self.trajectory = []
        self.pos_diff_history = []

    def observation_RK4(self, u, dt_ratio):
        def rk4_step(y, u, dt_num, dydt_func):
            k1 = dt_num * dydt_func(y, u)
            k2 = dt_num * dydt_func(y + 0.5 * k1, u)
            k3 = dt_num * dydt_func(y + 0.5 * k2, u)
            k4 = dt_num * dydt_func(y + k3, u)
            return y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        def dydt_length(L, Vel):
            if L + self.range >= self.up_lim and Vel >= 0:
                Vel = 0
            elif L - self.range <= self.low_lim and Vel <= 0:
                Vel = 0
            return Vel

        self.L1_last = self.L1
        self.L2_last = self.L2
        dt_num = self.dt / dt_ratio
        steps = int(self.dt / dt_num)
        Vel_L1 = u[0]
        Vel_L2 = u[1]
        total = np.zeros(5)
        for _ in range(steps+1):
            self.L1 = rk4_step(self.L1, Vel_L1, dt_num, dydt_length)
            self.L2 = rk4_step(self.L2, Vel_L2, dt_num, dydt_length)
            Vel_L1 = dydt_length(self.L1, Vel_L1)
            Vel_L2 = dydt_length(self.L2, Vel_L2)
            self.L1 = max(self.low_lim, min(self.L1, self.up_lim))
            self.L2 = max(self.low_lim, min(self.L2, self.up_lim))

            velocity = (self.a / 6) * (
                    ((Vel_L2 - Vel_L1) / (self.L1 + self.L2)) + 2 * (Vel_L1 / self.L2 - Vel_L2 / self.L1))

            self.pos += velocity * dt_num

        total[0] = self.L1
        total[1] = self.L2
        total[2] = self.pos
        total[3] = u[0]
        total[4] = u[1]

        self.trajectory.append(total)

    def get_trajectory(self):
        return self.trajectory

    def get_DOFs(self):
        return self.L1, self.L2

    def get_lims(self):
        return self.low_lim, self.up_lim

    def sum_last_elements(self, lst):
        if len(lst) > 4:
            return round(sum(lst[-4:]), 3)
        else:
            return round(sum(lst), 3)

    def get_history(self, agent_action, agent_action_output):
        self.prev_pos = self.pos
        self.observation_RK4(agent_action, self.dt_ratio)
        self.pos_diff = self.pos - self.prev_pos

        if abs(self.pos_diff) < self.range:
            stuck = True
        else:
            stuck = False

        self.pos_diff_history.append(self.pos_diff)
        self.sum_elements = self.sum_last_elements(self.pos_diff_history)

        results = {
            "last_elements": round(self.sum_elements, 3),
            "last_position": round(self.prev_pos, 3),
            "position": round(self.pos, 3),
            "stuck": stuck
        }

        history_entry = {
            "actions": agent_action_output,
            "states": {
                "self.DOF1_last": round(self.L1_last, 0),
                "self.DOF2_last": round(self.L2_last, 0),
                "self.DOF1": round(self.L1, 0),
                "self.DOF2": round(self.L2, 0),
            },
            "results": results
        }

        return self.L1, self.L2, history_entry

class LLMInteractionHandler:
    def __init__(self, swimmer):
        self.history_length = None
        self.turns = None
        self.temperature = None
        self.max_tokens = None
        self.DOF1 = None
        self.DOF2 = None
        self.low_lim = None
        self.up_lim = None
        self.swimmer = swimmer

    def reset_world(self):
        self.history_length = 3
        self.turns = 200
        self.temperature = 0
        self.max_tokens = 10
        self.swimmer.reset_world()
        self.DOF1, self.DOF2 = self.swimmer.get_DOFs()
        self.low_lim, self.up_lim = self.swimmer.get_lims()

    def parse_response(self, response_text):
        """Extract the action numbers from the response."""
        actions = [None, None]
        found = False

        pattern = r'\[(-?\d),?\s*(-?\d)\]'
        match = re.search(pattern, response_text)

        if match:
            left_hinge_action, right_hinge_action = map(int, match.groups())
            actions = [left_hinge_action, right_hinge_action]
            found = True

        return actions, found

    def get_valid_response(self, setup_message, agent_prompt):
        for _ in range(5):
            response_text = self.chat_with_gpt(setup_message, agent_prompt)
            filename = "response_output.txt"
            with open(filename, 'w') as file:
                file.write(response_text)
            actions_output, found = self.parse_response(response_text)
            actions = self.transform_vector(actions_output)
            if found:
                return actions_output, actions
        print("Warning: Default action taken after 5 unsuccessful attempts to get a valid action.")
        return [1, 0], [0, 0]

    def transform_vector(self, vector):
        a, b = vector
        if a == 1:
            return [b, 0]
        else:
            return [0, b]

    @retry(stop_max_attempt_number=6, wait_exponential_multiplier=1000, wait_exponential_max=60000)
    def chat_with_gpt(self, setup_message, agent_prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=setup_message + [{"role": "user", "content": agent_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message['content'].strip()

        except (error.OpenAIError) as e:
            print(f"Attempt failed due to: {str(e)}")
            raise e

    def simulate_interaction(self):
        self.reset_world()
        history = []
        for t in range(1, self.turns+1):
            setup_message = [
                {"role": "system",
                 "content": " "},
            ]

            n = self.history_length
            num_turns_to_include = min(n, len(history))

            history_str = ", ".join(
                [
                    f"at step {i + 1 + len(history) - num_turns_to_include}:  "
                    f"D1: {entry['states']['self.DOF1_last']}, D2: {entry['states']['self.DOF2_last']}, "
                    f"[changed DOF, ROC]: {entry['actions']}, "
                    f"after performing this action, D1: {entry['states']['self.DOF1']}, D2: {entry['states']['self.DOF2']}, "
                    f"swimmer moves from position {entry['results']['last_position']} to position {entry['results']['position']}, "
                    f"stuck: {entry['results']['stuck']}"
                    for i, entry in enumerate(history[-num_turns_to_include:])]
            )

            agent_prompt = (
                f"Determine a sequence of rate of change (ROC) in two degrees of freedom (DOF), D1 and D2, such that the swimmer achieves fastest long-term movement in positive x direction. "
                f"Only one DOF can change simultaneously, and all DOFs must range in [{round(self.low_lim, 3)}, {round(self.up_lim, 3)}]. "
                f"History: {history_str}. "
                f"Consider the impact of the output on future movement."
                f"Firstly, respond by indicating which DOF to change, choosing from 1 or 2; secondly, respond by specifying ROC for this DOF, choosing from -1, 0, 1; output format is [changed DOF, ROC]."
            )

            agent_action_output, agent_action = self.get_valid_response(setup_message, agent_prompt)

            self.DOF1, self.DOF2, history_entry = self.swimmer.get_history(agent_action, agent_action_output)
            history_entry['results']['last_position'] = round((history_entry['results']['last_position'] - (-2.5)) * (1000 - 0), 3)
            history_entry['results']['position'] = round((history_entry['results']['position'] - (-2.5)) * (1000 - 0), 3)
            last_four_elements = history_entry['results']['last_elements']

            trajectory = self.swimmer.get_trajectory()
            history.append(history_entry)

            # negative
            # if t > 5 and len(history) > 5 and last_four_elements > -0.01:
            #     print("limit:", last_four_elements)
            #     history = []
            # positive
            if t > 5 and len(history) > 5 and last_four_elements < 0.01:
                print("limit:", last_four_elements)
                history = []

            print(f"Step {t}:")
            print(f"Agent chose action {agent_action}")
            print(f"Agent's position: {self.swimmer.pos}")

            # Save data
            directory_name = f'sim_data/test'
            filename = f'{directory_name}/trajectory_try_1.txt'
            save_and_notify(filename, trajectory)

if __name__ == '__main__':
    # ThreeSphere() for NG's swimmer; ThreeLink() for purcell swimmer
    swimmer = ThreeLink()
    llm_learning = LLMInteractionHandler(swimmer)
    llm_learning.simulate_interaction()

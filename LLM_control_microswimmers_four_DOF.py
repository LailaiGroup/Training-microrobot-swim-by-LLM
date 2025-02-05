import openai
import math
import numpy as np
import os

from openai import OpenAIError
from retrying import retry
import re
import random

# Replace 'YOUR_OPENAI_API_KEY' with your actual API key
openai.api_key = ''

def save_and_notify(filename, data):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if os.path.exists(filename):
        pass
    np.savetxt(filename, data, fmt='%f')

class FourLink:
    def __init__(self):
        self.gamma = None
        self.theta_1 = None
        self.theta_2 = None
        self.theta_3 = None
        self.theta_4 = None
        self.Phi_1 = None
        self.Phi_2 = None
        self.Phi_3 = None
        self.Phi_1_last = None
        self.Phi_2_last = None
        self.Phi_3_last = None
        self.L = None
        self.dt = None
        self.dt_ratio = None
        self.xy_1 = None
        self.xy_2 = None
        self.xy_3 = None
        self.xy_4 = None
        self.xy_5 = None
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
        self.gamma = (2 * math.pi) / 9
        self.theta_1 = -math.pi / 3
        self.Phi_1 = self.gamma
        self.Phi_2 = self.gamma
        self.Phi_3 = self.gamma
        self.Phi_1_last = self.gamma
        self.Phi_2_last = self.gamma
        self.Phi_3_last = self.gamma
        self.theta_2 = self.Phi_1 + self.theta_1
        self.theta_3 = self.Phi_2 + self.theta_2
        self.theta_4 = self.Phi_3 + self.theta_3
        self.L = 1
        self.dt = 2 * np.abs(self.gamma)
        self.dt_ratio = 300
        self.range = 1e-6
        self.low_lim = -np.abs(self.gamma)
        self.up_lim = np.abs(self.gamma)
        self.xy_1 = np.array([-math.cos(self.theta_1) - 1 / 2, -math.sin(self.theta_1)])
        self.xy_2 = np.array([0, 0.0])
        self.xy_3 = np.array([0, 0.0])
        self.xy_4 = np.array([0, 0.0])
        self.xy_5 = np.array([0, 0.0])
        self.xy_2[0] = self.xy_1[0] + self.L * math.cos(self.theta_1)
        self.xy_2[1] = self.xy_1[1] + self.L * math.sin(self.theta_1)
        self.xy_3[0] = self.xy_2[0] + self.L * math.cos(self.theta_2)
        self.xy_3[1] = self.xy_2[1] + self.L * math.sin(self.theta_2)
        self.xy_4[0] = self.xy_3[0] + self.L * math.cos(self.theta_3)
        self.xy_4[1] = self.xy_3[1] + self.L * math.sin(self.theta_3)
        self.xy_5[0] = self.xy_4[0] + self.L * math.cos(self.theta_4)
        self.xy_5[1] = self.xy_4[1] + self.L * math.sin(self.theta_4)
        self.x_position = (self.xy_1[0] + self.xy_2[0] + self.xy_3[0] + self.xy_4[0] + self.xy_5[0]) / 5
        self.y_position = (self.xy_1[1] + self.xy_2[1] + self.xy_3[1] + self.xy_4[1] + self.xy_5[1]) / 5
        self.pos = np.array([self.x_position, self.y_position])
        self.prev_pos = self.pos
        self.pos_diff = self.pos - self.prev_pos
        self.trajectory = []
        self.pos_diff_history = []

    def observation_RK4(self, agent_action, dt_ratio):     
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

        Omega_2 = agent_action[0]
        Omega_3 = agent_action[1]
        Omega_4 = agent_action[2]
        total = np.zeros(17)
        self.Phi_1_last = self.Phi_1
        self.Phi_2_last = self.Phi_2
        self.Phi_3_last = self.Phi_3
        for _ in range(steps+1):
            self.Phi_1 = rk4_step(self.Phi_1, Omega_2, dt_num, dydt_angular_Phi)
            self.Phi_2 = rk4_step(self.Phi_2, Omega_3, dt_num, dydt_angular_Phi)
            self.Phi_3 = rk4_step(self.Phi_3, Omega_4, dt_num, dydt_angular_Phi)

            if self.Phi_1 + self.range >= self.up_lim:
                self.Phi_1 = self.up_lim
            elif self.Phi_1 - self.range <= self.low_lim:
                self.Phi_1 = self.low_lim

            if self.Phi_2 + self.range >= self.up_lim:
                self.Phi_2 = self.up_lim
            elif self.Phi_2 - self.range <= self.low_lim:
                self.Phi_2 = self.low_lim

            if self.Phi_3 + self.range >= self.up_lim:
                self.Phi_3 = self.up_lim
            elif self.Phi_3 - self.range <= self.low_lim:
                self.Phi_3 = self.low_lim

            self.theta_2 = self.theta_1 + self.Phi_1
            self.theta_3 = self.theta_2 + self.Phi_2
            self.theta_4 = self.theta_3 + self.Phi_3
            Omega_2 = dydt_angular_Phi(self.Phi_1, Omega_2)
            Omega_3 = dydt_angular_Phi(self.Phi_2, Omega_3)
            Omega_4 = dydt_angular_Phi(self.Phi_3, Omega_4)

            w = [Omega_2, Omega_3, Omega_4]
            x = [self.theta_1, self.Phi_1, self.Phi_2, self.Phi_3]

            velocity = self.cal_velocity_center(x, w)

            velocity_x = velocity[0]
            velocity_y = velocity[1]
            angular_velocity = velocity[2]

            self.xy_1[0] += velocity_x * dt_num
            self.xy_1[1] += velocity_y * dt_num
            self.theta_1 += angular_velocity * dt_num

            self.xy_2[0] = self.xy_1[0] + (self.L) * math.cos(self.theta_1)
            self.xy_2[1] = self.xy_1[1] + (self.L) * math.sin(self.theta_1)
            self.xy_3[0] = self.xy_2[0] + (self.L) * math.cos(self.theta_2)
            self.xy_3[1] = self.xy_2[1] + (self.L) * math.sin(self.theta_2)
            self.xy_4[0] = self.xy_3[0] + (self.L) * math.cos(self.theta_3)
            self.xy_4[1] = self.xy_3[1] + (self.L) * math.sin(self.theta_3)
            self.xy_5[0] = self.xy_4[0] + (self.L) * math.cos(self.theta_4)
            self.xy_5[1] = self.xy_4[1] + (self.L) * math.sin(self.theta_4)

        total[0] = self.xy_1[0]
        total[1] = self.xy_1[1]
        total[2] = self.xy_2[0]
        total[3] = self.xy_2[1]
        total[4] = self.xy_3[0]
        total[5] = self.xy_3[1]
        total[6] = self.xy_4[0]
        total[7] = self.xy_4[1]
        total[8] = self.xy_5[0]
        total[9] = self.xy_5[1]
        total[10] = self.theta_1
        total[11] = self.Phi_1
        total[12] = self.Phi_2
        total[13] = self.Phi_3
        total[14] = agent_action[0]
        total[15] = agent_action[1]
        total[16] = agent_action[2]

        self.trajectory.append(total)

    def cal_velocity_center(slef, x, w):

        w = np.squeeze(w)
        x = np.squeeze(x)

        A = np.zeros((3, 3))
        b = np.zeros(3)
        A[0, 0] = -0.25 * (
                    -12 + math.cos(2 * x[0]) + math.cos(2 * (x[0] + x[1])) + math.cos(2 * (x[0] + x[1] + x[2]))
                    + math.cos(2 * (x[0] + x[1] + x[2] + x[3])))

        A[0, 1] = -0.25 * (math.sin(2 * x[0]) + math.sin(2 * (x[0] + x[1])) + math.sin(2 * (x[0] + x[1] + x[2]))
                           + math.sin(2 * (x[0] + x[1] + x[2] + x[3])))

        A[0, 2] = -0.25 * (11 * math.sin(x[0]) + 8 * math.sin(x[0] + x[1]) + math.sin(x[0] + 2 * x[1])
                           + 5 * math.sin(x[0] + x[1] + x[2]) + math.sin(x[0] + x[1] + 2 * x[2])
                           + math.sin(x[0] + 2 * x[1] + 2 * x[2]) + 2 * math.sin(x[0] + x[1] + x[2] + x[3])
                           + math.sin(x[0] + x[1] + x[2] + 2 * x[3]) + math.sin(x[0] + x[1] + 2 * x[2] + 2 * x[3])
                           + math.sin(x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3]))

        b[0] = (0.25 * w[0] * (8 * math.sin(x[0] + x[1]) + 5 * math.sin(x[0] + x[1] + x[2])
                               + math.sin(x[0] + x[1] + 2 * x[2]) + 2 * math.sin(x[0] + x[1] + x[2] + x[3])
                               + math.sin(x[0] + x[1] + x[2] + 2 * x[3]) + math.sin(
                    x[0] + x[1] + 2 * x[2] + 2 * x[3]))
                + 0.25 * w[1] * (5 * math.sin(x[0] + x[1] + x[2]) + 2 * math.sin(x[0] + x[1] + x[2] + x[3])
                                 + math.sin(x[0] + x[1] + x[2] + 2 * x[3]))
                + 0.5 * w[2] * math.sin(x[0] + x[1] + x[2] + x[3]))

        A[1, 0] = 0.25 * (
                    -math.sin(2 * x[0]) - math.sin(2 * x[0] + 2 * x[1]) - math.sin(2 * x[0] + 2 * x[1] + 2 * x[2])
                    - math.sin(2 * x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3]))

        A[1, 1] = 0.25 * (12 + math.cos(2 * x[0]) + math.cos(2 * (x[0] + x[1])) + math.cos(2 * (x[0] + x[1] + x[2]))
                          + math.cos(2 * (x[0] + x[1] + x[2] + x[3])))

        A[1, 2] = 0.25 * (11 * math.cos(x[0]) + 8 * math.cos(x[0] + x[1]) + math.cos(x[0] + 2 * x[1])
                          + 5 * math.cos(x[0] + x[1] + x[2]) + math.cos(x[0] + x[1] + 2 * x[2])
                          + math.cos(x[0] + 2 * x[1] + 2 * x[2]) + 2 * math.cos(x[0] + x[1] + x[2] + x[3])
                          + math.cos(x[0] + x[1] + x[2] + 2 * x[3]) + math.cos(x[0] + x[1] + 2 * x[2] + 2 * x[3])
                          + math.cos(x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3]))

        b[1] = (-0.25 * w[0] * (
                8 * math.cos(x[0] + x[1]) + 5 * math.cos(x[0] + x[1] + x[2]) + math.cos(x[0] + x[1] + 2 * x[2])
                + 2 * math.cos(x[0] + x[1] + x[2] + x[3]) + math.cos(x[0] + x[1] + x[2] + 2 * x[3])
                + math.cos(x[0] + x[1] + 2 * x[2] + 2 * x[3]))
                - 0.25 * w[1] * (5 * math.cos(x[0] + x[1] + x[2]) + 2 * math.cos(x[0] + x[1] + x[2] + x[3])
                                 + math.cos(x[0] + x[1] + x[2] + 2 * x[3])) - 0.5 * w[2] * math.cos(
                    x[0] + x[1] + x[2] + x[3]))

        A[2, 0] = (1 / 12) * (-33 * math.sin(x[0]) - 24 * math.sin(x[0] + x[1]) - 3 * math.sin(x[0] + 2 * x[1])
                              - 15 * math.sin(x[0] + x[1] + x[2]) - 3 * math.sin(x[0] + x[1] + 2 * x[2])
                              - 3 * math.sin(x[0] + 2 * x[1] + 2 * x[2]) - 6 * math.sin(x[0] + x[1] + x[2] + x[3])
                              - 3 * math.sin(x[0] + x[1] + x[2] + 2 * x[3]) - 3 * math.sin(
                    x[0] + x[1] + 2 * x[2] + 2 * x[3])
                              - 3 * math.sin(x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3]))

        A[2, 1] = (1 / 12) * (33 * math.cos(x[0]) + 24 * math.cos(x[0] + x[1]) + 3 * math.cos(x[0] + 2 * x[1])
                              + 15 * math.cos(x[0] + x[1] + x[2]) + 3 * math.cos(x[0] + x[1] + 2 * x[2])
                              + 3 * math.cos(x[0] + 2 * x[1] + 2 * x[2]) + 6 * math.cos(x[0] + x[1] + x[2] + x[3])
                              + 3 * math.cos(x[0] + x[1] + x[2] + 2 * x[3]) + 3 * math.cos(
                    x[0] + x[1] + 2 * x[2] + 2 * x[3])
                              + 3 * math.cos(x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3]))

        A[2, 2] = (1 / 12) * (34 + 12 * math.cos(x[1]) + 3 * (3 + math.cos(2 * x[1])) + 12 * math.cos(x[2])
                              + 3 * (3 + math.cos(2 * x[2])) + 12 * math.cos(x[1] + x[2])
                              + 3 * (3 + math.cos(2 * (x[1] + x[2]))) + 6 * (
                                          3 * math.cos(x[1]) + math.cos(x[1] + 2 * x[2]))
                              + 12 * math.cos(x[3]) + 3 * math.cos(2 * x[3]) + 12 * math.cos(x[2] + x[3])
                              + 3 * (3 + math.cos(2 * (x[2] + x[3]))) + 12 * math.cos(x[1] + x[2] + x[3])
                              + 3 * math.cos(2 * (x[1] + x[2] + x[3])) + 6 * (
                                      3 * math.cos(x[2]) + math.cos(x[2] + 2 * x[3]))
                              + 6 * (3 * math.cos(x[1] + x[2]) + math.cos(x[1] + x[2] + 2 * x[3]))
                              + 6 * (3 * math.cos(x[1]) + math.cos(x[1] + 2 * (x[2] + x[3]))))

        b[2] = (-(1 / 12) * w[0] * (21 + 6 * math.cos(x[1]) + 12 * math.cos(x[2]) + 3 * (3 + math.cos(2 * x[2]))
                                    + 6 * math.cos(x[1] + x[2]) + 3 * (
                                                3 * math.cos(x[1]) + math.cos(x[1] + 2 * x[2]))
                                    + 12 * math.cos(x[3]) + 3 * math.cos(2 * x[3]) + 12 * math.cos(x[2] + x[3])
                                    + 3 * (3 + math.cos(2 * (x[2] + x[3]))) + 6 * math.cos(x[1] + x[2] + x[3])
                                    + 6 * (3 * math.cos(x[2]) + math.cos(x[2] + 2 * x[3]))
                                    + 3 * (3 * math.cos(x[1] + x[2]) + math.cos(x[1] + x[2] + 2 * x[3]))
                                    + 3 * (3 * math.cos(x[1]) + math.cos(x[1] + 2 * (x[2] + x[3]))))
                - (1 / 12) * w[1] * (17 + 6 * math.cos(x[2]) + 6 * math.cos(x[1] + x[2]) + 12 * math.cos(x[3])
                                     + 3 * math.cos(2 * x[3]) + 6 * math.cos(x[2] + x[3]) + 6 * math.cos(x[1] + x[2] + x[3])
                                     + 3 * (3 * math.cos(x[2]) + math.cos(x[2] + 2 * x[3]))
                                     + 3 * (3 * math.cos(x[1] + x[2]) + math.cos(x[1] + x[2] + 2 * x[3])))
                - (1 / 12) * w[2] * (4 + 6 * math.cos(x[3]) + 6 * math.cos(x[2] + x[3]) + 6 * math.cos(x[1] + x[2] + x[3])))

        AA = np.linalg.inv(A)
        velo = np.dot(AA, b)

        return velo

    def get_trajectory(self):
        return self.trajectory

    def get_DOFs(self):
        return self.Phi_1, self.Phi_2, self.Phi_3

    def get_lims(self):
        return self.low_lim, self.up_lim

    def sum_last_elements(self, lst):
        if len(lst) > 40:
            return sum(lst[-40:])
        else:
            return sum(lst)

    def get_history(self, agent_action_output, agent_action):
        self.prev_pos = self.pos
        self.observation_RK4(agent_action, self.dt_ratio)               
        x_position = (self.xy_1[0] + 2 * self.xy_2[0] + 3 * self.xy_3[0] + 2 * self.xy_4[0] + self.xy_5[0]) / 9
        y_position = (self.xy_1[1] + 2 * self.xy_2[1] + 3 * self.xy_3[1] + 2 * self.xy_4[1] + self.xy_5[1]) / 9
        self.pos = np.array([x_position, y_position])
        self.pos_diff = self.pos - self.prev_pos

        if abs(self.pos_diff[0]) < self.range:
            stuck = True
        else:
            stuck = False

        self.pos_diff_history.append(self.pos_diff[0])
        self.sum_elements = self.sum_last_elements(self.pos_diff_history)

        results = {
            "last_position": round(self.prev_pos[0], 3),
            "position": round(self.pos[0], 3),
            "stuck": stuck
        }

        history_entry = {
            "actions": agent_action_output,
            "states": {
                "self.DOF1_last": round(self.Phi_1_last, 3),
                "self.DOF2_last": round(self.Phi_2_last, 3),
                "self.DOF3_last": round(self.Phi_3_last, 3),
                "self.DOF1": round(self.Phi_1, 3),
                "self.DOF2": round(self.Phi_2, 3),
                "self.DOF3": round(self.Phi_3, 3),
            },
            "results": results
        }

        return self.Phi_1, self.Phi_2, self.Phi_3, history_entry, self.sum_elements

class FourSphere:
    def __init__(self):
        self.a = None
        self.L1 = None
        self.L2 = None
        self.L3 = None
        self.L1_last = None
        self.L2_last = None
        self.L3_last = None
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
        self.L1 = 6.0
        self.L2 = 10.0
        self.L3 = 10.0
        self.L1_last = 6.0
        self.L2_last = 10.0
        self.L3_last = 10.0
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
        self.L3_last = self.L3
        dt_num = self.dt / dt_ratio
        steps = int(self.dt / dt_num)

        Vel_L1 = u[0]
        Vel_L2 = u[1]
        Vel_L3 = u[2]
        total = np.zeros(7)
        for _ in range(steps+1):
            self.L1 = rk4_step(self.L1, Vel_L1, dt_num, dydt_length)
            self.L2 = rk4_step(self.L2, Vel_L2, dt_num, dydt_length)
            self.L3 = rk4_step(self.L3, Vel_L3, dt_num, dydt_length)
            Vel_L1 = dydt_length(self.L1, Vel_L1)
            Vel_L2 = dydt_length(self.L2, Vel_L2)
            Vel_L3 = dydt_length(self.L3, Vel_L3)
            self.L1 = max(self.low_lim, min(self.L1, self.up_lim))
            self.L2 = max(self.low_lim, min(self.L2, self.up_lim))
            self.L3 = max(self.low_lim, min(self.L3, self.up_lim))

            e1 = -0.75 - 0.1875 * ((self.a / self.L1) - (self.a / self.L2) - (self.a / self.L3) + (self.a / (self.L1 + self.L2)) -
                        (self.a / (self.L2 + self.L3)) + (self.a / (self.L1 + self.L2 + self.L3)))
            e2 = -0.5 - 0.375 * ((self.a / self.L1) - (self.a / self.L3))
            e3 = -0.25 - 0.1875 * ((self.a / self.L1) + (self.a / self.L2) - (self.a / self.L3) + (self.a / (self.L1 + self.L2)) -
                        (self.a / (self.L2 + self.L3)) - (self.a / (self.L1 + self.L2 + self.L3)))

            velocity_1 = e1 * Vel_L1 + e2 * Vel_L2 + e3 * Vel_L3

            velocity = (velocity_1 + (velocity_1 + Vel_L1) + (velocity_1 + Vel_L1 + Vel_L2) + (velocity_1 + Vel_L1 + Vel_L2 + Vel_L3)) / 4

            self.pos += velocity * dt_num

        total[0] = self.L1
        total[1] = self.L2
        total[2] = self.L3
        total[3] = self.pos
        total[4] = u[0]
        total[5] = u[1]
        total[6] = u[2]

        self.trajectory.append(total)

    def get_trajectory(self):
        return self.trajectory

    def get_DOFs(self):
        return self.L1, self.L2, self.L3

    def get_lims(self):
        return self.low_lim, self.up_lim

    def sum_last_elements(self, lst):
        if len(lst) > 5:
            return round(sum(lst[-5:]), 3)
        else:
            return round(sum(lst), 3)

    def get_history(self, agent_action_output, agent_action):
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
                "self.DOF3_last": round(self.L3_last, 0),
                "self.DOF1": round(self.L1, 0),
                "self.DOF2": round(self.L2, 0),
                "self.DOF3": round(self.L3, 0),
            },
            "results": results
        }

        return self.L1, self.L2, self.L3, history_entry, self.sum_elements

class LLMInteractionHandler:
    def __init__(self, swimmer):
        self.history_length = None
        self.turns = None
        self.temperature = None
        self.max_tokens = None
        self.DOFs = None
        self.DOF1 = None
        self.DOF2 = None
        self.DOF3 = None
        self.low_lim = None
        self.up_lim = None
        self.swimmer = swimmer

    def reset_world(self):
        self.history_length = 20
        self.turns = 200
        self.temperature = 0
        self.max_tokens = 20
        self.swimmer.   reset_world()
        self.DOF1, self.DOF2, self.DOF3 = self.swimmer.get_DOFs()
        self.low_lim, self.up_lim = self.swimmer.get_lims()

    def get_valid_response(self, setup_message, agent_prompt):
        for _ in range(5):
            response_text = self.chat_with_gpt(setup_message, agent_prompt)
            filename = "response_output.txt"
            with open(filename, 'w') as file:
                file.write(response_text)
            actions_output, found = self.parse_response(response_text)
            if found:
                all_joint_actions = self.transform_vector(actions_output)
                return actions_output, all_joint_actions
        print("Warning: Default action taken after 5 unsuccessful attempts to get a valid action.")
        return [1, 0], [0, 0, 0]

    def parse_response(self, response_text):
        """Extract the action numbers from the response."""
        actions = [None, None]
        found = False

        pattern = r'\[(-?\d),?\s*(-?\d)\]'
        match = re.search(pattern, response_text)

        if match:
            joint_num, joint_action = map(int, match.groups())
            actions = [joint_num, joint_action]
            found = True
        return actions, found

    def transform_vector(self, vector):
        a, b = vector
        if a == 1:
            return [b, 0, 0]
        elif a == 2:
            return [0, b, 0]
        else:
            return [0, 0, b]

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

        except OpenAIError as e:
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

            num_turns_to_include = min(self.history_length, len(history))

            history_str = ", ".join(
                [
                    f"at step {i + 1 + len(history) - num_turns_to_include}:  "
                    f"D1: {entry['states']['self.DOF1_last']}, D2: {entry['states']['self.DOF2_last']}, D3: {entry['states']['self.DOF3_last']}, "
                    f"[changed DOF, ROC]: {entry['actions']}, "
                    f"after performing this action, D1: {entry['states']['self.DOF1']}, D2: {entry['states']['self.DOF2']}, D3: {entry['states']['self.DOF3']},"
                    f"swimmer moves from position {entry['results']['last_position']} to position {entry['results']['position']}, "
                    f"stuck: {entry['results']['stuck']}"
                    for i, entry in enumerate(history[-num_turns_to_include:])]
            )

            agent_prompt = (
                f"Determine a sequence of rate of change (ROC) in three degrees of freedom (DOF), D1, D2 and D3, such that the swimmer achieves fastest long-term movement in positive x direction. "
                f"Only one DOF can change simultaneously, and all DOFs must range in [{round(self.low_lim, 3)}, {round(self.up_lim, 3)}]. "
                f"History: {history_str}\n"
                f"Consider the impact of the output on future movement. "
                f"Firstly, respond by indicating which DOF to change, choosing from 3, 2 or 1; secondly, respond by specifying ROC for this DOF, choosing from -1, 1; only output as [changed DOF, ROC], no other info."
            )

            agent_action_output, agent_action = self.get_valid_response(setup_message, agent_prompt)

            self.DOF1, self.DOF2, self.DOF3, history_entry, last_5_elements= self.swimmer.get_history(agent_action_output, agent_action)
            history_entry['results']['last_position'] = round((history_entry['results']['last_position'] - (-2.5)) * (1000 - 0), 3)
            history_entry['results']['position'] = round((history_entry['results']['position'] - (-2.5)) * (1000 - 0), 3)

            trajectory = self.swimmer.get_trajectory()
            history.append(history_entry)

            # # negative
            # if t > 5 and len(history) > 5 and last_5_elements > -0.01:
            #      print("limit:", last_5_elements)
            #      history = []
            # positive
            if t > 5 and len(history) > 5 and last_5_elements < 0.05:
                print("limit:", last_5_elements)
                history = []

            print(f"Step {t}:")
            print("action_output" + str(agent_action_output))
            print(f"Agent chose action {agent_action}")
            print(f"Agent's position: {self.swimmer.pos}")
            print("\n")

            # Save data
            directory_name = f'sim_data/test'
            filename = f'{directory_name}/trajectory_try_1.txt'
            save_and_notify(filename, trajectory)

if __name__ == '__main__':
    # FourSphere() for four-sphere swimmer; FourLink() for four-link swimmer
    swimmer = FourLink()
    llm_learning = LLMInteractionHandler(swimmer)
    llm_learning.simulate_interaction()

# -*- coding: utf-8 -*-

import gym
import pandas as pd
from gym import spaces
import numpy as np
import pypower.api as pp
from pypower.api import case118
from pypower.idx_bus import PD, QD, VM, VA, BUS_AREA
from pypower.idx_gen import PG, QG, VG
from scipy.io import loadmat
import random
import copy
import os
import json
import torch

with open(os.path.join(r'D:\DQN_stastic\code\DQN\code', 'package_DQN.json'), 'r') as fp:
    json_data = json.load(fp)

def tanh(x, k=180):
    # 先比例缩放，再用tanh修正，保持正负区间都有值
    return (np.exp(x/k) - np.exp(-x/k)) / (np.exp(x/k) + np.exp(-x/k))

class CasFailSimEnvCase118(gym.Env):

    def __init__(self):
        self.data_save_dir = r'D:\DQN_stastic\code\DQN\my2\prevent_Control_IEEE39_Data_4'
        if not os.path.exists(self.data_save_dir):
            os.makedirs(self.data_save_dir)
        self.ppc = case118()

        ##初始化，便于训练和生成样本
        # self.ppc['gen'][0:3, VG] = 1.025
        # self.ppc['gen'][3:6, VG] = 0.975
        self.ppc['gen'][0:27, VG] = 1.025  # 1.025
        self.ppc['gen'][27:54, VG] = 0.975  # 0.975
        self.ppc['gen'][4:16, VG] = 1.05
        self.ppc['gen'][21:25, VG] = 1.05
        # self.ppc['gen'][0, VG] = 0.985

        # self.line_total = len(self.ppc['branch'][:, 0])
        # self.failed_line_thre = np.ceil(failed_scale_thre*self.line_total)
        # self.rewards_set = rewards_set
        self.ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
        self.ppc = pp.runpf(self.ppc, self.ppopt)[0]
        self.action_space = spaces.Discrete((len(self.ppc['gen'][:, 0])-1)*5)  # all gen*（0.95，0.975，1.0，1.025，1.05）去除平衡机
        # print(self.action_space)
        self.obs_v_dim = len(self.ppc['bus'][:, VM]) #118
        self.obs_r_dim = 53
        # self.observation_space = spaces.Box(low=-1.5, high=-1.5, shape=(self.obs_dim, 1))
        # print(self.observation_space)
        self.state_bvm = None
        self.state_ra = None
        self.done = False
        self.reward = 0
        self.oc_str = r"D:\DQN_stastic\code\DQN\code"
        self.opercond = loadmat(self.oc_str + "\\abnormal_num.mat")

        self.test_num = self.opercond["sim_abnormal_test"]
        self.train_num = self.opercond["sim_abnormal_train"]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # run cascading failure simulation
        # print(action)
        sim_success, voltage, angle, gen, action = run_sim(self.ppc, action, self.action_gen_vm)
        # 存储动作
        self.action_gen_vm[gen] = action
        print(f'   choose:{gen},action:{action}')
        print(self.action_gen_vm)
        # next_state
        self.state_bvm = voltage
        self.state_ra = angle
        tide = abs(np.average(voltage) - 1)
        #reward
        if sim_success == 0:
            self.reward = json_data['simfail_reward']
        elif all(0.95 < voltage) and all(voltage < 1.05):
            self.reward = json_data['done_reward']
            self.done = True
        else:
            self.reward = json_data['fail_reward']

        return self.state_bvm, self.state_ra, self.reward, self.done, tide

    def reset(self, i_case):
        self.done = False
        # 获取具体仿真工况
        self.i_case = i_case
        # Set the base operating condition here.
        self.ppc = case118()

        # 改变母线，必须改变相应发电机的端电压设定值
        # self.ppc['gen'][0:5, VG] = 1.025
        # self.ppc['gen'][5:10, VG] = 0.975
        self.ppc['gen'][0:27, VG] = 1.025  # 1.025
        self.ppc['gen'][27:54, VG] = 0.975  # 0.975
        self.ppc['gen'][4:16, VG] = 1.05
        self.ppc['gen'][21:25, VG] = 1.05
        # self.ppc['gen'][0, VG] = 0.985

        #根据case初始化负荷水平
        PD_original = copy.deepcopy(self.ppc['bus'])
        QD_original = copy.deepcopy(self.ppc['bus'])
        bus_area_1 = []
        bus_area_2 = []
        bus_area_3 = []
        for item in range(len(self.ppc['bus'][:, BUS_AREA])):
            if self.ppc['bus'][item, BUS_AREA] == 1:
                bus_area_1.append(item)
            elif self.ppc['bus'][item, BUS_AREA] == 2:
                bus_area_2.append(item)
            elif self.ppc['bus'][item, BUS_AREA] == 3:
                bus_area_3.append(item)

        self.ppc['bus'][bus_area_1, PD] = PD_original[bus_area_1, PD] * ((1.4 - 0.6) / 1000 *
                                                                         self.train_num[i_case, 0] + 0.6)
        self.ppc['bus'][bus_area_1, QD] = QD_original[bus_area_1, QD] * ((1.4 - 0.6) / 1000 *
                                                                         self.train_num[i_case, 0] + 0.6)
        self.ppc['bus'][bus_area_2, PD] = PD_original[bus_area_2, PD] * ((1.4 - 0.6) / 1000 *
                                                                         self.train_num[i_case, 0] + 0.6)
        self.ppc['bus'][bus_area_2, QD] = QD_original[bus_area_2, QD] * ((1.4 - 0.6) / 1000 *
                                                                         self.train_num[i_case, 0] + 0.6)
        self.ppc['bus'][bus_area_3, PD] = PD_original[bus_area_3, PD] * ((1.4 - 0.6) / 1000 *
                                                                         self.train_num[i_case, 0] + 0.6)
        self.ppc['bus'][bus_area_3, QD] = QD_original[bus_area_3, QD] * ((1.4 - 0.6) / 1000 *
                                                                         self.train_num[i_case, 0] + 0.6)


        # 相关变量初始化，稳定性与否
        unstability = None

        # # 记录一个运行工况下，已经采取过的控制措施
        # self.actions_list = []
        # self.repeat_action = False

        # 得到受控发电机的端电压设定值,MDP state
        # self.action_gen_vm = pd.DataFrame([copy.deepcopy(self.ppc['gen'][:, VG])], columns=range(30, 40))

        indices = np.array([1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40, 42, 46, 49, 54, 55,
                           56, 59, 61, 62, 65, 66, 69, 70, 72, 73, 74, 76, 77, 80, 85, 87, 89, 90, 91, 92, 99, 100,
                           103, 104, 105, 107, 110, 111, 112, 113, 116])
        # indices = np.array([1, 2, 3, 6, 8, 9, 12])
        self.action_gen_vm = pd.DataFrame([copy.deepcopy(self.ppc['gen'][:, VG])], columns=indices)

        # 创建case目录,如果已经存在，就不需要创建了
        self.case_dir = os.path.join(self.data_save_dir, f'case_{self.i_case}')
        if not os.path.exists(self.case_dir):
            os.mkdir(self.case_dir)

        ##仿真条件
        self.sim_condition={}

        # 单独保存每个工况的初始仿真条件为json文件
        sim_condition_json_path = os.path.join(self.case_dir, f'case_{self.i_case}_sim_condition.json')
        if not os.path.exists(sim_condition_json_path):
            save_conf_json(self.sim_condition, sim_condition_json_path)

        r = pp.runpf(self.ppc, self.ppopt)
        voltage = r[0]['bus'][:, VM]
        # angle = tanh(r[0]['bus'][[1, 2, 5, 7, 8, 11], VA] - r[0]['bus'][0, VA])
        angle = tanh(r[0]['bus'][[3, 5, 7, 9, 11, 14, 17, 18, 23, 24, 25, 26, 30, 31, 33, 35, 39, 41, 45, 48, 53, 54,
                                  55, 58, 60, 61, 64, 65, 68, 69, 71, 72, 73, 75, 76, 79, 84, 86, 88, 89, 90, 91, 98,
                                  99, 102, 103, 104, 106, 109, 110, 111, 112, 115], VA] - r[0]['bus'][0, VA])
        self.state_bvm = voltage
        self.state_ra = angle
        return self.state_bvm, self.state_ra


    '''def reset2(self,i_case):
        self.done=False
        # 获取具体仿真工况
        self.i_case = i_case
        # Set the base operating condition here.
        self.ppc = case118()

        # 改变母线，必须改变相应发电机的端电压设定值
        self.ppc['gen'][0:5, VG] = 1.025
        self.ppc['gen'][5:10, VG] = 0.975

        #根据case初始化负荷水平
        PD_original = copy.deepcopy(self.ppc['bus'])
        QD_original = copy.deepcopy(self.ppc['bus'])
        bus_area_1 = []
        bus_area_2 = []
        bus_area_3 = []
        for item in range(len(self.ppc['bus'][:, BUS_AREA])):
            if self.ppc['bus'][item, BUS_AREA] == 1:
                bus_area_1.append(item)
            elif self.ppc['bus'][item, BUS_AREA] == 2:
                bus_area_2.append(item)
            elif self.ppc['bus'][item, BUS_AREA] == 3:
                bus_area_3.append(item)

        self.ppc['bus'][bus_area_1, PD] = PD_original[bus_area_1, PD] * ((1.2 - 0.8) / 1000 * self.test_num[i_case,0] + 0.8)
        self.ppc['bus'][bus_area_1, QD] = QD_original[bus_area_1, QD] * ((1.2 - 0.8) / 1000 * self.test_num[i_case,0] + 0.8)
        self.ppc['bus'][bus_area_2, PD] = PD_original[bus_area_2, PD] * ((1.2 - 0.8) / 1000 * self.test_num[i_case,0] + 0.8)
        self.ppc['bus'][bus_area_2, QD] = QD_original[bus_area_2, QD] * ((1.2 - 0.8) / 1000 * self.test_num[i_case,0] + 0.8)
        self.ppc['bus'][bus_area_3, PD] = PD_original[bus_area_3, PD] * ((1.2 - 0.8) / 1000 * self.test_num[i_case,0] + 0.8)
        self.ppc['bus'][bus_area_3, QD] = QD_original[bus_area_3, QD] * ((1.2 - 0.8) / 1000 * self.test_num[i_case,0] + 0.8)


        # 相关变量初始化，稳定性与否
        unstability = None

        # # 记录一个运行工况下，已经采取过的控制措施
        # self.actions_list = []
        # self.repeat_action = False

        # 得到受控发电机的端电压设定值,MDP state
        self.action_gen_vm = pd.DataFrame([copy.deepcopy(self.ppc['gen'][:, VG])],columns=range(30,40))

        # 创建case目录,如果已经存在，就不需要创建了
        self.case_dir = os.path.join(self.data_save_dir, f'case_{self.i_case}')
        if not os.path.exists(self.case_dir):
            os.mkdir(self.case_dir)

        ##仿真条件
        self.sim_condition={}

        # 单独保存每个工况的初始仿真条件为json文件
        sim_condition_json_path = os.path.join(self.case_dir, f'case_{self.i_case}_sim_condition.json')
        if not os.path.exists(sim_condition_json_path):
            save_conf_json(self.sim_condition, sim_condition_json_path)

        r = pp.runpf(self.ppc, self.ppopt)
        voltage = r[0]['bus'][:, VM]
        angle = tanh(r[0]['bus'][[29, 31, 32, 33, 34, 35, 36, 37, 38], VA] - r[0]['bus'][30, VA])
        self.state_bvm = voltage
        self.state_ra = angle
        return self.state_bvm, self.state_ra'''

    def render(self, mode='human'):
        return None

    def close(self):
        return None

# def run_sim(grid,action,pre_action_df):
#     '''
#     解析action动作为具体的发发电机动作，运行潮流
#     :param grid: 网络
#     :param action: 动作序号
#     :return:
#     '''
#     action_list = [0.95, 0.975, 1, 1.025, 1.05]
#     action_num = action+1
#     gen_num = action_num//5
#     gen_action = action_num%5
#     ##有个平衡机在31号，编号不连续
#     if action_num <= 5:
#         gen = 30
#         action = action_list[gen_action-1]
#         pass
#     else:
#         if gen_action != 0:
#             gen = gen_num + 31
#             action = action_list[gen_action - 1]
#         else:
#             gen = gen_num + 30
#             action = action_list[- 1]
#         pass
#     #未动作前的设定电压
#     grid['gen'][:, VG] = pre_action_df[:]
#     #修改动作
#     grid['gen'][gen-30, VG] = action
#     r = pp.runpf(grid, pp.ppoption(VERBOSE=0, OUT_ALL=0))
#     success = r[0]['success']
#     voltage = r[0]['bus'][:, VM]
#     # common = r[0]['bus'][0:28, VM]
#     angle = tanh(r[0]['bus'][[29, 31, 32, 33, 34, 35, 36, 37, 38], VA] - r[0]['bus'][30, VA])
#     return success, voltage, angle, gen, action
#     pass

def run_sim(grid,action,pre_action_df):
    '''
    解析action动作为具体的发发电机动作，运行潮流
    :param grid: 网络
    :param action: 动作序号
    :return:
    '''
    action_list = [0.95, 0.975, 1, 1.025, 1.05]
    gen_list = [4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40, 42, 46, 49, 54, 55, 56, 59, 61, 62,
                65, 66, 69, 70, 72, 73, 74, 76, 77, 80, 85, 87, 89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110,
                111, 112, 113, 116]
    # gen_list = [2, 3, 6, 8, 9, 12]
    action_num = action+1
    gen_num = action_num//5
    gen_action = action_num%5
    ##有个平衡机在1号，编号不连续
    if action_num <= 5:
        action = action_list[gen_action-1]
        if gen_action != 0:
            gen = gen_list[gen_num]
        else:
            gen = gen_list[gen_num - 1]
            pass
        pass
    else:
        if gen_action != 0:
            gen = gen_list[gen_num]
            action = action_list[gen_action - 1]
        else:
            gen = gen_list[gen_num - 1]
            action = action_list[- 1]
            pass
    #未动作前的设定电压
    grid['gen'][:, VG] = pre_action_df[:]
    #修改动作
    if gen_action != 0:
        grid['gen'][gen_num+1, VG] = action
    else:
        grid['gen'][gen_num, VG] = action
    r = pp.runpf(grid, pp.ppoption(VERBOSE=0, OUT_ALL=0))
    success = r[0]['success']
    voltage = r[0]['bus'][:, VM]
    angle = tanh(r[0]['bus'][[3, 5, 7, 9, 11, 14, 17, 18, 23, 24, 25, 26, 30, 31, 33, 35, 39, 41, 45, 48, 53, 54,
                              55, 58, 60, 61, 64, 65, 68, 69, 71, 72, 73, 75, 76, 79, 84, 86, 88, 89, 90, 91, 98,
                              99, 102, 103, 104, 106, 109, 110, 111, 112, 115], VA] - r[0]['bus'][0, VA])
    # angle = tanh(r[0]['bus'][[1, 2, 5, 7, 8, 11], VA] - r[0]['bus'][0, VA])
    return success, voltage, angle, gen, action
    pass

def save_conf_json(content, json_path):
    """
    将配置选项写入json文件
    """
    import json
    with open(json_path, 'w') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    env = CasFailSimEnvCase118()
    voltage=env.reset(0)
    env.step(2)
    # print(env.state)
    env.step(3)
    # print(env.state)
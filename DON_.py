# coding=utf-8
import os
import scipy.io as scio
import random
from collections import deque, namedtuple
from itertools import count
import gym
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import math
from sim_env import CasFailSimEnvCase118
import json
import utils as utils

unsuccess_test=[]  ##挑选出无法解决的工况

curr_time = utils.get_curr_time()

project_dir = r'D:\DQN_stastic\code\DQN'
code_dir = os.path.join(project_dir, 'code')
# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
device = torch.device("cuda:0")
'''Transition = namedtuple('Transition',
                        ('state_bvm',  'action',    #gt:generator trip
                         'next_state_bvm',  'reward'))'''
Transition = namedtuple('Transition',
                        ('state_bvm',  'state_ra',  'action',    #gt:generator trip
                         'next_state_bvm',  'next_state_ra',  'reward'))

with open(os.path.join(code_dir, 'package_DQN.json'), 'r') as fp:
    json_data = json.load(fp)

num_episodes = json_data['num_episodes']
num_iteration = json_data['num_iteration']
frequency_eval = json_data['frequency_eval']
arg_seed = json_data['arg_seed']

oc_str = r"D:\DQN_stastic\code\DQN\code"
opercond = scio.loadmat(oc_str + "\\abnormal_num.mat")

train_sim_condition_list = np.array(range(6000))
eval_sim_condition_list = np.array(range(1000))

# 隐藏层的大小，每层隐含层神经元个数是相同的
HIDDEN_N = json_data['hyperpara']['hidden_n']

# 决策网络模型_DQN
class Model(nn.Module):
    def __init__(self, OBS_v_N, OBS_r_N, ACT_N):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(OBS_v_N, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.conv2 = nn.Conv1d(HIDDEN_N, HIDDEN_N, kernel_size=3, stride=1, padding=1)
        self.conv2.weight.data.normal_(0, 0.1)
        self.conv2.bias.data.zero_()
        # self.fc2 = nn.Linear(HIDDEN_N, HIDDEN_N)
        # self.fc2.weight.data.normal_(0, 0.1)
        # self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()
        self.fc4 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.zero_()
        self.fc5 = nn.Linear(OBS_r_N, HIDDEN_N)
        self.fc5.weight.data.normal_(0, 0.1)
        self.fc5.bias.data.zero_()
        self.conv6 = nn.Conv1d(HIDDEN_N, HIDDEN_N, kernel_size=3, stride=1, padding=1)
        self.conv6.weight.data.normal_(0, 0.1)
        self.conv6.bias.data.zero_()
        # self.fc6 = nn.Linear(HIDDEN_N, HIDDEN_N)
        # self.fc6.weight.data.normal_(0, 0.1)
        # self.fc6.bias.data.zero_()
        self.fc7 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc7.weight.data.normal_(0, 0.1)
        self.fc7.bias.data.zero_()
        self.fc8 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc8.weight.data.normal_(0, 0.1)
        self.fc8.bias.data.zero_()
        self.fc9 = nn.Linear(530, ACT_N)
        self.fc9.weight.data.normal_(0, 0.1)
        self.fc9.bias.data.zero_()

    def forward(self, obs_bvm, obs_ra):
        obs_bvm = obs_bvm.to(device)
        obs_bvm = F.relu(self.fc1(obs_bvm))
        obs_bvm = obs_bvm.to(device)
        obs_bvm = F.relu(self.conv2(obs_bvm.unsqueeze(-1)).squeeze(-1))
        # obs_bvm = F.relu(self.fc2(obs_bvm))
        obs_bvm = obs_bvm.to(device)
        obs_bvm = F.relu(self.fc3(obs_bvm))
        action_bvm = self.fc4(obs_bvm)
        # action_bvm = action_bvm.to(device)
        obs_ra = obs_ra.to(device)
        obs_ra = F.relu(self.fc5(obs_ra))
        obs_ra = obs_ra.to(device)
        obs_ra = F.relu(self.conv6(obs_ra.unsqueeze(-1)).squeeze(-1))
        # obs_ra = F.relu(self.fc6(obs_ra))
        obs_ra = obs_ra.to(device)
        obs_ra = F.relu(self.fc7(obs_ra))
        action_ra = self.fc8(obs_ra)
        # action_ra = action_ra.to(device)
        action_val = torch.cat([action_bvm, action_ra], dim=1)
        action_val = action_val.to(device)
        action_val = self.fc9(action_val)
        return action_val

    '''def __init__(self, OBS_N, ACT_N):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(OBS_N, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()
        self.fc4 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.zero_()

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = F.relu(self.fc3(obs))
        action_val = self.fc4(obs)
        return action_val'''

"""
# 决策网络模型_DDQN
class DuelingModel(nn.Module):
    def __init__(self, OBS_N, ACT_N):
        super(DuelingModel, self).__init__()
        self.fc1 = nn.Linear(OBS_N, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()
        self.fc4_1 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc4_1.weight.data.normal_(0, 0.1)
        self.fc4_1.bias.data.zero_()
        self.fc4_2 = nn.Linear(HIDDEN_N, 1)
        self.fc4_2.weight.data.normal_(0, 0.1)
        self.fc4_2.bias.data.zero_()

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = F.relu(self.fc3(obs))
        # action advantages
        action_adv = self.fc4_1(obs)
        # state value
        state_val = self.fc4_2(obs)
        return state_val + action_adv - action_adv.mean()
"""

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 智能体
class Agent:
    def __init__(self,
                 use_dbqn=False,
                 use_per=False,
                 batch_size=json_data['hyperpara']['batch_size'],
                 gamma=json_data['hyperpara']['gamma'],
                 eps_start=json_data['hyperpara']['eps_start'],
                 eps_end=json_data['hyperpara']['eps_end'],
                 eps_decay=json_data['hyperpara']['eps_decay'],
                 target_update=json_data['hyperpara']['target_update'],
                 learning_rate=json_data['hyperpara']['learning_rate'],
                 memory_capacity=json_data['hyperpara']['memory_capacity']
                 ):
        self.agent_name = 'DBQN' if use_dbqn else 'DQN'
        if use_per:
            self.agent_name = 'PER_' + self.agent_name
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.e_greed = None
        self.target_update = target_update
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity

        # Get state size so that we can initialize layers correctly based on shape
        self.env = CasFailSimEnvCase118()

        self.v_dim = self.env.obs_v_dim
        self.r_dim = self.env.obs_r_dim

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n
        # Build policy net and target net
        self.policy_net = Model(self.v_dim, self.r_dim, self.n_actions).to(device)
        self.target_net = Model(self.v_dim, self.r_dim, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Set optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_capacity)

        self.steps_done = 0
        self.episode_durations = []

        # Save the file of code and result
        self.runs_dir = os.path.join(project_dir, json_data['runs_dir'])
        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)
        self.code_dir = code_dir

        self.agent_dir = os.path.join(self.runs_dir, self.agent_name)
        if not os.path.exists(self.agent_dir):
            os.mkdir(self.agent_dir)
        self.curr_time = utils.get_curr_time()
        self.log_dir = os.path.join(self.agent_dir, 'log_' + self.curr_time)
        os.mkdir(self.log_dir)
        # Save code file
        self.log_path = os.path.join(self.log_dir, self.curr_time + '.log')
        utils.Logger(self.log_path)
        utils.backup_code(self.code_dir, os.path.join(self.log_dir, self.curr_time + '_backup.zip'),
                          exclude=['.history', '.idea', '.vscode', '__pycache__'])
        # Save result
        # model: final, best
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.final_model_dir = os.path.join(self.model_dir, 'final')
        self.best_model_dir = os.path.join(self.model_dir, 'best')
        # summary data, for tensorboard
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        # creat tensorboard class
        self.writer = SummaryWriter(self.summary_dir)
        # 决策成功的工况计数
        # 模型的历史最优性能,默认为0
        self.best_total_success = 0
        # 存储训练过程中的结果，方便绘图
        self.result_dir = os.path.join(self.log_dir, 'result')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        # 要存储的训练过程中数据，变量声明
        self.eval_rewards = []
        self.eval_actions = []
        self.eval_successes = []
        self.losses = []
        self.epsilons = []

    def select_action(self, s_bvm, s_ra):
        sample = random.random()
        self.e_greed = self.eps_end + (self.eps_start - self.eps_end) * \
                       math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > self.e_greed:
            # 随机数大于ε时，采用贪婪策略选择动作
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # observe = self.policy_net(s_bvm, s_bvp, s_ra, s_gt)
                # selected = self.policy_net(s_bvm, s_bvp, s_ra, s_gt).max(1)[1].view(1, 1)
                return self.policy_net(s_bvm, s_ra).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self, i_episode):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # 提示：这里就是为了处理最后一个时刻的Q值计算问题
        # non_final_mask_bvm = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state_bvm)), device=device, dtype=torch.bool)
        # non_final_mask_ra = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state_ra)), device=device, dtype=torch.bool)
        # non_final_mask = torch.cat((non_final_mask_bvm, non_final_mask_ra), dim=-1)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state_bvm)), device=device, dtype=torch.bool)
        non_final_next_states_bvm = torch.cat([s for s in batch.next_state_bvm if s is not None])
        non_final_next_states_ra = torch.cat([s for s in batch.next_state_ra if s is not None])

        state_batch_bvm = torch.cat(batch.state_bvm)
        state_batch_ra = torch.cat(batch.state_ra)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch_bvm, state_batch_ra).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        '''next_state_values[non_final_mask] = self.target_net(non_final_next_states_bvm,
                                                            ).max(1)[0].detach()'''
        # 利用估计网络得到下阶段的决策，需要扩充维度
        non_final_next_action = self.policy_net(non_final_next_states_bvm,
                                                non_final_next_states_ra,
                                                ).max(1)[1].unsqueeze(1)
        # 利用目标网络评估下阶段决策的Q值，需要缩减维度
        next_state_values[non_final_mask] = self.target_net(non_final_next_states_bvm,
                                                            non_final_next_states_ra,
                                                            ). \
            gather(1, non_final_next_action).detach().squeeze(1)


        # Compute the expected Q values
        '''expected_state_action_values = (next_state_values * self.gamma) + reward_batch'''
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # 限制梯度
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # 记录损失函数
        self.losses.append(loss.item())

        # Record loss
        self.writer.add_scalar("train/value_loss", loss.item(), i_episode)

    def save(self, save_model_dir):
        # 保存模型文件
        torch.save(self.policy_net.state_dict(), save_model_dir)
        print(save_model_dir + " saved.")

    def load(self):
        # 加载模型文件
        self.policy_net.load_state_dict(torch.load(self.best_model_dir))
        # 目标模型要同步一下
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(self.best_model_dir + " loaded.")

def train(agent, i_episode):
    # Set the ID of sim_condition  训练是循环重复的
    i_case = train_sim_condition_list[i_episode % len(train_sim_condition_list)]
    print(f'Episodes: {i_episode + 1}/{num_episodes}. Case: {i_case}\n')
    # Initialize the environment and state
    s_bvm, s_ra = agent.env.reset(i_case)
    s_bvm = state_tensor(s_bvm).to(device)
    s_ra = state_tensor(s_ra).to(device)
    total_reward = 0
    print('Train')

    action = agent.select_action(s_bvm, s_ra)
    # action = torch.tensor([[action]], device=device, dtype=torch.long)
    print('Action {}: '.format(1), end='')
    next_s_bvm, next_s_ra, reward, done, tide1 = agent.env.step(action.item())
    next_s_bvm = state_tensor(next_s_bvm).to(device)
    next_s_ra = state_tensor(next_s_ra).to(device)
    total_reward += reward
    reward = torch.FloatTensor([reward]).to(device)
    if done:
        next_s_bvm = None  # 最后一个Q值即为奖励
        next_s_ra = None
        # Perform one step of the optimization (on the policy network)
        agent.optimize_model(i_episode)
        # print('weight updated.')
        agent.epsilons.append(agent.e_greed)
        agent.writer.add_scalar("train/e_greed", agent.e_greed, i_episode)
        agent.writer.add_scalar("train/finish_step", -1 + 1, i_episode)
        agent.writer.add_scalar("train/total_reward", total_reward, i_episode)
        print('*' * 60, end=' ')
        print(curr_time, end=' ')
        print(agent.agent_name, end=' ')
        print('*' * 10)
        agent.memory.push(s_bvm, s_ra, action, next_s_bvm, next_s_ra, reward)
        # s_bvm = next_s_bvm
        # s_ra = next_s_ra
    else:
        fal = {action.item()}
        agent.memory.push(s_bvm, s_ra, action, next_s_bvm, next_s_ra, reward)
        s_bvm = next_s_bvm
        s_ra = next_s_ra
        action = agent.select_action(s_bvm, s_ra)
        actor = action.item()
        # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
        for t in count():
            t = t + 1
            print('Action {}: '.format(t + 1), end='')
            next_s_bvm, next_s_ra, reward, done, tide = agent.env.step(actor)
            next_s_bvm = state_tensor(next_s_bvm).to(device)
            next_s_ra = state_tensor(next_s_ra).to(device)
            total_reward += reward
            reward = torch.FloatTensor([reward]).to(device)
            if done:
                next_s_bvm = None  # 最后一个Q值即为奖励
                next_s_ra = None
                agent.memory.push(s_bvm, s_ra, action, next_s_bvm, next_s_ra, reward)
                s_bvm = next_s_bvm
                s_ra = next_s_ra
            else:
                fal.add(actor)
                print(fal)
                agent.memory.push(s_bvm, s_ra, action, next_s_bvm, next_s_ra, reward)
                s_bvm = next_s_bvm
                s_ra = next_s_ra
                if tide < tide1:
                    tide1 = tide
                    actor_list = [0.95, 0.975, 1, 1.025, 1.05]
                    actor_num = actor + 1
                    genact_num = actor_num // 5
                    gen_actor = actor_num % 5
                    if actor_num <= 5:
                        actbvm = actor_list[gen_actor - 1]
                        if actbvm > 1.025:
                            actor = actor - 1
                            # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
                        else:
                            actor = actor + 1
                            # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
                        pass
                    else:
                        if gen_actor != 0:
                            actor = actor + 1
                            # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
                        else:
                            actor = actor - 1
                            # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
                        pass
                else:
                    tide1 = tide
                    action = agent.select_action(s_bvm, s_ra)
                    actor = action.item()
                    # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
                    while True:
                        if action.item() in fal:
                            action = agent.select_action(s_bvm, s_ra)
                            actor = action.item()
                            # actor = torch.tensor([[actor]], device=device, dtype=torch.long)
                        else:
                            break

            if done or t == num_iteration - 1:
                # Perform one step of the optimization (on the policy network)
                agent.optimize_model(i_episode)
                # print('weight updated.')
                agent.epsilons.append(agent.e_greed)
                agent.writer.add_scalar("train/e_greed", agent.e_greed, i_episode)
                agent.writer.add_scalar("train/finish_step", t + 1, i_episode)
                agent.writer.add_scalar("train/total_reward", total_reward, i_episode)
                print('*' * 60, end=' ')
                print(curr_time, end=' ')
                print(agent.agent_name, end=' ')
                print('*' * 10)
                break

    # for t in count():
    #     # Select and perform an action
    #     action = agent.select_action(s_bvm, s_ra)
    #     print('Action {}: '.format(t + 1), end='')
    #     next_s_bvm, next_s_ra, reward, done, tide = agent.env.step(action.item())
    #     # print(f'Try times: {t + 1}. Acton: G4-G7({next_s_gt}). Stability: {done}\n')
    #     next_s_bvm = state_tensor(next_s_bvm)
    #     next_s_ra = state_tensor(next_s_ra)
    #     total_reward += reward
    #     reward = torch.FloatTensor([reward]).to(device)
    #     # print(next_s_bvm.type(), reward.type())
    #     if done:
    #         next_s_bvm = None  # 最后一个Q值即为奖励
    #         next_s_ra = None
    #
    #     # Store the transition in memory
    #     agent.memory.push(s_bvm, s_ra, action, next_s_bvm, next_s_ra, reward)
    #
    #     # Move to the next state
    #     s_bvm = next_s_bvm
    #     s_ra = next_s_ra
    #
    #     if done or t == num_iteration - 1:
    #         # Perform one step of the optimization (on the policy network)
    #         agent.optimize_model(i_episode)
    #         # print('weight updated.')
    #         agent.epsilons.append(agent.e_greed)
    #         agent.writer.add_scalar("train/e_greed", agent.e_greed, i_episode)
    #         agent.writer.add_scalar("train/finish_step", t + 1, i_episode)
    #         agent.writer.add_scalar("train/total_reward", total_reward, i_episode)
    #         print('*' * 60, end=' ')
    #         print(curr_time, end=' ')
    #         print(agent.agent_name, end=' ')
    #         print('*' * 10)
    #         break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % agent.target_update == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

def evaluate(agent, i_episode):
    print('=' * 120)
    print('evaluate')
    # 评估样本集总的动作次数
    eval_total_t = 0
    # 评估样本集总的奖励值
    eval_total_reward = 0
    # 评估有效决策的工况数量
    eval_total_success = 0
    # 记录评估过程中的失败工况
    eval_failed_cases = []
    for i_case_index in range(len(eval_sim_condition_list)):
        i_case = eval_sim_condition_list[i_case_index]
        # Initialize the environment and state
        s_bvm, s_ra = agent.env.reset(i_case)
        s_bvm = state_tensor(s_bvm).to(device)
        s_ra = state_tensor(s_ra).to(device)
        total_reward = 0
        for t in count():
            # Select and perform an action
            action = agent.policy_net(s_bvm, s_ra).max(1)[1].view(1, 1)
            # action = torch.tensor([[action]], device=device, dtype=torch.long)
            next_s_bvm, next_s_ra, reward, done, tide = agent.env.step(action.item())
            next_s_bvm = state_tensor(next_s_bvm).to(device)
            next_s_ra = state_tensor(next_s_ra).to(device)
            total_reward += reward

            # Move to the next state
            s_bvm = next_s_bvm
            s_ra = next_s_ra

            # Perform one step of the optimization (on the policy network)
            if done or t == num_iteration - 1:
                eval_total_t += t + 1
                eval_total_reward += total_reward
                if done:
                    eval_total_success += 1
                else:
                    eval_failed_cases.append(i_case)

                break

    # 存储性能最优的模型
    if eval_total_success > agent.best_total_success:
        agent.best_total_success = eval_total_success
        agent.save(agent.best_model_dir)

        # 当评估结果达到新高时，记录该模型的决策性能
        # record(agent)

    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'Episodes: {i_episode + 1}/{num_episodes}. Total evaluation steps: {eval_total_t}.'
          f' Total succeed OCs: {eval_total_success}.'
          f' Time: {time_now}\n')
    print(f'Failed cases: {eval_failed_cases}\n')
    # 存储测试集上的结果
    agent.writer.add_scalar("evaluate/total_iterations", eval_total_t, i_episode)
    agent.writer.add_scalar("evaluate/total_rewards", eval_total_reward, i_episode)
    agent.writer.add_scalar("evaluate/total_successes", eval_total_success, i_episode)

    # 记录评估结果
    agent.eval_rewards.append(eval_total_reward)
    agent.eval_actions.append(eval_total_t)
    agent.eval_successes.append(eval_total_success)

def record(agent):
    # 评估样本集总的动作次数
    eval_total_t = 0
    # 评估样本集总的奖励值
    eval_total_reward = 0
    print('=' * 120)
    print('record')
    for i_case_index in range(len(eval_sim_condition_list)):
        i_case = eval_sim_condition_list[i_case_index]
        print('=' * 120)
        record_num = 'State: {}'.format(i_case)
        print(record_num)
        # Initialize the environment and state
        s_bvm, s_ra = agent.env.reset(i_case)
        s_bvm = state_tensor(s_bvm)
        s_ra = state_tensor(s_ra)
        total_reward = 0
        for t in count():
            # Select and perform an action
            action = agent.policy_net(s_bvm, s_ra).max(1)[1].view(1, 1)
            next_s_bvm, next_s_ra, reward, done, tide = agent.env.step(action.item())
            next_s_bvm = state_tensor(next_s_bvm)
            next_s_ra = state_tensor(next_s_ra)
            total_reward += reward

            # print the decision information
            process_text = '\n'.join([
                'Times [{}]'.format(t + 1),
                'Control measures: {}'.format(agent.env.action_gen_vm),
                'Reword: {}\n'.format(reward),
            ])
            print(process_text)

            # Move to the next state
            s_bvm = next_s_bvm
            s_ra = next_s_ra

            # Perform one step of the optimization (on the policy network)
            if done or t == num_iteration - 1:
                eval_total_t += t + 1
                eval_total_reward += total_reward
                result_text = '\n'.join([
                    'Total times [{}]\n'.format(t + 1),
                    'Final measures: {}'.format(agent.env.action_gen_vm),
                    'Total reword: {}'.format(total_reward),
                ])
                print(result_text)
                break

def state_tensor(s_bvm):
    s_bvm = torch.FloatTensor(s_bvm).unsqueeze(0).to(device)
    return s_bvm

def main():
    for repeat_time in range(json_data['repeat_num']):
        # 创建智能体
        agent = Agent()
        agent.env = agent.env.unwrapped
        agent.env.seed(arg_seed)
        np.random.seed(arg_seed)

        print('Training process start...')
        for i_episode in range(num_episodes):
            # 训练
            train(agent, i_episode)
            # evaluation
            if (i_episode+1) % frequency_eval == 0:
                evaluate(agent, i_episode)
                agent.save(agent.final_model_dir)

        print('Record the decisions made by the final agent.')
        record(agent)

        # save training data for figure plotting (step 3)
        resultdata = os.path.join(agent.result_dir, f'{agent.curr_time}.mat')
        scio.savemat(resultdata, {'eval_rewards': agent.eval_rewards, 'eval_actions': agent.eval_actions,
                                    'eval_successes': agent.eval_successes,
                                    'losses': agent.losses, 'epsilons': agent.epsilons})

        print('Complete')
        agent.env.render()
        agent.env.close()


if __name__ == "__main__":
    main()




#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：q-learning.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/12 3:42 PM 
'''
import numpy as np
import traceback
import gym
import pandas as pd
from config import curr_config_class
iter_max = 100
initial_lr = 1
min_lr = 0.003
gamma = 1
t_max = 29000
eps = 0.1
q_table = np.random.rand(31, 301, 12, 6, 5)
if __name__ == '__main__':
    env_name = "ILPD-v0"
    env = gym.make(env_name)
    action = None
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        for j in range(t_max):
            obs = env.get_obs()
            pa = obs[0]
            pb = obs[1]
            pc = obs[2]
            pe = obs[3]
            if pe > 10:
                env.clear_over_time_car()
                obs = env.get_obs()
                pa = obs[0]
                pb = obs[1]
                pc = obs[2]
                pe = obs[3]
            try:
                if np.random.uniform(0, 1) < eps:
                    action = np.random.choice(10)  # 10就是env.action_space.n
                else:
                    action = np.argmax(q_table[pa][pb][pc][pe])
            except Exception as e:
                traceback.print_exc()
                print(pa,pb,pc,pd)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.step_forward()
            #update q_table
            a = obs[0]
            b = obs[1]
            c = obs[2]
            d = obs[3]
            if d > 10:
                env.clear_over_time_car()
                obs = env.get_obs()
                a = obs[0]
                b = obs[1]
                c = obs[2]
                d = obs[3]
            #use q-learning update
            try:
                q_table[pa][pb][pc][pe][action] = q_table[pa][pb][pc][pe][action] + eta * (reward + gamma * np.max(q_table[a][b][c][d]) - q_table[pa][pb][pc][pe][action])
            except Exception as e:
                traceback.print_exc()
                print(pa, pb, pc, pe)
                print(a, b, c, d)
            if done:
                break
        if i % 5 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i + 1, total_reward))
solution_policy = np.argmax(q_table, axis=4)
sp = pd.DataFrame(columns=['q_value'])
solution_policy = list(solution_policy.flatten())
sp['q_value'] = solution_policy
sp.to_csv(curr_config_class.Q_TABLE_DIRECTORY)






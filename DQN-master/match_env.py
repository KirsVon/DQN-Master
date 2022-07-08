import datetime

import gym
import numpy as np
from gym import spaces, error
from gym import utils
from gym.utils import seeding
from entity.time_window import time_window
from tool.kuhn_munkras import kuhn_munkras

class MatchEnv(gym.Env):
    def __init__(self, max_car_num=30, max_lp_num=300, time_district_num=12, max_time_window_length=10):
        self.max_time_window_length = max_time_window_length
        self.action_spaces = spaces.Discrete(self.max_time_window_length)
        self.observation_space = spaces.MultiDiscrete(
            [max_car_num, max_lp_num, max_time_window_length, time_district_num])
        self.time_window = time_window
        self.matcher = kuhn_munkras(self.time_window)

    """
            #Description: 时间窗口内左右节点匹配
        """

    def match(self):
        self.matcher.change_batch(self.time_window)
        reward, match_res = self.matcher.km()
        unbound_lp_list = self.time_window.node_clear(match_res)
        self.time_window.drop_sent_load_plan(unbound_lp_list)
        return reward

    def time_minus(self, front_time_str: str, rear_time_str:str)->datetime.timedelta:
        front = datetime.datetime.strptime(front_time_str, "%Y%m%d%H%M%S")
        rear = datetime.datetime.strptime(rear_time_str, "%Y%m%d%H%M%S")
        tmp = rear - front
        return tmp

    @property
    def _get_obs(self):
        ob = np.zeros(4,dtype=int)
        time_str = self.time_window.time
        ob[0] = len(time_window.car_list)
        ob[1] = len(time_window.can_be_sent_load_plan)
        ob[2] = int(self.time_window.time[8:10]) / 2
        if len(self.time_window.car_list) != 0:
            first_car_time_Str = self.time_window.car_list[0].arrive_time
            time_delta = self.time_minus(first_car_time_Str, time_str)
            ob[3] = int(time_delta.seconds / 60) + 1
        else:
            ob[3] = 0
        return ob



    def step(self, a):
        reward = 0.0
        self.time_window.get_next_min()
        ob = self._get_obs
        if a <= ob[2] and a != 0:
            reward = self.match()
            ob = self._get_obs
        done = False
        if time_window.time >= time_window.end_time:
            done = True
        return ob, reward, done, {}

    def reset(self):
        self.time_window.reset()
        return self._get_obs





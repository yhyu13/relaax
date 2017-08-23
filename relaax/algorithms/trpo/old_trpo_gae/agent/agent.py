from __future__ import print_function

import numpy as np
import time
from collections import defaultdict

from ... import trpo_config

from ..common import network


class Agent(object):
    def __init__(self, parameter_server, relaax_session):
        self.ps = parameter_server
        self.relaax_session = relaax_session

        self._episode_timestep = 0   # timestep for current episode (round)
        self._episode_reward = 0     # score accumulator for current episode (round)
        self._stop_training = False  # stop training flag to prevent the training further

        self.data = defaultdict(list)

        self.policy = network.make_policy_wrapper(relaax_session, parameter_server.metrics)

        # counter for global updates at parameter server
        self._n_iter = self.ps.session.call_wait_for_iteration()
        relaax_session.op_set_weights(weights=self.ps.session.op_get_weights())

        if trpo_config.config.use_filter:
            self.obs_filter, _ = network.make_filters(trpo_config.config)
            state = self.ps.session.call_get_filter_state()
            self.obs_filter.rs.set(*state)

        self.server_latency_accumulator = 0     # accumulator for averaging server latency
        self.collecting_time = time.time()      # timer for collecting experience

    def act(self, state):
        start = time.time()

        obs = state
        if trpo_config.config.use_filter:
            obs = self.obs_filter(state)
        self.data["observation"].append(obs)

        action, agentinfo = self.policy.act(np.reshape(obs, obs.shape + (1,)))
        self.data["action"].append(action)

        for (k, v) in agentinfo.iteritems():
            self.data[k].append(v)

        self.server_latency_accumulator += time.time() - start
        return action

    def reward_and_act(self, reward, state):
        if not self.reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if self.reward(reward):
            return None
        return self.reset()

    def reset(self):
        score = self._episode_reward

        latency = self.server_latency_accumulator / self._episode_timestep
        self.server_latency_accumulator = 0
        self.ps.metrics.scalar('server_latency', latency)

        self._send_experience(terminated=
                              (self._episode_timestep < trpo_config.config.PG_OPTIONS.timestep_limit))
        return score

    def reward(self, reward):
        self._episode_reward += reward

        # reward = self.reward_filter(reward)
        self.data["reward"].append(reward)

        self._episode_timestep += 1

        return self._stop_training

    def _send_experience(self, terminated=False):
        self.data["terminated"] = terminated
        self.data["filter_diff"] = (0, np.zeros(1), np.zeros(1))
        if trpo_config.config.use_filter:
            mean, std = self.obs_filter.rs.get_diff()
            self.data["filter_diff"] = (self._episode_timestep, mean, std)
        self.ps.session.call_send_experience(self._n_iter, dict(self.data), self._episode_timestep)

        self.data.clear()
        self._episode_timestep = 0
        self._episode_reward = 0

        old_n_iter = self._n_iter
        self._n_iter = self.ps.session.call_wait_for_iteration()
        if self._n_iter == -1:
            self._n_iter = old_n_iter
            return

        if self._n_iter > trpo_config.config.PG_OPTIONS.n_iter:
            self._stop_training = True
            return

        if old_n_iter < self._n_iter:
            print('Collecting time for {} iteration: {}'.format(old_n_iter+1,
                  time.time() - self.collecting_time))
            self.relaax_session.op_set_weights(weights=self.ps.session.op_get_weights())
            self.collecting_time = time.time()

        if trpo_config.config.use_filter:
            state = self.ps.session.call_get_filter_state()
            self.obs_filter.rs.set(*state)

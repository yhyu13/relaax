from builtins import object

import logging
import numpy as np

from relaax.server.common import session
from relaax.common.algorithms.lib import episode  # dataset
from relaax.common.algorithms.lib.utils import ZFilter

from relaax.algorithms.trpo.lib.core import Categorical, DiagGauss

from .. import dppo_config
from .. import dppo_model

logger = logging.getLogger(__name__)


class DPPOBatch(object):
    def __init__(self, parameter_server, exploit, metrics):
        self.exploit = exploit
        self.metrics = metrics
        self.ps = parameter_server
        model = dppo_model.Model(assemble_model=True)
        self.session = session.Session(model.model)

        self.update_rms = model.op_update_rms
        self.session.op_initialize()
        self.episode = None
        self.steps = 0

        self.global_step = 0
        self.pol_loss = None
        self.vf_loss = None

        if dppo_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state = self.lstm_zero_state = model.lstm_zero_state
            self.mini_batch_lstm_state = None
        self.terminal = False

        self.last_state = None
        self.last_action = None
        self.last_prob = None
        self.last_terminal = None

        self.final_state = None
        self.final_value = None

        self.w_step = None

        self.mini_batch_size = dppo_config.config.batch_size
        if dppo_config.config.mini_batch is not None:
            self.mini_batch_size = dppo_config.config.mini_batch

        if dppo_config.config.output.continuous:
            self.prob_type = DiagGauss(dppo_config.config.output.action_size)
        else:
            self.prob_type = Categorical(dppo_config.config.output.action_size)

        if dppo_config.config.use_old_filter:
            self.filter = ZFilter(dppo_config.config.input.shape, clip=5.0)

    @property
    def size(self):
        return self.episode.size

    def begin(self):
        self.load_shared_parameters()
        if dppo_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state

        self.episode = episode.Dataset(['reward', 'state', 'action', 'old_prob', 'terminal'])
        self.episode.begin()

    def step(self, reward, state, terminal):
        self.terminal = terminal
        if dppo_config.config.use_old_filter and self.filter.rs._n > 0:
            state = self.filter(state, update=False)
        self.final_state = state
        self.steps += 1
        if reward is not None:
            self.push_experience(reward)
        if terminal and state is not None:
            logger.debug("DPPOBatch.step doesn't act in case of terminal.")
            state = None
        else:
            assert state is not None
        if not terminal:
            state = np.asarray(state)
            if state.size == 0:
                state = np.asarray([0])
            state = np.reshape(state, state.shape + (1,))
        action, prob = self.get_action_and_prob(state)

        if not terminal:
            self.metrics.histogram('state', self.final_state, self.global_step)
            self.metrics.histogram('action', action, self.global_step)
        self.global_step += 1

        self.keep_state_action_prob(state, action, prob)
        self.last_terminal = terminal
        return action

    def end(self):
        if not self.exploit:
            batch, steps = self.get_batch()
            # Send PPO policy gradient M times and value function gradient B times
            # from https://arxiv.org/abs/1707.02286
            iterations = min(dppo_config.config.policy_iterations, dppo_config.config.value_func_iterations)

            for i in range(iterations):
                if dppo_config.config.use_lstm:
                    self.mini_batch_lstm_state = self.initial_lstm_state
                for mini_batch in batch.iterate_once(self.mini_batch_size):
                    self.update_policy(mini_batch)

            logger.debug('Policy & Value function update finished')
            self.ps.session.op_inc_global_step(increment=steps)

            if dppo_config.config.use_old_filter:
                for i in range(steps):
                    state = batch.experience['state'][i]
                    state = np.reshape(state, newshape=(state.shape[0],))
                    self.filter.rs.push(state)
            elif dppo_config.config.use_filter:
                self.update_rms(self.session._tf_session, np.asarray(batch.experience['state']))

    def get_batch(self):
        batch = self.episode.subset(elements=self.episode.size,
                                    stochastic=not dppo_config.config.use_lstm,
                                    keys=['state', 'action', 'old_prob'])
        experience = self.episode.end()

        values, self.final_value = self.compute_state_values(experience['state'])
        values = np.append(values, np.asarray([self.final_value]))

        terminals = experience['terminal']
        terminals.append(self.terminal)

        adv, vtarg = compute_adv_and_vtarg(experience['reward'], values, terminals)
        if dppo_config.config.norm_adv:
            adv = (adv - adv.mean()) / adv.std()

        if dppo_config.config.vf_clipped_loss:
            batch.extend(adv=adv, vtarg=vtarg, old_vpred=values[:-1])
        else:
            batch.extend(adv=adv, vtarg=vtarg)

        steps = len(experience['state'])
        adv = batch.experience['adv']
        vtarg = batch.experience['vtarg']
        for i in range(steps):
            self.metrics.scalar('adv', adv[i], self.global_step - steps + i)
            self.metrics.scalar('vtarg', vtarg[i], self.global_step - steps + i)

        return batch, steps

    def update_policy(self, experience):
        self.apply_gradients(self.compute_gradients(experience))
        self.load_shared_parameters()  # update_iter=True
        self.metrics.scalar('pol_loss', self.pol_loss, self.w_step)
        self.metrics.scalar('vf_loss', self.vf_loss, self.w_step)

    def reset(self):
        logger.debug('Environment terminated within {} steps.'.format(self.steps))
        self.steps = 0
        if dppo_config.config.use_lstm:
            self.lstm_state = self.lstm_zero_state

    # Helper methods

    def push_experience(self, reward):
        assert self.last_state is not None
        assert self.last_action is not None
        assert self.last_prob is not None
        assert self.last_terminal is not None

        self.episode.step(
            reward=reward,
            state=self.last_state,
            action=self.last_action,
            old_prob=self.last_prob,
            terminal=self.last_terminal
        )
        self.last_state = None
        self.last_action = None
        self.last_prob = None
        self.last_terminal = None

    def get_action_and_prob(self, state):
        if state is None:
            return None, None
        action, prob = self.action_and_prob_from_policy(state)
        assert action is not None
        return action, prob

    def keep_state_action_prob(self, state, action, prob):
        assert self.last_state is None
        assert self.last_action is None
        assert self.last_prob is None

        self.last_state = state
        self.last_action = action
        self.last_prob = prob

    def load_shared_parameters(self, update_iter=False):
        # Load policy parameters from server if they are fresh
        if update_iter:
            new_step = self.ps.session.op_n_step()
        else:
            new_weights, new_step = self.ps.session.op_get_weights_signed()
        msg = "Current policy weights: {}, received weights: {}".format(self.w_step, new_step)

        if (self.w_step is None) or (new_step > self.w_step):
            logger.debug(msg + ", updating weights")
            if not update_iter:
                self.session.op_assign_weights(weights=new_weights)
            self.w_step = new_step
        else:
            logger.debug(msg + ", keeping old weights")

    def action_and_prob_from_policy(self, state):
        assert state is not None
        state = np.asarray(state)
        state = np.reshape(state, (1,) + state.shape)

        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            probabilities, self.lstm_state[0] = \
                self.session.op_get_action(state=state, lstm_state=self.lstm_state[0], lstm_step=[1])
        else:
            probabilities = self.session.op_get_action(state=state)

        # logger.debug("probs: {}".format(probabilities))
        return self.prob_type.sample(probabilities)[0], probabilities[0]

    def compute_gradients(self, experience):
        feeds = dict(state=experience['state'], action=experience['action'],
                     advantage=experience['adv'], old_prob=experience['old_prob'],
                     ytarg_ny=experience['vtarg'])
        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            feeds.update(dict(lstm_ac_state=self.mini_batch_lstm_state[0],
                              lstm_vf_state=self.mini_batch_lstm_state[1],
                              lstm_step=[len(experience['state'])]))
        if dppo_config.config.add_vf_to_pol_loss:
            feeds.update(dict(vf_loss=self.vf_loss))
        if dppo_config.config.vf_clipped_loss:
            feeds.update(dict(vpred_old=experience['old_vpred']))

        if dppo_config.config.use_lstm:
            gradients, self.pol_loss, self.vf_loss, self.mini_batch_lstm_state[0],\
                self.mini_batch_lstm_state[1] = self.session.op_compute_gradients(**feeds)
        else:
            gradients, self.pol_loss, self.vf_loss = self.session.op_compute_gradients(**feeds)
        return gradients

    def apply_gradients(self, gradients):
        self.ps.session.op_submit_gradients(gradients=gradients, step=self.w_step)

    def compute_state_values(self, states):
        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            values, self.lstm_state[1] = \
                self.session.op_value(state=states, lstm_state=self.initial_lstm_state[1],
                                      lstm_step=[len(states)])
            if not self.terminal:
                final_state = np.reshape(self.final_state, (1,) + self.final_state.shape + (1,))
                l_value, tmp = self.session.op_value(state=final_state,
                                                     lstm_state=self.lstm_state[1],
                                                     lstm_step=[1])
        else:
            values = self.session.op_value(state=states)
            if not self.terminal:
                final_state = np.reshape(self.final_state, (1,) + self.final_state.shape + (1,))
                l_value = self.session.op_value(state=final_state)

        last_value = 0 if self.terminal else l_value[0][0]
        return values, last_value


def compute_adv_and_vtarg(rewards, values, terminals):
    exp_size = len(rewards)
    adv = np.empty(exp_size, 'float32')
    gamma = dppo_config.config.gamma
    lam = dppo_config.config.lam
    lastgaelam = 0
    for t in reversed(range(exp_size)):
        non_terminal = 1 - terminals[t + 1]
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * non_terminal * lastgaelam
    vtarg = adv + values[:-1]
    return adv, vtarg

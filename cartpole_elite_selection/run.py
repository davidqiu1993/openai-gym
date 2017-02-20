import pdb
import time

import copy
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

max_episodes = 10000
max_steps = 500


class RLAgent:
  def __init__(self, state_space, action_space):
    self.state_space = state_space # continuous
    self.action_space = action_space # discrete

    self.selection_model_training_batch_size = 16
    self.selection_model_training_epoch = 50
    self.selection_model_training_verbose = 0
    self.selection_model_learning_rate = 0.001

    self.epsilon = 0.70 # exploration rate
    self.epsilon_decay = 0.99
    self.epsilon_max = 0.70
    self.epsilon_min = 0.02

    self.last_state = None
    self.last_action = None
    self.last_threshold = 0

    self.selection_model = Sequential()
    self.sequence_memory = [] # [ (state, action, reward, next_state), ... ]
    self.batch_memory = [] # [ (states, actions, accumulated_rewards), ... ]

    self._build_model()


  def _build_model(self):
    self.selection_model.add(Dense(40, input_shape=self.state_space.shape, activation='relu'))
    self.selection_model.add(Dense(self.action_space.n, activation='softmax'))
    self.selection_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.selection_model_learning_rate))


  def _train_selection_model(self, elite_states, elite_actions):
    """
    elite_states = [ [...state elements...], ... ]
    elite_actions = [ action, ... ]
    """
    categorical_elite_actions = to_categorical(elite_actions, \
                                               nb_classes=self.action_space.n)
    self.selection_model.fit(elite_states, categorical_elite_actions, \
                             verbose=self.selection_model_training_verbose, \
                             nb_epoch=self.selection_model_training_epoch)


  def _predict_action(self, state):
    action_probs = self.selection_model.predict(np.array([state]), verbose=0)[0]
    action = np.random.choice(self.action_space.n, p=action_probs)
    return action


  def next_action(self, new_reward, new_state):
    if self.last_action != None and self.last_state != None:
      self.sequence_memory.append((self.last_state, self.last_action, new_reward, new_state))

    action = self.action_space.sample()
    if np.random.rand() > self.epsilon:
      action = self._predict_action(new_state)

    self.last_action = action
    return action


  def done(self, new_reward, new_state):
    if self.last_action != None and self.last_state != None:
      self.sequence_memory.append((self.last_state, self.last_action, new_reward, new_state))


  def next_episode(self, init_state):
    if len(self.sequence_memory) > 0:
      states = []
      actions = []
      accumulated_rewards = 0
      for i in range(len(self.sequence_memory)):
        state, action, reward, next_state = self.sequence_memory[i]
        states.append(state)
        actions.append(action)
        accumulated_rewards += reward
      self.batch_memory.append((states, actions, accumulated_rewards))
      self.sequence_memory = []

    if len(self.batch_memory) >= self.selection_model_training_batch_size:
      batch_states = []
      batch_actions = []
      batch_accumulated_rewards = []
      for i in range(len(self.batch_memory)):
        states, actions, accumulated_rewards = self.batch_memory[i]
        batch_states.append(states)
        batch_actions.append(actions)
        batch_accumulated_rewards.append(accumulated_rewards)

      mid_accumulated_rewards = np.percentile(batch_accumulated_rewards, 50)
      threshold = max(self.last_threshold * 0.0, mid_accumulated_rewards)
      #threshold = mid_accumulated_rewards
      """
      if threshold <= self.last_threshold * 1.2:
        epsilon = self.epsilon
        for i in range(int(len(self.batch_memory) * 1.5)):
          epsilon = epsilon / self.epsilon_decay
        self.epsilon = self.epsilon_max if epsilon > self.epsilon_max else epsilon
      """
      self.last_threshold = threshold

      elite_states = []
      elite_actions = []
      for i in range(len(self.batch_memory)):
        if batch_accumulated_rewards[i] > threshold:
          elite_states_i = []
          elite_actions_i = []
          for j in range(len(batch_states[i])):
            assert len(batch_states[i]) == len(batch_actions[i])
            elite_states_i.append(batch_states[i][j])
            elite_actions_i.append(batch_actions[i][j])
          elite_states.append(elite_states_i)
          elite_actions.append(elite_actions_i)
      self.batch_memory = []

      if len(elite_states) > 0:
        assert len(elite_states) == len(elite_actions)
        elite_states, elite_actions = map(np.concatenate, [elite_states, elite_actions])
        print('trains selection model (threshold={}, elite_samples={})'.format(threshold, len(elite_states)))
        self._train_selection_model(elite_states, elite_actions)
      else:
        print('no improvements for selection model (threshold={})'.format(threshold))

    self.last_state = init_state
    self.last_action = None

    epsilon = self.epsilon * self.epsilon_decay
    if epsilon < self.epsilon_min:
      epsilon = self.epsilon_min
    self.epsilon = epsilon
    print('episode starts (epsilon={})'.format(self.epsilon))


class World:
  def __init__(self):
    self.env = gym.make('CartPole-v0')
    self.agent = RLAgent(self.env.observation_space, self.env.action_space)


  def run(self):
    for episode in range(max_episodes):
      observation = self.env.reset()
      reward = 0
      self.agent.next_episode(observation)
      for t in range(max_steps):
        #print('episode={}, t={}'.format(episode, t))
        #self.env.render()
        action = self.agent.next_action(reward, observation)
        observation, reward, done, info = self.env.step(action)
        #print('observation={}, reward={}'.format(observation, reward));
        if done:
          reward = 0
          break
      self.agent.done(reward, observation)
      print('episode finished (episode={}, t={})'.format(episode, t+1))
    return


if __name__ == '__main__':
  world = World()
  world.run()
  quit()



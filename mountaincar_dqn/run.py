import gym
import os
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

#os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32' # GPU Acceleration


class QModel(object):
  def __init__(self, state_space, action_space, minibatch, lr):
    super(QModel, self).__init__()

    self.state_space = state_space # continuous state space
    self.action_space = action_space # discrete action space

    self.minibatch = minibatch # mini-batch size
    self.lr = lr # learning rate

    self.model = self._buildModel()

  def _buildModel(self):
    model = Sequential()

    model.add(Dense(output_dim=64, activation='relu', input_dim=self.state_space.shape[0]))
    model.add(Dense(output_dim=self.action_space.n, activation='linear'))

    model.compile(loss='mse', optimizer=RMSprop(lr=self.lr))

    return model

  def train(self, X, Y, epoch=1, verbose=0):
    self.model.fit(X, Y, batch_size=self.minibatch, nb_epoch=epoch, verbose=verbose)

  def predict(self, state):
    prediction = self.model.predict(state)
    return prediction

  def predictAction(self, state):
    prediction = self.predict(state.reshape(1, self.state_space.shape[0]))
    action = np.argmax(prediction.flatten())
    return action

  def save(self, path):
    self.model.save(path)

  def load(self, path):
    self.model.load_weights(path)


class Memory(object):
  def __init__(self, maxlen):
    super(Memory, self).__init__()
    
    self.maxlen = maxlen
    self.memory = deque(maxlen=self.maxlen)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def samples(self, n):
    n = min(n, len(self.memory))
    samples = random.sample(self.memory, n)
    return samples


class RLAgent(object):
  def __init__(self, state_space, action_space):
    super(RLAgent, self).__init__()

    self.state_space = state_space # continuous state space
    self.action_space = action_space # discrete action space

    self.gamma = 0.99 # discount rate

    self.epsilon_max = 0.10
    self.epsilon_min = 0.05
    self.epsilon_decay_factor = 0.01
    self.epsilon_decay_count = 0
    self.epsilon = self.epsilon_max # exploration rate

    self.minibatch = 64 # mini-batch size

    self.q_model_lr = 0.00025 # learning rate
    self.q_model = QModel(self.state_space, self.action_space, self.minibatch, self.q_model_lr)

    self.memory_maxlen = 100000
    self.memory = Memory(self.memory_maxlen)

    self.episode = 0
    self.last_state = None
    self.last_action = None

  def _epsilon_decay(self):
    self.epsilon_decay_count += 1
    self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                   np.exp(- self.epsilon_decay_factor * self.epsilon_decay_count)

  def _replay(self):
    batch = self.memory.samples(self.minibatch)
    batch_size = len(batch)

    states = np.array([ sample[0] for sample in batch ])
    predictions_states = self.q_model.predict(states)

    no_state = np.zeros(self.state_space.shape[0])
    next_states = np.array([ (no_state if sample[4] else sample[3]) for sample in batch ])
    predictions_next_states = self.q_model.predict(next_states)

    X = np.zeros((batch_size, self.state_space.shape[0]))
    Y = np.zeros((batch_size, self.action_space.n))

    for i in range(batch_size):
      state, action, reward, next_state, done = batch[i]

      target = predictions_states[i]
      if done:
        target[action] = reward
      else:
        target[action] = reward + self.gamma * np.amax(predictions_next_states[i])

      X[i] = state
      Y[i] = target

    self.q_model.train(X, Y)

  def next_episode(self, init_state):
    if self.episode > 0:
      self._epsilon_decay()

    self.last_state = init_state
    self.last_action = None

    self.episode += 1

  def observe(self, reward, next_state, done):
    self.memory.remember(self.last_state, self.last_action, reward, next_state, done)

    self.last_state = next_state

    self._replay()

  def act(self):
    if np.random.random() < self.epsilon:
      self.last_action = random.randint(0, self.action_space.n - 1)
    else:
      self.last_action = self.q_model.predictAction(self.last_state)
    return self.last_action

  def save(self, path):
    self.q_model.save(path)

  def load(self, path):
    self.q_model.load(path)


class Environment(object):
  def __init__(self, problem, save=None, save_freq=100, load=None):
    super(Environment, self).__init__()
    
    self.problem = problem
    self.env = gym.make(self.problem)

    self.agent = RLAgent(self.env.observation_space, self.env.action_space)

    self.save = save # save path
    self.save_freq = save_freq # save frequency
    self.load = load # load path

    if self.load is not None:
      self.agent.load(self.load)
      print('load: {}'.format(self.load))


  def run(self, max_episodes, max_timesteps, render_episode=None):
    for episode in range(max_episodes):
      total_reward = 0
      total_timesteps = 0

      init_state = self.env.reset()
      self.agent.next_episode(init_state)
      if render_episode is not None and episode >= render_episode:
        self.env.render()

      #for timestep in range(max_timesteps):
      while True:
        total_timesteps += 1

        action = self.agent.act()
        next_state, reward, done, _info = self.env.step(action)
        total_reward += reward
        self.agent.observe(reward, next_state, done)
        if render_episode is not None and episode >= render_episode:
          self.env.render()

        if done:
          break

      print('episode: {}, timesteps: {}, rewards: {}, memory: {}, epsilon: {}'.format( \
            episode, \
            total_timesteps, \
            total_reward, \
            len(self.agent.memory.memory), \
            np.round(self.agent.epsilon, 2)))

      if episode % self.save_freq == 0:
        self.agent.save(self.save)
        print('save: {}'.format(self.save))


if __name__ == '__main__':
  PROBLEM        = 'MountainCar-v0'
  SAVE_PATH      = './mountain_v0_001.h5'
  SAVE_FREQ      = 5
  LOAD_PATH      = None #'./mountain_v0_001.h5'
  MAX_EPISODES   = 10000
  MAX_TIMESTEPS  = None
  RENDER_EPISODE = 0
  
  environment = Environment(PROBLEM, save=SAVE_PATH, save_freq=SAVE_FREQ, load=LOAD_PATH)
  environment.run(MAX_EPISODES, MAX_TIMESTEPS, render_episode=RENDER_EPISODE)



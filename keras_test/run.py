import copy
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

training_sample_size = 50000;
verification_sample_size = 1000;

value_lower_bound = -50000
value_upper_bound = 50000
reward_lower_bound = -100
reward_upper_bound = 1

model = Sequential()
training_samples = []
verification_samples = []


def build_model():
  model.add(Dense(64, input_dim=2, activation='tanh', init='he_uniform'))
  model.add(Dense(128, activation='tanh', init='he_uniform'))
  model.add(Dense(128, activation='tanh', init='he_uniform'))
  model.add(Dense(2, activation='linear', init='he_uniform'))
  model.compile(loss='mse', optimizer=RMSprop(lr=0.05))
  return


def get_random_sample(min, max):
  sample = (max - min) * np.random.random_sample() + min;
  return sample


def generate_samples():
  lbound = value_lower_bound
  ubound = value_upper_bound
  lreward = reward_lower_bound
  ureward = reward_upper_bound
  for i in range(training_sample_size):
    inputs = [get_random_sample(lbound, ubound), get_random_sample(lbound, ubound)]
    output_sum = inputs[0] + inputs[1]
    output_reward = ureward if output_sum > 0 else lreward
    training_samples.append((inputs, output_sum, output_reward))
  for i in range(verification_sample_size):
    inputs = [get_random_sample(lbound, ubound), get_random_sample(lbound, ubound)]
    output_sum = inputs[0] + inputs[1]
    output_reward = ureward if output_sum > 0 else lreward
    verification_samples.append((inputs, output_sum, output_reward))
  return


def train_model():
  X = []
  Y = []
  for i in range(training_sample_size):
    inputs, output_sum, output_reward = training_samples[i]
    X.append(inputs)
    Y.append([output_sum, output_reward])
  model.fit(X, Y, nb_epoch=8)
  return


def evaluate_model():
  X = []
  Y = []
  for i in range(verification_sample_size):
    inputs, output_sum, output_reward = verification_samples[i]
    X.append(inputs)
    Y.append([output_sum, output_reward])
  predictions = model.predict(X);
  count_sample = 0;
  average_abs_delta_sum = 0;
  average_abs_delta_reward = 0;
  for i in range(verification_sample_size):
    average_abs_delta_sum = (count_sample / (count_sample + 1)) * average_abs_delta_sum + \
                            np.abs((1 / (count_sample + 1)) * (Y[i][0] - predictions[i][0]))
    average_abs_delta_reward = (count_sample / (count_sample + 1)) * average_abs_delta_reward + \
                            np.abs((1 / (count_sample + 1)) * (Y[i][1] - predictions[i][1]))
    count_sample = count_sample + 1
  error_sum = average_abs_delta_sum / (value_upper_bound - value_lower_bound)
  error_reward = average_abs_delta_reward / (reward_upper_bound - reward_lower_bound)
  print("error_sum: %.2f%%, error_reward: %.2f%%" % (error_sum * 100, error_reward * 100))
  return


def run():
  build_model()
  generate_samples()
  train_model()
  evaluate_model()
  return


run()
quit()



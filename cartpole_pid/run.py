import gym


def next_action(env, observation, reward):
  p_distance = 1.00
  d_distance = 0.20
  w_distance = 0.10
  p_angle = 1.00
  d_angle = 0.20
  w_angle = 5.00

  action = env.action_space.sample()

  control = w_distance * (p_distance * observation[0] + d_distance * observation[1]) +  w_angle * (p_angle * observation[2] + d_angle * observation[3])
  if control < 0:
    action = 0
  else:
    action = 1

  return action


def run():
  env = gym.make('CartPole-v0')
  for i_episode in range(5):
    observation = env.reset()
    reward = 0
    for t in range(1000):
      env.render()
      action = next_action(env, observation, reward)
      observation, reward, done, info = env.step(action)
      if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
  return


run()

quit()



import gym
from stable_baselines.trpo_mpi import TRPO

env = gym.make('LunarLanderContinuous-v2')

model = TRPO('MlpPolicy', env)
model.learn(total_timesteps=1000000, verbose=2, tb_log_name="./trpologpath")
model.save('trpo')


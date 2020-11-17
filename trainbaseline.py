import gym
from stable_baselines.sac import SAC

env = gym.make('LunarLanderContinuous-v2')

model = SAC('MlpPolicy', env)
model.learn(total_timesteps=1000000, verbose=2, tb_log_name="./saclogpath")
model.save('sac')


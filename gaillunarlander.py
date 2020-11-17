import gym
from gendata import gen
from traingailhelpers import returnCallbacks, train

## Generate training data
data  = gen('LunarLanderContinuous-v2', "LunarLanderContinuous-v2.pkl", "tmpacro")

env = gym.make('LunarLanderContinuous-v2')
callbacks = returnCallbacks(env, "./acro", "./acrobm", "./acrolog", 10000)
train(env, data, './pendtb', "./tmp", './pendmodel', 1000000, callbacks)

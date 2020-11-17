import gym
from stable_baselines import SAC
from stable_baselines.gail import generate_expert_traj, ExpertDataset

def gen(envName, preTrainedFile, savePath):
    env = gym.make(envName)
    model = SAC.load(preTrainedFile, env)
    generate_expert_traj(model, savePath, env=env, n_episodes=10)
    return ExpertDataset(expert_path=savePath+'.npz', traj_limitation=10, verbose=2)

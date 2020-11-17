## import statements
import gym
import pybulletgym
import stable_baselines
import pybulletgym
from stable_baselines.gail import GAIL
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


env = gym.make("HumanoidPyBulletEnv-v0")
def vecfn():
    return env

def main():
    vecenv = env
    ## initialize callback methods
    checkpoint_cb = CheckpointCallback(save_freq=250000, save_path='./humanoidtest1kon/')
    eval_cb = EvalCallback(env, best_model_save_path='./logson/', log_path='./humanoid1kevalon2/', eval_freq=8000000, deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_cb])

    ## trajectories
    dataset = ExpertDataset(expert_path='humanoidtest1.npz', traj_limitation=50, verbose=2)

    ## train for 1M timesteps
    model = GAIL('MlpPolicy', vecenv, dataset, lam=0.97, verbose=2, tensorboard_log="./humanoid2ktblog2on")
    model.learn(total_timesteps=8000000, callback=callbacks)
    model.save("hum1ktest22on")


if __name__ == "__main__":
    main()
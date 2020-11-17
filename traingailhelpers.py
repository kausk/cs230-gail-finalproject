## import statements
import gym
import stable_baselines
## import pybulletgym
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.gail import GAIL
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

def returnCallbacks(env, checkpoint, bestmodel, logpath, freq):
    checkpoint_cb = CheckpointCallback(save_freq=freq, save_path=checkpoint)
    eval_cb = EvalCallback(env, best_model_save_path=bestmodel, log_path=logpath, eval_freq=freq, deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_cb])

def train(env, dataset, logPath, plotPath, savePath, timesteps, callbacks):
    ## train for 1M timesteps
    env = Monitor(env, plotPath)
    print("starting model")
    model = GAIL('MlpPolicy', env, dataset, lam=0.97, verbose=2, tensorboard_log=logPath)
    print("learning for timesteps", timesteps)
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(logPath)
    results_plotter.plot_results([plotPath], timesteps, results_plotter.X_TIMESTEPS, "test") ## from https://stable-baselines.readthedocs.io/en/master/guide/examples.html


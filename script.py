from spinup import ppo
import tensorflow as tf
import gym
import rocket_lander_gym
env_fn = lambda : gym.make('RocketLander-v0')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='oupput/', exp_name='experiment_name')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=2000, logger_kwargs=logger_kwargs)

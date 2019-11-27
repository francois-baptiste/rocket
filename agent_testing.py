from spinup.utils.test_policy import load_policy, run_policy
import rocket_lander_gym
import gym

_, get_action = load_policy('output')
env = gym.make('RocketLander-v0')
run_policy(env, get_action)
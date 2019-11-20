import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
gym.logger.set_level(40)
from utils.continuous_environments import Environment
env = Environment(gym.make(LunarLanderContinuous-v2), args.consecutive_frames)
env.reset()
state_dim = env.get_state_size()
action_space = gym.make("LunarLanderContinuous-v2").action_space
action_dim = action_space.high.shape[0]
act_range = action_space.high
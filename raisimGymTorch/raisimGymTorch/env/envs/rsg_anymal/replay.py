import argparse
import os
from ruamel.yaml import YAML, dump, RoundTripDumper
from ruamel.yaml.compat import StringIO
import time
import numpy as np
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--state", help="state replay path", type=str, default="")
args = parser.parse_args()
session_key = time.strftime("%Y-%m-%d-%H-%M-%S")
model_path = "./model/" + session_key

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."


# config
cfg = YAML().load(open(task_path + "/cfg.yaml", "r"))
string_io = StringIO()
YAML().dump(cfg["environment"], string_io)
time_step = cfg["environment"]["control_dt"]

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", string_io.getvalue()))
env.seed(cfg["seed"])
assert isinstance(env, VecEnv)
env.seed(cfg["seed"])
env.reset()
print("ob_dim:", env.num_obs, "/ act_dim:", env.num_acts, "/ num_envs:", env.num_envs)

state_replay = np.load(args.state + "/state_replay.npy")
obs_means = np.load(args.state + "/obs_means.npy")
obs_stds = np.load(args.state + "/obs_stds.npy")

# if model_path does not exist, create it
if not os.path.exists(model_path):
    os.makedirs(model_path)

env.start_video_recording(
    os.path.abspath(model_path) + "/video/replay.mp4"
)
print("Visualizing replay")
print(obs_means.shape)
print(obs_stds.shape)
print(obs_means)
print(obs_stds)
print(state_replay.shape)
input()
for k in range(state_replay.shape[0]):
    env.set_observation(np.tile(state_replay[k, :] * obs_stds + obs_means, (env.num_envs, 1)).astype(np.float32), initialize=(k == 0))
    time.sleep(time_step)
    print("forward velocity", state_replay[k, 17])
env.stop_video_recording()
env.turn_off_visualization()

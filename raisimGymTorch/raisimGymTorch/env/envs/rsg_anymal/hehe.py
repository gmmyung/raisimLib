import time
from typing import Tuple
from numpy.typing import NDArray
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
from ruamel.yaml import YAML, dump, RoundTripDumper
from ruamel.yaml.compat import StringIO
import torch, os
from torch import nn
import numpy as np
from tqdm import tqdm
import plotille
import curses
import argparse
import shutil

# POC of training a world model in Raisim

episode_num = 5000
episode_len = 200
bptt_window = 128
bptt_step = 5
test_ratio = 0.2
train_epoch = 4000
batch_size = 128
learning_rate = 0.001

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode", help="set mode either train or test", type=str, default="train"
)
parser.add_argument("-w", "--weight", help="trained weight path", type=str, default="")
args = parser.parse_args()
session_key = time.strftime("%Y-%m-%d-%H-%M-%S")

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."


# config
cfg = YAML().load(open(task_path + "/cfg.yaml", "r"))
string_io = StringIO()
YAML().dump(cfg["environment"], string_io)

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", string_io.getvalue()))
env.seed(cfg["seed"])
assert isinstance(env, VecEnv)
env.seed(cfg["seed"])
env.reset()
print("ob_dim:", env.num_obs, "/ act_dim:", env.num_acts, "/ num_envs:", env.num_envs)


def random_action(num_envs, length, num_acts):
    # random walk
    actions = np.zeros((num_envs, length, num_acts))
    action = np.random.uniform(-1, 1, (num_envs, num_acts))
    for i in range(length):
        actions[:, i, :] = action
        action += np.random.uniform(-0.1, 0.1, (num_envs, num_acts))
        actions.clip(-1, 1)
    return actions.astype(np.float32)


states_history = np.zeros((episode_num, episode_len, env.num_obs))
action_history = np.zeros((episode_num, episode_len, env.num_acts))
states_timeseries = np.zeros((env.num_envs, episode_len, env.num_obs))
action_timeseries = np.zeros((env.num_envs, episode_len, env.num_acts))
time_step = cfg["environment"]["control_dt"]

# World Rollout
for j in tqdm(range(episode_num // env.num_envs + 1)):
    env.reset()
    if j % 1 == 0:
        visualize = True
        env.turn_on_visualization()
    actions = random_action(env.num_envs, episode_len, env.num_acts)
    for i in range(episode_len):
        start = time.time()
        states = env.observe(False)
        states_timeseries[:, i, :] = states
        action_timeseries[:, i, :] = actions[:, i, :]
        env.step(actions[:, i, :].astype(np.float32))
        end = time.time()
        if visualize:
            time.sleep(max(0, time_step - (end - start)))
    visualize = False
    env.turn_off_visualization()
    start_idx = j * env.num_envs
    end_idx = min((j + 1) * env.num_envs, episode_num)
    states_history[start_idx:end_idx, :, :] = states_timeseries[
        0 : end_idx - start_idx, :, :
    ]
    action_history[start_idx:end_idx, :, :] = action_timeseries[
        0 : end_idx - start_idx, :, :
    ]
env.close()

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create RNN training sequences
states_history = torch.from_numpy(states_history).type(torch.float32).to(device)
action_history = torch.from_numpy(action_history).type(torch.float32).to(device)

test_episode_num = int(episode_num * test_ratio)
train_episode_num = episode_num - test_episode_num
bptt_windows_per_episode = (episode_len - bptt_window) // bptt_step + 1


def build_sequence(input: torch.Tensor) -> torch.Tensor:
    episode_num, episode_len, dim = input.shape
    out = torch.zeros((episode_num * bptt_windows_per_episode, bptt_window, dim))
    for i in range(episode_num):
        for j in range(bptt_windows_per_episode):
            start_idx = j * bptt_step
            end_idx = j * bptt_step + bptt_window
            out[i * bptt_windows_per_episode + j, :, :] = input[i, start_idx:end_idx, :]
    return out


# Build training and test sequences
train_input_action = build_sequence(action_history[:train_episode_num])
train_input_state = build_sequence(states_history[:train_episode_num])
test_input_action = build_sequence(action_history[train_episode_num:])
test_input_state = build_sequence(states_history[train_episode_num:])


# Create RNN (LSTM, GRU) model
class LSTM(nn.Module):
    def __init__(self, stack_num=1) -> None:
        super().__init__()
        # Stacked LSTM
        self.stack_num = stack_num
        self.lstms = nn.ModuleList(
            [
                nn.LSTMCell(
                    (env.num_acts + env.num_obs) if i == 0 else env.num_obs, env.num_obs
                )
                for i in range(stack_num)
            ]
        )

        # Output layer
        # This is mandatory because LSTMCell output ranges from [0,1]
        self.mlp = nn.Sequential(
            nn.Linear(env.num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.num_obs),
        )

    def forward(
        self, action: torch.Tensor, state: torch.Tensor, first: bool = False
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x = torch.cat((state, action), 1)
        if first:
            self.hidden_state = [
                torch.zeros(batch_size, env.num_obs) for _ in range(self.stack_num)
            ]
            self.cell_state = [
                torch.zeros(batch_size, env.num_obs) for _ in range(self.stack_num)
            ]
            self.hidden_state[-1] = state
        for i, lstm in enumerate(self.lstms):
            if i == 0:
                self.hidden_state[i], self.cell_state[i] = lstm(
                    x, (self.hidden_state[i], self.cell_state[i])
                )
            else:
                self.hidden_state[i], self.cell_state[i] = lstm(
                    self.hidden_state[i - 1], (self.hidden_state[i], self.cell_state[i])
                )

        return self.mlp(self.hidden_state[-1])


# Train RNN model
model = LSTM(stack_num=1)

# Load model if path is given
if args.weight != "":
    # find file with extension pt within model_path
    model_path = args.weight
    model_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
    if len(model_files) == 0:
        raise ValueError("No model file found in {}".format(model_path))
    model_file = model_files[0]
    model_path = model_path + "/" + model_file
    model.load_state_dict(torch.load(model_path))
    print("weight loaded from {}".format(args.weight))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if args.weight == "":
    training_loss_history = np.array([])
    test_loss_history_teacher_forcing = np.array([])
    test_loss_history_scheduled_sampling = np.array([])
    epoch_timestamp = np.array([])
    training_start_time = time.time()


else:
    training_loss_history = np.load(args.weight + "/training_loss_history.npy")
    test_loss_history_teacher_forcing = np.load(
        args.weight + "/test_loss_history_teacher_forcing.npy"
    )
    test_loss_history_scheduled_sampling = np.load(
        args.weight + "/test_loss_history_scheduled_sampling.npy"
    )
    epoch_timestamp = np.load(args.weight + "/epoch_timestamp.npy")
    training_start_time = time.time() - epoch_timestamp[-1]
    print("loss history loaded from {}".format(args.weight))

fig = plotille.Figure()
fig.width = os.get_terminal_size().columns - 23
fig.height = os.get_terminal_size().lines - 6

# Get normalizing factor by observation dimension
obs_means = train_input_state.flatten(start_dim=0, end_dim=1).mean(0)
obs_stds = train_input_state.flatten(start_dim=0, end_dim=1).std(0)

# Normalize data
train_input_state = (train_input_state - obs_means) / obs_stds
test_input_state = (test_input_state - obs_means) / obs_stds
loss_mean = 1

# Save model
model_path = "./model/" + session_key
if not os.path.exists(model_path):
    os.makedirs(model_path)
os.makedirs(model_path, exist_ok=True)
min_loss = float("inf")


for i in tqdm(range(train_epoch)):
    # Mix data
    idx = torch.randperm(train_input_action.shape[0])
    train_input_action = train_input_action[idx]
    train_input_state = train_input_state[idx]

    loss_sum = 0

    # Create batches
    for j in range((train_episode_num * bptt_windows_per_episode) // batch_size):
        batch_start_idx = j * batch_size
        batch_end_idx = (j + 1) * batch_size
        batch_action = train_input_action[batch_start_idx:batch_end_idx:, :]
        batch_state = train_input_state[batch_start_idx:batch_end_idx, :, :]

        # Train
        optimizer.zero_grad()
        output = batch_state[:, 0, :]

        bptt_length = 128
        warmup_length = 64

        output_history = torch.zeros(batch_size, bptt_length - 1, env.num_obs)

        for k in range(bptt_length - 1):
            action = batch_action[:, k, :]
            if warmup_length > k:
                output = model(action, batch_state[:, k, :], first=(k == 0))
            else:
                output = model(action, output.detach(), first=(k == 0))

            output_history[:, k, :] = output
        loss = criterion(
            output_history[:, warmup_length:, :],
            batch_state[:, warmup_length + 1 : bptt_length, :],
        )
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.item()
    loss_mean = loss_sum / (
        (train_episode_num * bptt_windows_per_episode) // batch_size
    )
    training_loss_history = np.append(training_loss_history, loss_mean)
    epoch_timestamp = np.append(epoch_timestamp, time.time() - training_start_time)

    # Test
    if i % 1 == 0:
        test_window = 128
        warmup_length = 64
        with torch.no_grad():
            env.turn_on_visualization()
            output_history = torch.zeros(
                test_episode_num * bptt_windows_per_episode,
                test_window - 1,
                env.num_obs,
            )

            # Teacher Forcing LSTM loss
            for k in range(test_window - 1):
                action = test_input_action[:, k, :]
                output = model(action, test_input_state[:, k, :], first=(k == 0))
                output_history[:, k, :] = output
            loss = criterion(output_history, test_input_state[:, 1:test_window, :])
            test_loss_history_teacher_forcing = np.append(
                test_loss_history_teacher_forcing, loss.item()
            )

            # Scheduled Sampling LSTM loss
            output = test_input_state[:, 0, :]
            output_history = torch.zeros(
                test_episode_num * bptt_windows_per_episode,
                test_window - 1,
                env.num_obs,
            )
            for k in range(test_window - 1):
                action = test_input_action[:, k, :]
                if warmup_length > k:
                    output = model(action, test_input_state[:, k, :], first=(k == 0))
                else:
                    output = model(action, output.detach(), first=(k == 0))
                output_history[:, k, :] = output

            # Visualize
            if not os.path.exists(model_path + "/video"):
                os.makedirs(model_path + "/video")
            env.start_video_recording(
                os.path.abspath(model_path) + "/video/{}_ground_truth.mp4".format(i)
            )
            print("Visualizing ground truth")
            print("model_path:", model_path)
            visualization_index = np.random.randint(
                test_episode_num * bptt_windows_per_episode
            )
            for k in range(warmup_length - 1, test_window - 1):
                env.set_observation(
                    np.tile(
                        (
                            test_input_state[visualization_index, k, :] * obs_stds
                            + obs_means
                        )
                        .cpu()
                        .numpy(),
                        (env.num_envs, 1),
                    )
                )
                time.sleep(time_step)
            env.stop_video_recording()
            env.start_video_recording(
                os.path.abspath(model_path) + "/video/{}_prediction.mp4".format(i)
            )
            print("Visualizing prediction")
            for k in range(warmup_length - 1, test_window - 1):
                env.set_observation(
                    np.tile(
                        (
                            output_history[visualization_index, k, :] * obs_stds
                            + obs_means
                        )
                        .cpu()
                        .numpy(),
                        (env.num_envs, 1),
                    )
                )
                time.sleep(time_step)
            env.stop_video_recording()
            env.turn_off_visualization()
            loss = criterion(
                output_history[:, warmup_length:, :],
                test_input_state[:, warmup_length + 1 : test_window, :],
            )
            test_loss_history_scheduled_sampling = np.append(
                test_loss_history_scheduled_sampling, loss.item()
            )
            if loss < min_loss:
                min_loss = loss
                # Delete all file with pt extension in model_path
                for f in os.listdir(model_path):
                    if f.endswith(".pt"):
                        os.remove(os.path.join(model_path, f))
                    if (
                        f == "training_loss_history.npy"
                        or f == "epoch_timestamp.npy"
                        or f == "test_loss_history_teacher_forcing.npy"
                        or f == "test_loss_history_scheduled_sampling.npy"
                    ):
                        os.remove(os.path.join(model_path, f))
                # Create new model folder
                os.makedirs(model_path, exist_ok=True)
                # Save model
                torch.save(
                    model.state_dict(),
                    model_path
                    + "/model_loss_mean{}_epoch{}.pt".format(
                        test_loss_history_scheduled_sampling[-1], i
                    ),
                )
                # Save data
                np.save(
                    model_path + "/training_loss_history.npy", training_loss_history
                )
                np.save(
                    model_path + "/test_loss_history_teacher_forcing.npy",
                    test_loss_history_teacher_forcing,
                )
                np.save(
                    model_path + "/test_loss_history_scheduled_sampling.npy",
                    test_loss_history_scheduled_sampling,
                )
                np.save(model_path + "/epoch_timestamp.npy", epoch_timestamp)

    # Plot data
    fig.clear()
    fig.plot(
        epoch_timestamp,
        training_loss_history,
        label="Training loss",
        lc="blue",
    )
    fig.plot(
        epoch_timestamp,
        test_loss_history_teacher_forcing,
        label="Teacher Forcing Test loss",
        lc="red",
    )
    fig.plot(
        epoch_timestamp,
        test_loss_history_scheduled_sampling,
        label="Scheduled Sampling Test loss",
        lc="green",
    )
    fig.set_x_limits(min_=0, max_=epoch_timestamp[-1])
    os.system("clear")
    print(fig.show(legend=True))

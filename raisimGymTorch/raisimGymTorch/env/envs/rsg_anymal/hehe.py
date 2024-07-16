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
bptt_step = 10
test_ratio = 0.2
train_epoch = 1000 
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
    if j == 0:
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
        if np.isnan(states[0, :]).any():
            print(states[0, :])
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

        self.hidden_state = [
            torch.zeros(512, env.num_obs) for _ in range(stack_num)
        ]
        self.cell_state = [
            torch.zeros(512, env.num_obs) for _ in range(stack_num)
        ]

    def forward(
        self, action: torch.Tensor, state: torch.Tensor, first: bool = False
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x = torch.cat((state, action), 1)
        if first:
            for i in range(self.stack_num):
                self.hidden_state[i] = torch.zeros(batch_size, env.num_obs)
                self.cell_state[i] = torch.zeros(batch_size, env.num_obs)
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

    def deepcopy(self):
        new_model = LSTM(stack_num=self.stack_num)
        new_model.load_state_dict(self.state_dict())
        return new_model

# Train RNN model
model = LSTM(stack_num=5)

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
fig.height = os.get_terminal_size().lines - 12

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
            loss = criterion(
                output_history[:, warmup_length:, :],
                test_input_state[:, warmup_length + 1 : test_window, :],
            )
            test_loss_history_scheduled_sampling = np.append(
                test_loss_history_scheduled_sampling, loss.item()
            )

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
                    ),
                    initialize=(k == warmup_length - 1),
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
                    ),
                    initialize=(k == warmup_length - 1),
                )
                time.sleep(time_step)
            env.stop_video_recording()
            env.turn_off_visualization()
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

env.turn_on_visualization()
env.reset()
# model warmup with test data
for i in range(32):
    action = test_input_action[0, i, :].tile(512, 1)
    env.step(action.cpu().numpy().astype(np.float32))
    obs = (torch.Tensor(env.observe(False)[0, :]).to(device) - obs_means) / obs_stds
    output = model(action, obs.tile(512, 1).detach(), first=(i == 0))

print("model warmup done")

state_replay = np.zeros((400, env.num_obs))

for i in range(200):
    # generate random action
    actions = torch.Tensor(random_action(512, 64, env.num_acts)).to(device)
    if i != 0:
        actions.requires_grad = False
        print(actions[top_k_action, 1:64, :].shape)
        actions[0:top_k_action.shape[0] * 16, 0:63, :] = actions[top_k_action, 1:64, :].repeat(16,1, 1)
        # actions[top_k_action, 63, :] = torch.rand(
        #     512, env.num_acts, dtype=torch.float32, device=device
        # )
        actions[0:top_k_action.shape[0] * 16, 0:63, :] += torch.normal(
            torch.zeros(top_k_action.shape[0] * 16, 63, env.num_acts, dtype=torch.float32, device=device),
            torch.ones(top_k_action.shape[0] * 16, 63, env.num_acts, dtype=torch.float32, device=device) * 0.05,
        )
        # actions += torch.normal(
        #     torch.zeros(512, 64, env.num_acts, dtype=torch.float32, device=device),
        #     torch.ones(512, 64, env.num_acts, dtype=torch.float32, device=device) * 0.1,
        # )

    actions.requires_grad = True
    # step on world model 64 times

    # Optimize action
    optimizer = torch.optim.Adam([actions], lr=0.005)
    output_history = [torch.zeros(512, env.num_obs) for _ in range(64)]
    print("optimizing")
    for j in tqdm(range(4)):
        optimizer.zero_grad()
        rollout_model = model.deepcopy() 
        for k in range(64):
            output = rollout_model(actions[:, k, :], output.detach())
            output_history[k] = output
        # loss is sum of forward velocity
        loss = 0
        for k in range(64):
            # if k >= 16:
            loss += torch.sum((output_history[k][:, 17] * obs_stds[17] + obs_means[17] -5).pow(2) )
            loss += torch.sum(2 * (output_history[k][:,47] * obs_stds[47] + obs_means[47]))
            loss += torch.sum(output_history[k][:, 20:23].norm(dim=1))

        loss.backward()
        optimizer.step()

    # visualize
    # action is mean of top 10 action sequence
    output_history = torch.stack(output_history, dim=1)

    loss_per_episode = torch.zeros(512, dtype=torch.float32, device=device)
    loss_per_episode += torch.sum((output_history[:, :, 17] * obs_stds[17] + obs_means[17] - 5).pow(2), dim=1)
    loss_per_episode += torch.sum(2 * (output_history[:, :, 47] * obs_stds[47] + obs_means[47]), dim=1)
    loss_per_episode += torch.sum(output_history[:, :, 20:23].norm(dim=2), dim=1)

    top_k_action = torch.topk(-loss_per_episode, 16, dim=0).indices
    top_action = torch.max(-loss_per_episode, dim=0).indices

    # mean of top 10 action sequence
    # action = torch.mean(actions[top_k_action, :, :], dim=0).tile(512, 1, 1)
    action = actions[top_action, :, :].tile(512, 1, 1)
    for k in range(2):
        env.step(action[0, k, :].tile(env.num_envs, 1).cpu().detach().numpy().astype(np.float32))
        state = (torch.Tensor(env.observe(False)[0, :]).to(device) - obs_means) / obs_stds
        model(action[:, k, :], torch.Tensor(state).to(device).tile(512, 1).detach())
        print("forward velocity", state[17] * obs_stds[17] + obs_means[17])
        print("contact", state[47] * obs_stds[47] + obs_means[47])

        state_replay[i * 2 + k, :] = state.cpu().detach().numpy()

if not os.path.exists(model_path + "/video"):
    os.makedirs(model_path + "/video")
env.start_video_recording(
    os.path.abspath(model_path) + "/video/replay.mp4"
)

env.reset()
print("Visualizing replay")
for k in range(400):
    env.set_observation(np.tile(state_replay[k, :] * obs_stds.cpu().numpy() + obs_means.cpu().numpy(), (env.num_envs, 1)).astype(np.float32), initialize=(k == 0))
    time.sleep(time_step)
# save replay
np.save(model_path + "/state_replay.npy", state_replay)
np.save(model_path + "/obs_means.npy", obs_means.cpu().numpy())
np.save(model_path + "/obs_stds.npy", obs_stds.cpu().numpy())
env.stop_video_recording()
env.turn_off_visualization()

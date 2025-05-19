import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from collections import defaultdict
from tqdm import tqdm
import torch
from torch import nn
import typing as tt
from training.lib.model import ModelActor, ModelCritic
import multiprocessing
from snake_env.envs.snake_env import SnakeEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import ToTensorImage, TransformedEnv
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
import torch.optim as optim
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.envs.transforms import Compose
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type

GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 64

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

print(f"Using device: {device}")

snake_env = SnakeEnv()
env = GymWrapper(snake_env) # This will place observations under the key "observation"

print(f"Observation spec before transform: {env.observation_spec}")
print(f"Action spec before transform: {env.action_spec}")

env = TransformedEnv(env, Compose(
    # ToTensorImage expects "observation" as default in_key if env.observation_spec is standard
    # It will output a tensor under the same key "observation"
    ToTensorImage(in_keys=["pixels"], dtype=torch.float32)
))

# Test the env and get initial data spec
initial_data = env.reset() 
transformed_obs_shape = initial_data["pixels"].shape # Should be (C, H, W), e.g., (3, 84, 84)

print(f"Observation shape after transform: {transformed_obs_shape}")
print(f"Action spec after transform: {env.action_spec.shape}")

actor_net_core = ModelActor(transformed_obs_shape, env.action_spec.shape[0]).to(device)
critic_net = ModelCritic(transformed_obs_shape).to(device)

# Actor network produces raw parameters for the distribution
actor_network_with_extractor = nn.Sequential(
    actor_net_core,
    NormalParamExtractor() # Splits the output of actor_net_core into loc and scale
).to(device)

# This module will process "observation" and output "loc" and "scale"
policy_params_module = TensorDictModule(
    module=actor_network_with_extractor,
    in_keys=["pixels"], # Takes "observation" from environment
    out_keys=["loc", "scale"] # Outputs distribution parameters
)

# ProbabilisticActor samples an action from "loc" and "scale" and outputs "action"
policy_module = ProbabilisticActor(
    module=policy_params_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    out_keys=["action"], # This "action" key will be used by the collector and environment
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True, # Important for PPO
).to(device)


value_module = TensorDictModule(
    critic_net,
    in_keys=["pixels"], # Takes "observation"
    out_keys=["state_value"],
)

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=TRAJECTORY_SIZE,
    total_frames=1_000_000,
    split_trajs=False,
    device=device
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=PPO_BATCH_SIZE),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=GAMMA,
    lmbda=GAE_LAMBDA,
    value_network=value_module,
    average_gae=True,
    device=device
)

ppo_loss = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=PPO_EPS,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optimizer = optim.Adam(ppo_loss.parameters(), lr=LEARNING_RATE_ACTOR)


logs = defaultdict(list)
eval_str = ""
pbar = tqdm(total=1_000_000)

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(TRAJECTORY_SIZE):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(TRAJECTORY_SIZE // PPO_BATCH_SIZE):
            subdata = replay_buffer.sample(PPO_BATCH_SIZE)
            loss_vals = ppo_loss(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(ppo_loss.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 trajectories of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
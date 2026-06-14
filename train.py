from marl_racing_env import marl_racing_environment_v0

import os
from pathlib import Path

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

def get_obs_act_spaces():
    temp_env = marl_racing_environment_v0.parallel_env(render_mode=None)
    first_agent = temp_env.possible_agents[0]
    temp_env.close()

    return temp_env.observation_space(first_agent), temp_env.action_space(first_agent)

def env_creator(args):
    env = marl_racing_environment_v0.parallel_env(render_mode=None)
    return env

def main():
    ray.init()

    env_name = "marl_racing_environment_v0"

    def _make_rllib_env(config):
        base = env_creator(config)
        wrapped = ParallelPettingZooEnv(base)
        wrapped._agent_ids = set(getattr(base, "possible_agents", []))
        return wrapped

    register_env(env_name, _make_rllib_env)

    obs_space, act_space = get_obs_act_spaces()

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            clip_actions=False,
            disable_env_checking=True,
        )
        .multi_agent(
            policies={
                "shared_policy" : (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=None,
        )
        .env_runners(num_env_runners=4, rollout_fragment_length=128)
        .training(
            train_batch_size=2048,
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            # Start with entropy 0.1 at timestep 0, decay to 0.01 by 500k timesteps
            entropy_coeff_schedule = [
                [0, 0.1],
                [1_000_000, 0.01]
            ],
            vf_loss_coeff=0.25,
            num_epochs=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    PROJECT_ROOT = Path(__file__).resolve().parent
    storage_uri = (
            PROJECT_ROOT / "artifacts" / "ray_results" / env_name
    ).resolve().as_uri()

    tune.run(
        "PPO",
        name="PPO_Spwnmix_T2",
        stop={"timesteps_total": 3_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=storage_uri,
        config=config.to_dict(),
    )

    ray.shutdown()

if __name__ == "__main__":
    main()
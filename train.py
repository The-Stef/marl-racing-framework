from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from marl_racing_env import marl_racing_environment_v0
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pathlib import Path
from ray import tune
import ray
import os

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
            clip_actions=True,
            normalize_actions=True,
            disable_env_checking=True,
        )
        .multi_agent(
            policies={
                "shared_policy" : (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=None,
        )
        .env_runners(
            num_env_runners=6,
            num_envs_per_env_runner=4,
            rollout_fragment_length="auto")
        .training(
            train_batch_size=4096,
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            entropy_coeff_schedule=[
                [0, 0.005],
                [1_000_000, 0.001],
                [3_000_000, 0.0],
            ],
            model={
                "free_log_std": True,
            },
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
        name="PPO_CURRICULUM_UPDATES_T1",
        stop={"timesteps_total": 3_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=storage_uri,
        config=config.to_dict(),
    )

    ray.shutdown()

if __name__ == "__main__":
    main()
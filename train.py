from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from marl_racing_env import marl_racing_environment_v0
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pathlib import Path
from ray import tune
import ray
import os

class CurriculumCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        global_step = result["timesteps_total"]

        try:
            # Use foreach_env_runner to iterate over all workers
            algorithm.env_runner_group.foreach_env_runner(
                lambda worker: worker.foreach_env(
                    # Use .par_env to access your MARLRacingEnv instance
                    # inside the ParallelPettingZooEnv wrapper
                    lambda env: env.par_env.set_global_step(global_step)
                )
            )
        except Exception as e:
            print(f"Warning: Failed to update curriculum step: {e}")

def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == "car_0":
        return "fixed_policy"
    return "learning_policy"

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
        .callbacks(CurriculumCallback)
        .environment(
            env=env_name,
            clip_actions=True,
            normalize_actions=True,
            disable_env_checking=True,
        )
        .multi_agent(
            policies={
                "fixed_policy": (None, obs_space, act_space, {}),
                "learning_policy": (None, obs_space, act_space, {}),
            },
            # policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["learning_policy"],
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
                [300_000, 0.005],
                [700_000, 0.001],
                [1_000_000, 0.0005],
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

    # tune.run(
    #     "PPO",
    #     name="JUST_LAP_T2",
    #     stop={"timesteps_total": 5_000_000 if not os.environ.get("CI") else 50000},
    #     checkpoint_freq=10,
    #     storage_path=storage_uri,
    #     config=config.to_dict(),
    #     resume=True,
    # )

    CHECKPOINT_111 = r"C:\Users\dusno\Desktop\marl-racing-framework\artifacts\ray_results\marl_racing_environment_v0\JUST_LAP_T2\PPO_marl_racing_environment_v0_8265b_00000_0_2026-06-27_17-40-22\checkpoint_000111"

    SAVE_DIR = PROJECT_ROOT / "artifacts" / "manual_checkpoints" / "fixed_111_vs_learner"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    algo = config.build()

    old_algo = Algorithm.from_checkpoint(CHECKPOINT_111)
    old_weights = old_algo.get_policy("shared_policy").get_weights()

    algo.get_policy("fixed_policy").set_weights(old_weights)
    algo.get_policy("learning_policy").set_weights(old_weights)

    old_algo.stop()

    target_timesteps = 1_000_000
    save_every = 10

    for i in range(10_000):
        result = algo.train()

        timesteps = result.get("timesteps_total", 0)
        reward_mean = result.get("env_runners", {}).get("episode_reward_mean")

        print(f"iter={i} timesteps={timesteps} reward_mean={reward_mean}")

        if i % save_every == 0:
            ckpt_dir = SAVE_DIR / f"iter_{i:05d}_steps_{timesteps}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = algo.save(str(ckpt_dir))
            print("saved:", checkpoint)

        if timesteps >= target_timesteps:
            final_dir = SAVE_DIR / f"final_steps_{timesteps}"
            final_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = algo.save(str(final_dir))
            print("final saved:", checkpoint)
            break

    algo.stop()

    ray.shutdown()

if __name__ == "__main__":
    main()
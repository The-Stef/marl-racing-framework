from marl_racing_env import marl_racing_environment_v0

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env


ENV_NAME = "marl_racing_environment_v0"

CHECKPOINT_PATH = r"C:\Users\dusno\ray_results\marl_racing_environment_v0\PPO\PPO_marl_racing_environment_v0_19c49_00000_0_2026-06-07_02-16-16\checkpoint_000023"

ADDITIONAL_ENV_STEPS = 500_000
SAVE_EVERY_N_ITERATIONS = 10


def make_rllib_env(config):
    base_env = marl_racing_environment_v0.parallel_env(render_mode=None)
    wrapped_env = ParallelPettingZooEnv(base_env)
    wrapped_env._agent_ids = set(base_env.possible_agents)
    return wrapped_env


def main():
    ray.init(ignore_reinit_error=True)

    # The checkpoint config refers to this env name, so register it before loading.
    register_env(ENV_NAME, make_rllib_env)

    # Load the previous PPO algorithm state.
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)

    start_steps = None
    target_steps = None

    iteration = 0

    while True:
        result = algo.train()
        iteration += 1

        current_steps = result.get("num_env_steps_sampled_lifetime")

        return_mean = (
            result.get("env_runners", {})
            .get("episode_return_mean")
        )

        len_mean = (
            result.get("env_runners", {})
            .get("episode_len_mean")
        )

        if start_steps is None:
            start_steps = current_steps
            target_steps = start_steps + ADDITIONAL_ENV_STEPS

            print(f"Starting from env steps: {start_steps}")
            print(f"Target env steps:        {target_steps}")

        print(
            f"iter={iteration} | "
            f"env_steps={current_steps} | "
            f"return_mean={return_mean} | "
            f"len_mean={len_mean}"
        )

        if iteration % SAVE_EVERY_N_ITERATIONS == 0:
            checkpoint = algo.save()
            print("Saved checkpoint:", checkpoint)

        if current_steps >= target_steps:
            checkpoint = algo.save()
            print("Finished additional training.")
            print("Final checkpoint:", checkpoint)
            break

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
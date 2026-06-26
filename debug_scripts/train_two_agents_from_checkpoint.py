# debug_scripts/train_two_agents_from_checkpoint.py

from pathlib import Path
import argparse
import os

import ray

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from marl_racing_env import marl_racing_environment_v0


ENV_NAME = "marl_racing_environment_v0"


def make_base_env(num_agents: int, render_mode=None):
    try:
        env = marl_racing_environment_v0.parallel_env(
            render_mode=render_mode,
            num_agents=num_agents,
        )
    except TypeError as exc:
        raise TypeError(
            "parallel_env(...) does not seem to accept num_agents. "
            "Check marl_racing_environment_v0.parallel_env and make sure it forwards "
            "num_agents into MARLRacingEnv."
        ) from exc

    actual_agents = len(getattr(env, "possible_agents", []))
    if actual_agents != num_agents:
        env.close()
        raise RuntimeError(
            f"Expected {num_agents} possible_agents, got {actual_agents}. "
            f"possible_agents={getattr(env, 'possible_agents', None)}"
        )

    return env


def env_creator(config):
    num_agents = int(config.get("num_agents", 2))

    base_env = make_base_env(num_agents=num_agents, render_mode=None)
    wrapped_env = ParallelPettingZooEnv(base_env)
    wrapped_env._agent_ids = set(base_env.possible_agents)

    return wrapped_env


def get_spaces(num_agents: int):
    env = make_base_env(num_agents=num_agents, render_mode=None)

    first_agent = env.possible_agents[0]
    obs_space = env.observation_space(first_agent)
    act_space = env.action_space(first_agent)

    env.close()

    return obs_space, act_space


def copy_config_unfrozen(config):
    try:
        return config.copy(copy_frozen=False)
    except TypeError:
        return config.copy()


def build_algo(config):
    if hasattr(config, "build_algo"):
        return config.build_algo()

    return config.build()


def extract_checkpoint_path(save_result):
    """
    RLlib save() may return either:
    - a plain path-like string
    - a TrainingResult with .checkpoint.path
    This keeps the printed output readable.
    """
    checkpoint = getattr(save_result, "checkpoint", None)

    if checkpoint is not None and hasattr(checkpoint, "path"):
        return checkpoint.path

    return str(save_result)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", required=True)

    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--target-timesteps", type=int, default=3_000_000)
    parser.add_argument("--checkpoint-every-timesteps", type=int, default=250_000)

    parser.add_argument("--run-name", default="PPO_LIDAR_T5_to_2Agents_side_by_side_3M")
    parser.add_argument(
        "--output-root",
        default=r"artifacts\ray_results\marl_racing_environment_v0",
    )

    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--num-envs-per-env-runner", type=int, default=4)

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    run_dir = output_root / args.run_name
    checkpoint_out_dir = run_dir / "checkpoints"
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)

    register_env(ENV_NAME, env_creator)

    ray.init(ignore_reinit_error=True)

    print("Loading source checkpoint:")
    print(checkpoint_path)

    source_algo = PPO.from_checkpoint(str(checkpoint_path))
    source_policy = source_algo.get_policy("shared_policy")

    if source_policy is None:
        source_algo.stop()
        ray.shutdown()
        raise RuntimeError("Could not find policy_id='shared_policy' in source checkpoint.")

    source_weights = source_policy.get_weights()
    config = copy_config_unfrozen(source_algo.config)

    source_algo.stop()

    obs_space, act_space = get_spaces(num_agents=args.num_agents)

    config = (
        config.environment(
            env=ENV_NAME,
            env_config={"num_agents": args.num_agents},
            disable_env_checking=True,
        )
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
        )
        .resources(
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    print("Building fresh two-agent PPO trainer...")
    algo = build_algo(config)

    print("Copying shared_policy weights from source checkpoint...")
    algo.get_policy("shared_policy").set_weights(source_weights)

    print("Starting two-agent warm-start training.")
    print("run_dir:", run_dir)
    print("target_timesteps:", args.target_timesteps)
    print("checkpoint_every_timesteps:", args.checkpoint_every_timesteps)

    iteration = 0
    next_checkpoint_at = args.checkpoint_every_timesteps

    try:
        while True:
            iteration += 1
            result = algo.train()

            timesteps_total = int(result.get("timesteps_total", 0) or 0)
            reward_mean = result.get("episode_reward_mean", None)

            print(
                f"iter={iteration} "
                f"timesteps_total={timesteps_total} "
                f"episode_reward_mean={reward_mean}"
            )

            if timesteps_total >= next_checkpoint_at:
                checkpoint_dir = checkpoint_out_dir / f"checkpoint_{timesteps_total:09d}"
                saved = algo.save(str(checkpoint_dir))
                print("saved:", extract_checkpoint_path(saved))

                while next_checkpoint_at <= timesteps_total:
                    next_checkpoint_at += args.checkpoint_every_timesteps

            if timesteps_total >= args.target_timesteps:
                break

        final_dir = checkpoint_out_dir / f"checkpoint_final_{args.target_timesteps:09d}"
        final_saved = algo.save(str(final_dir))
        print("final saved:", extract_checkpoint_path(final_saved))

    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
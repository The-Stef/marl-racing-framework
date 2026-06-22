# debug_scripts/check_latest_checkpoints.py

from pathlib import Path
import argparse
import re

import numpy as np
import ray

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from marl_racing_env import marl_racing_environment_v0


ENV_NAME = "marl_racing_environment_v0"


def checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def make_env(config=None):
    base_env = marl_racing_environment_v0.parallel_env(render_mode=None)
    wrapped_env = ParallelPettingZooEnv(base_env)
    wrapped_env._agent_ids = set(base_env.possible_agents)
    return wrapped_env


def find_run_dir(root: Path, run_name: str) -> Path:
    run_root = (root / run_name).resolve()

    if not run_root.exists():
        raise FileNotFoundError(f"Run folder not found: {run_root}")

    trial_dirs = [p for p in run_root.iterdir() if p.is_dir()]

    if not trial_dirs:
        return run_root

    return max(trial_dirs, key=lambda p: p.stat().st_mtime).resolve()


def find_checkpoints(run_dir: Path, last_n: int):
    checkpoints = [p.resolve() for p in run_dir.rglob("checkpoint_*") if p.is_dir()]
    checkpoints = sorted(checkpoints, key=checkpoint_number)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {run_dir}")

    return checkpoints[-last_n:]


def run_rollout(algo, seed: int, max_steps: int, explore: bool):
    env = marl_racing_environment_v0.parallel_env(render_mode=None)
    obs, infos = env.reset(seed=seed)

    total_reward = 0.0
    done_reason = "not_done"

    actions_seen = []
    rewards_seen = []
    lap_seen = []

    step = 0

    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}

        for agent in env.agents:
            action, _, _ = algo.compute_single_action(
                obs[agent].astype(np.float32),
                policy_id="shared_policy",
                explore=explore,
                full_fetch=True,
            )

            action = np.asarray(action, dtype=np.float32)
            actions[agent] = action

            if agent == env.agents[0]:
                actions_seen.append(action.copy())

        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += float(sum(rewards.values()))

        for info in infos.values():
            done_reason = info.get("done_reason", done_reason)
            rewards_seen.append(float(info.get("reward", 0.0)))
            lap_seen.append(float(info.get("lap_progress", 0.0)))

        if all(terms.values()) or all(truncs.values()):
            break

    env.close()

    actions_arr = np.asarray(actions_seen, dtype=np.float32)
    rewards_arr = np.asarray(rewards_seen, dtype=np.float32)
    lap_arr = np.asarray(lap_seen, dtype=np.float32)

    if actions_arr.size:
        boundary_fraction = float(np.isclose(np.abs(actions_arr), 1.0, atol=1e-5).mean())
        action_mean = np.round(actions_arr.mean(axis=0), 3)
        first_action = np.round(actions_arr[0], 3)
        last_action = np.round(actions_arr[-1], 3)
    else:
        boundary_fraction = 0.0
        action_mean = np.array([0.0, 0.0])
        first_action = np.array([0.0, 0.0])
        last_action = np.array([0.0, 0.0])

    return {
        "seed": seed,
        "explore": explore,
        "steps": step + 1,
        "done_reason": done_reason,
        "total_reward": total_reward,
        "lap_final": float(lap_arr[-1]) if lap_arr.size else 0.0,
        "lap_min": float(lap_arr.min()) if lap_arr.size else 0.0,
        "positive_rewards": int((rewards_arr > 0).sum()) if rewards_arr.size else 0,
        "negative_rewards": int((rewards_arr < 0).sum()) if rewards_arr.size else 0,
        "boundary_fraction": boundary_fraction,
        "action_mean": action_mean,
        "first_action": first_action,
        "last_action": last_action,
    }


def print_results(label: str, results):
    mean_reward = np.mean([r["total_reward"] for r in results])
    mean_lap = np.mean([r["lap_final"] for r in results])
    mean_boundary = np.mean([r["boundary_fraction"] for r in results])
    crashes = sum(r["done_reason"] == "car_crash" for r in results)

    print(f"\n{label}")
    print(
        f"summary: mean_reward={mean_reward:.3f} "
        f"mean_lap={mean_lap:.3f} "
        f"mean_boundary={mean_boundary:.3f} "
        f"crashes={crashes}/{len(results)}"
    )

    for r in results:
        print(
            f"  seed={r['seed']} "
            f"steps={r['steps']} "
            f"reward={r['total_reward']:.3f} "
            f"lap={r['lap_final']:.3f} "
            f"lap_min={r['lap_min']:.3f} "
            f"pos_rewards={r['positive_rewards']} "
            f"neg_rewards={r['negative_rewards']} "
            f"done={r['done_reason']} "
            f"boundary={r['boundary_fraction']:.3f} "
            f"mean_action={r['action_mean'].tolist()} "
            f"first_action={r['first_action'].tolist()} "
            f"last_action={r['last_action'].tolist()}"
        )


def inspect_checkpoint(checkpoint: Path, seeds, max_steps: int, include_explore: bool):
    checkpoint = checkpoint.resolve()

    print("\n" + "=" * 100)
    print(f"CHECKPOINT {checkpoint.name}")
    print(checkpoint)
    print("=" * 100)

    algo = PPO.from_checkpoint(str(checkpoint))

    deterministic_results = [
        run_rollout(algo, seed=seed, max_steps=max_steps, explore=False)
        for seed in seeds
    ]
    print_results("deterministic", deterministic_results)

    if include_explore:
        explore_results = [
            run_rollout(algo, seed=seed, max_steps=max_steps, explore=True)
            for seed in seeds
        ]
        print_results("explore=True", explore_results)

    algo.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="PPO_LIDAR_T5")
    parser.add_argument("--root", default=r"artifacts\ray_results\marl_racing_environment_v0")
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    parser.add_argument("--include-explore", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_dir = find_run_dir(root, args.run_name)
    checkpoints = find_checkpoints(run_dir, args.last)

    print("run_dir:", run_dir)
    print("checking checkpoints:")
    for checkpoint in checkpoints:
        print(" ", checkpoint.name)

    register_env(ENV_NAME, make_env)

    ray.init(ignore_reinit_error=True)

    try:
        for checkpoint in checkpoints:
            inspect_checkpoint(
                checkpoint=checkpoint,
                seeds=args.seeds,
                max_steps=args.steps,
                include_explore=args.include_explore,
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

# py -m debug_scripts.check_latest_checkpoints --run-name PPO_LIDAR_T5 --last 3 --steps 1000 --include-explore
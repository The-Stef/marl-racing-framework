from marl_racing_env import marl_racing_environment_v0

import argparse
import os
import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

ENV_NAME = "marl_racing_environment_v0"

def env_creator(config):
    base_env = marl_racing_environment_v0.parallel_env(render_mode=None)
    wrapped_env = ParallelPettingZooEnv(base_env)
    wrapped_env._agent_ids = set(base_env.possible_agents)
    return wrapped_env

def main():
    parser = argparse.ArgumentParser(
        description="Render pretrained policy loaded from checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the RLlib checkpoint.",
    )

    args = parser.parse_args()
    checkpoint_path = os.path.expanduser(args.checkpoint_path)

    register_env(ENV_NAME, env_creator)

    ray.init(ignore_reinit_error=True)

    algo = PPO.from_checkpoint(checkpoint_path)

    env = marl_racing_environment_v0.parallel_env(render_mode="human")

    observations, infos = env.reset(seed=42)

    reward_sums = {
        agent: 0.0
        for agent in env.possible_agents
    }

    step_count = 0

    while env.agents:
        actions = {}

        for agent in env.agents:
            obs = observations[agent]

            action = algo.compute_single_action(
                obs,
                policy_id="shared_policy",
                explore=False,
            )

            action = np.asarray(action, dtype=np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

            actions[agent] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if step_count <= 20:
            print("step", step_count)
            print("actions:", actions)
            print("observations:", observations)
            print("rewards:", rewards)
            print("terminations:", terminations)
            print("truncations:", truncations)
            print("infos:", infos)
            print("lap progress:", env.LAP_PROGRESS)
            print()

        step_count += 1

        for agent, reward in rewards.items():
            reward_sums[agent] += float(reward)

    env.close()
    algo.stop()
    ray.shutdown()

    print("Reward sums:")
    print(reward_sums)

if __name__ == "__main__":
    main()
from marl_racing_env import marl_racing_environment_v0

def main():
    env = marl_racing_environment_v0.parallel_env(render_mode="human")
    observations, infos = env.reset(seed=42)

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
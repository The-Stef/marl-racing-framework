from stable_baselines3.common.env_checker import check_env
from env.SimpleRacingEnv import SimpleRacingEnv

def main():
    env = SimpleRacingEnv()
    check_env(env, skip_render_check=True)
    env.close()
    print("Environment check passed.")

if __name__ == "__main__":
    main()
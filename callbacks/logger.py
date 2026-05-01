# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
from numbers import Real

from stable_baselines3.common.callbacks import BaseCallback

class LoggerCallback(BaseCallback):
    """
    A custom logging callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # Get individual value: self.locals['infos'][0]['test_field']}
        # Set individual value (number): self.logger.record_mean("custom/test_field", self.locals['infos'][0]['test_field'])

        infos = self.locals.get("infos", [])

        for info in infos:
            for key, value in info.items():
                if key == "TimeLimit.truncated":
                    continue

                entry_name = f"custom/{key}"

                if isinstance(value, bool):
                    self.logger.record(entry_name, value)
                elif isinstance(value, Real):
                    self.logger.record_mean(entry_name, float(value))
                elif isinstance(value, str):
                    self.logger.record(entry_name, value)

        return True
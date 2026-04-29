# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
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

        # self.__class__.__name__ - BaseCallback?
        # self.model.__class__.__name__ - SAC
        # self.logger - <stable_baselines3.common.logger.Logger object at 0x000001C194AC9AD0>
        # self.logger.__class__.__name__ - Logger
        # self.globals - a bunch of stuff
        # self.locals - a bunch of USEFUL stuff



        # Get the value
        # print(f"YO: {self.locals['infos'][0]['the_testt_reward']}")

        # Save the value
        self.logger.record_mean("custom/the_testt_reward", self.locals['infos'][0]['the_testt_reward'])

        return True

from stable_baselines.common.callbacks import BaseCallback
from measure_size import total_size

def get_size(variable,ignore=[]):
    while True:
        try:
            return total_size(variable,ignore=ignore)
        except RuntimeError: # Sometimes a dictionary will change while being iterated (TODO: move this to total_size function?)
            continue

class MemoryCallback(BaseCallback):
    def __init__(self, compute_at=100000, values_to_compute=30,verbose=0):
        super(MemoryCallback, self).__init__(verbose)
        self.memory = list()
        self.values_computed = 0
        self.values_to_compute = values_to_compute
        self.compute_at = compute_at
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if self.values_computed<self.values_to_compute and self.num_timesteps>self.compute_at:
            locals_memory = get_size(self.locals,ignore=[self])
            try:
                nnet_params = self.locals['self'].get_parameters()
            except AttributeError:
                # PPO calls the callback from a Runner object, model is at self.model
                nnet_params = self.locals['self'].model.get_parameters()
            nnet_memory = get_size(nnet_params,ignore=[self])
            self.memory.append(locals_memory+nnet_memory)
            self.values_computed += 1

    def _on_step(self):
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.model.memory_usage = self.memory

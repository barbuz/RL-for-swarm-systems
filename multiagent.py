import threading
import numpy as np
import gym
import os
import sys
import pickle
from glob import glob
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecEnvWrapper, VecNormalize, DummyVecEnv
from stable_baselines.bench import Monitor
from vecmonitor import VecMonitor
import communication as comm


class Multiagent(object):
    """
    Combines a number of homogeneous agents to act in a multiagent environment
    """

    def __init__(self, alg=None, policy=None, env_name=None, seed=None, logdir=None, n_envs=None, normalize=False,
                 swarm=False, communication=None, communication_params=[], _init_setup_model=True, **kwargs):
        self.agents = list()

        self.env_params = dict()
        agent_params = dict()
        for key, val in kwargs.items():
            if key.startswith("_env_param_"):
                self.env_params[key[11:]] = val
            else:
                agent_params[key] = val
        kwargs = agent_params
        if n_envs is None:
            if type(env_name) == str:
                env = gym.make(env_name, **self.env_params)
            else:
                env = env_name
            if seed is not None:
                env.seed(seed)
                env.action_space.seed(seed)
            env = MultiEnv(env)
            monitor = Monitor
        else:
            assert int(n_envs) == n_envs and n_envs > 0, "n_envs should be a positive integer"

            def make_env(rank):
                def _init():
                    env = gym.make(env_name, **self.env_params)
                    if seed is not None:
                        env.seed(seed + rank)
                        env.action_space.seed(seed + rank)
                    return env

                return _init

            env = DummyMultiVecEnv([make_env(i) for i in range(n_envs)])
            if normalize:
                env = VecNormalize(env)
            env = MultiVecEnv(env)
            monitor = VecMonitor
        self.env = env
        self.swarm = swarm
        if _init_setup_model:
            for i in range(env.n_agents):
                if logdir is None:
                    agent_logfile = None
                else:
                    agent_logfile = os.path.join(logdir, "agent{}_monitor.csv".format(i))
                agent_env = monitor(env, agent_logfile)
                if seed is None:
                    agentseed = seed
                else:
                    agentseed = seed+i
                self.agents.append(alg(policy, agent_env, seed=agentseed, **kwargs))
            if self.swarm:
                if communication is None or communication.lower()=="perfect":
                    comm_class = comm.PerfectCommunication
                elif communication.lower() == "distance":
                    comm_class = comm.DistanceCommunication
                elif communication.lower() == "probability":
                    comm_class = comm.ProbabilityCommunication
                else:
                    raise ValueError("Communication value {} unknown. Accepted values are 'perfect', 'distance', and 'probability'.".format(communication))
                try:
                    communication_params = iter(communication_params)
                except: #Allow passing a single value
                    communication_params = [communication_params]

                comm_manager = comm_class(self.agents,self.env,*communication_params)
                self.env.set_comm_manager(comm_manager)
                # for agent in self.agents:
                #     agent.swarm = self.agents

    def set_random_seed(self, seed=None):
        # Ignore if the seed is None
        if seed is None:
            return

        for agent in self.agents:
            agent.set_random_seed(seed)

    def run_for_all(self, fun, *args, kwargs=None):
        threads = list()
        results = [None for _ in self.agents]
        for n, agent in enumerate(self.agents):
            a = [agent, *[arg[n] for arg in args]]
            if kwargs is None:
                k = dict()
            else:
                k = kwargs[n]

            def threadfun(n):
                results[n] = fun(*a, **k)

            # threadfun(n)
            thread = threading.Thread(target=threadfun, name=str(n), args=[n])
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        return results

    def learn(self, *args, callbacks=None):
        def agentlearn(agent, **kwargs):
            agent.learn(*args, **kwargs)
        kwargs = [{'callback':c} for c in callbacks]
        self.run_for_all(agentlearn, kwargs=kwargs)

    def learn_callback(self, total_timesteps, call_timesteps, callback, *args, start_timesteps=0, **kwargs):
        timesteps_passed = start_timesteps
        timesteps_left = total_timesteps - timesteps_passed
        while timesteps_passed < total_timesteps:
            learning_timesteps = min(call_timesteps, total_timesteps - timesteps_passed)
            self.learn(learning_timesteps, reset_num_timesteps=False)
            timesteps_passed += learning_timesteps
            callback(self, timesteps_passed)

    def reset(self):
        result = self.run_for_all(lambda agent: agent.env.reset())
        return result

    def step(self, actions):
        results = self.run_for_all((lambda agent, action: agent.env.step(action)), actions)
        return zip(*results)  # obs, rew, done, info

    def predict(self, observation, deterministic=True):
        results = self.run_for_all((lambda agent, obs: agent.predict(obs,deterministic=deterministic)), observation)
        return zip(*results)  # actions, states

    def save(self, path, *args, **kwargs):
        # pickle.dump(self,open(os.path.join(path,"multiagent.pkl"),'wb'))
        for n, agent in enumerate(self.agents):
            agent.save(os.path.join(path, "agent" + str(n) + ".zip"), *args, **kwargs)

    @classmethod
    def load(cls, path, env, agent_files=None, n_envs=None, logdir=None, agent_class=None):
        if agent_class is None:
            try:
                agent_class, env = pickle.load(os.path.join(path, "multiagent.zip"))
            except:
                raise ValueError("An agent_class should be provided to load from " + path)

        if n_envs is None:
            monitor = Monitor
        else:
            monitor = VecMonitor

        if agent_files is None:
            agent_files = []
            n = 0
            # Try .zip first (new version) and .pkl second (old version)
            files = glob(os.path.join(path, "*agent" + str(n) + ".zip"))
            if len(files)==0:
                files = glob(os.path.join(path, "*agent" + str(n) + ".pkl"))
                if len(files)==0:
                    raise RuntimeError("No agents found in " + path)
            while files:
                if len(files) > 1:
                    raise RuntimeError("More than one file matches " + os.path.join(path, "agent" + str(n) + ".zip"))
                n += 1
                agent_files.append(files[0])
                # Try .zip first (new version) and .pkl second (old version)
                files = glob(os.path.join(path, "*agent" + str(n) + ".zip"))
                if len(files)==0:
                    files = glob(os.path.join(path, "*agent" + str(n) + ".pkl"))

        self = cls(env_name=env, n_envs=n_envs, logdir=logdir, _init_setup_model=False)

        for filename in agent_files:
            print("loading", filename)
            agent = agent_class.load(filename)
            if logdir is None:
                agent_logfile = None
            else:
                agent_logfile = os.path.join(logdir, "agent{}_monitor.csv".format(n))
                nmonitor=1
                while os.path.exists(agent_logfile):
                    nmonitor+=1
                    agent_logfile = os.path.join(logdir, "agent{}_monitor{}.csv".format(n,nmonitor))
            agent_env = monitor(self.env, agent_logfile)
            agent.set_env(agent_env)
            self.agents.append(agent)

        return self

    def evaluate(self, env_name, seed=None, deterministic=False):
        try:
            seeds = iter(seed)
        except TypeError:  # Not iterable
            seeds = [seed]

        env = gym.make(env_name, **self.env_params)

        results = list()
        for seed in seeds:
            rewards = np.zeros(len(self.agents))
            env.seed(seed)
            self.set_random_seed(seed)
            obs = env.reset()
            while True:
                # actions, _ = self.predict(obs)
                actions = [agent.predict(ob, deterministic=deterministic)[0] for (agent,ob) in zip(self.agents,obs)]
                obs, rew, done, infos = env.step(actions)

                rewards += rew
                if done:
                    break
            results.append(np.mean(rewards))

        # Randomize seeds to avoid coming back to the same state after each evaluation
        # os.urandom cannot be seeded, always returns a "true" random value
        self.set_random_seed(int.from_bytes(os.urandom(4), sys.byteorder))

        return np.mean(results)


def divide_space(space, n_agents):
    if isinstance(space, gym.spaces.Box):
        assert space.low.shape[0] == n_agents, "agents must be distributed on the first dimension of the space"
        assert np.all(space.low == space.low[0]), "agents must be uniform"
        return gym.spaces.Box(space.low[0], space.high[0])
    elif isinstance(space, gym.spaces.MultiDiscrete):
        assert space.nvec.shape[0] == n_agents, "agents must be distributed on the first dimension of the space"
        assert np.all(space.nvec == space.nvec[0]), "agents must be uniform"
        if space.nvec.ndim == 1:
            return gym.spaces.Discrete(int(space.nvec[0]))
        else:
            return gym.spaces.MultiDiscrete(space.nvec[0])
    else:
        raise ValueError("Unsupported space type: ", type(space))


def get_info(info, id):
    """Rebuild the info dict for a single agent from a multiagent info"""
    myinfo = dict()
    for key, val in info.items():
        if key == "agents_info":
            myinfo.update(val)
        else:
            myinfo[key] = val
    return myinfo


class MultiEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.n_agents = env.n_agents
        self._multi_event = threading.Event()
        self._multi_actions = dict()
        self._multi_obs = None
        self._multi_reset = 0
        self._mutex = threading.Lock()

        self.observation_space = divide_space(env.observation_space, self.n_agents)
        self.action_space = divide_space(env.action_space, self.n_agents)
        self.manage_comm = False

    def set_comm_manager(self,comm_manager):
        self.comm_manager = comm_manager
        self.manage_comm = True

    def step(self, action):
        id = self.get_id()
        self._mutex.acquire()
        self._multi_actions[id] = action
        if len(self._multi_actions) == self.n_agents:
            actions = [self._multi_actions[a] for a in sorted(self._multi_actions)]
            observations, rewards, done, infos = self.env.step(actions)
            if self.manage_comm:
                self.comm_manager.connect()
            self._multi_obs = observations
            self._multi_rew = rewards
            self._done = done
            self._multi_inf = infos
            self._multi_event.set()
            self._multi_event = threading.Event()  # New event for next multistep
            self._multi_actions = dict()
            self._mutex.release()

        else:
            self._mutex.release()
            self._multi_event.wait()

        return self._multi_obs[id], self._multi_rew[id], self._done, get_info(self._multi_inf, id)

    def reset(self, **kwargs):
        self._mutex.acquire()
        self._multi_reset += 1
        if self._multi_reset == self.n_agents:
            self._multi_obs = self.env.reset(**kwargs)
            if self.manage_comm:
                self.comm_manager.connect()
            self._multi_inf = [None for _ in range(self.n_agents)]
            self._multi_event.set()
            self._multi_reset = 0
            self._multi_event = threading.Event()
            self._mutex.release()
        else:
            self._mutex.release()
            self._multi_event.wait()
        return self._multi_obs[self.get_id()]

    def get_id(self):
        return int(threading.current_thread().name) if self.n_agents>1 else 0

class MultiVecEnv(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.n_agents = env.get_attr("n_agents")[0]
        self._multi_event = threading.Event()
        self._multi_actions = dict()
        self._multi_obs = None
        self._multi_reset = 0
        self._multi_wait = 0
        self._mutex = threading.Lock()

        self.observation_space = divide_space(env.observation_space, self.n_agents)
        self.action_space = divide_space(env.action_space, self.n_agents)
        self.manage_comm = False

    def set_comm_manager(self,comm_manager):
        self.comm_manager = comm_manager
        self.manage_comm = True

    def step_async(self, actions):
        id = self.get_id()
        self._mutex.acquire()
        self._multi_actions[id] = actions
        if len(self._multi_actions) == self.n_agents:
            actions = [*zip(*(self._multi_actions[a] for a in sorted(self._multi_actions)))]
            self.venv.step_async(actions)
        self._mutex.release()

    def step_wait(self):
        id = self.get_id()
        self._mutex.acquire()
        self._multi_wait += 1
        if self._multi_wait == self.n_agents:
            observations, rewards, done, infos = self.venv.step_wait()
            if self.manage_comm:
                self.comm_manager.connect()
            self._multi_obs = observations
            self._multi_rew = rewards
            self._done = done
            self._multi_inf = infos
            self._multi_actions = dict()
            self._multi_event.set()
            self._multi_wait = 0
            self._multi_event = threading.Event()  # New event for next multistep
            self._mutex.release()
        else:
            self._mutex.release()
            self._multi_event.wait()
        my_infos = [get_info(info, id) for info in self._multi_inf]
        return self._multi_obs[:, id], self._multi_rew[:, id], self._done, my_infos

    def reset(self, **kwargs):
        id = self.get_id()
        self._mutex.acquire()
        self._multi_reset += 1
        if self._multi_reset == self.n_agents:
            self._multi_obs = self.venv.reset(**kwargs)
            if self.manage_comm:
                self.comm_manager.connect()
            self._multi_inf = [None for _ in range(self.n_agents)]
            self._multi_event.set()
            self._multi_reset = 0
            self._multi_event = threading.Event()
            self._mutex.release()
        else:
            self._mutex.release()
            self._multi_event.wait()

        return self._multi_obs[:, id]

    def get_id(self):
        return int(threading.current_thread().name) if self.n_agents > 1 else 0


class DummyMultiVecEnv(DummyVecEnv):
    """Modified DummyVecEnv to allow working with a list of rewards"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf_rews = [0] * self.num_envs

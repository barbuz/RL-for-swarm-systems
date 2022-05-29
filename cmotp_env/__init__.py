from gym.envs.registration import register
import cmotp_env.cmotp_v0 as cmotp


import inspect

# register all classes in cmotp_v0.py as gym environments
for name,classfun in inspect.getmembers(cmotp, inspect.isclass):
    register(
        id=name+'-v0',
        entry_point='cmotp_env.cmotp_v0:'+name,
        max_episode_steps=10000
    )

# register all classes in cmotp_v1.py as gym environments
for name,classfun in inspect.getmembers(cmotp, inspect.isclass):
    register(
        id=name+'-v1',
        entry_point='cmotp_env.cmotp_v1:'+name,
        max_episode_steps=10000
    )

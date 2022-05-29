from gym.envs.registration import register
import foodcollecting_env.foodcollecting as fc


import inspect

# register all classes in foodcollecting.py as gym environments
for name,classfun in inspect.getmembers(fc, inspect.isclass):
    register(
        id=name+'-v0',
        entry_point='foodcollecting_env.foodcollecting:'+name,
        max_episode_steps=1000
    )

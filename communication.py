import numpy as np

class Communication:
    def __init__(self,swarm,env=None):
        self.swarm = swarm
        self.env = env

    def seed(self,seed):
        """Seed the random number generator of this object"""
        pass

    def connect(self):
        """Set the 'swarm' field of each member of the swarm to the list of agents
        they can communicate with"""
        pass

class PerfectCommunication(Communication):
    def __init__(self,swarm,env=None):
        super().__init__(swarm,env)
        self.to_be_connected = True

    def connect(self):
        if self.to_be_connected: # Connections never change, so only set them once
            for agent in self.swarm:
                agent.swarm = self.swarm
            self.to_be_connected = False

class ProbabilityCommunication(Communication):
    def __init__(self,swarm,env,connection_probability):
        super().__init__(swarm,env)
        self.cp = connection_probability
        self.rng = np.random.default_rng()

    def seed(self,seed=None):
        self.rng.seed(seed)

    def connect(self):
        connections = self.rng.random([len(self.swarm),len(self.swarm)])<self.cp
        np.fill_diagonal(connections, True)  # You can always communicate with yourself

        for agent,conns in zip(self.swarm,connections):
            agent.swarm = [a for (a, c) in zip(self.swarm, conns) if c]

class DistanceCommunication(Communication):
    """Communicate with other agents within a distance. Assumes environment has
    an ag_ag_dist field as a numpy vector with pairwise distances between agents
    and an agent_dist_norm field as a number with the maximum distance in the environment"""

    def __init__(self,swarm,env,max_dist):
        super().__init__(swarm,env)
        diag = generic_getattr(self.env,"agent_dist_norm")

        self.max_dist = max_dist*diag # Scale to the diagonal of the environment


    def connect(self):
        ag_ag_dist = generic_getattr(self.env,"ag_ag_dist")
        connections = ag_ag_dist<=self.max_dist
        
        for agent,conns in zip(self.swarm,connections):
            agent.swarm = [a for (a, c) in zip(self.swarm, conns) if c]


def generic_getattr(env,attr):
    # Get an attribute from a potentially vectorized environment, that could be wrapped
    try:
        return env.get_attr(attr)[0]
    except AttributeError:
        return getattr(env.unwrapped,attr)

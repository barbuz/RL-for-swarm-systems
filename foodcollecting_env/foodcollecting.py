import gym
import numpy as np
import os
import pyglet

def do_not_render():
    # Do nothing instead of rendering this Geom
    pass


class FoodCollecting(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, size=50, agents_area=0.2, n_agents=3, n_targets=100, cooperation_level=0.0, targets_outside=None):
        self.size = size
        self.agents_area = agents_area
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.cooperation_level = cooperation_level
        if targets_outside is None:
            self.targets_outside = self.agents_area<1
        else:
            self.targets_outside = targets_outside
        self.agent_speed = 1
        self.agent_turn = 2 * np.pi / 7
        self.agent_obs_sectors = 7

        # distance normalization factor for observations
        self.agent_dist_norm = self.size * np.sqrt(2)

        # self.action_space = gym.spaces.Discrete(3)
        low = np.zeros((self.n_agents, self.agent_obs_sectors * 3), dtype=np.float32)
        high = np.ones((self.n_agents, self.agent_obs_sectors * 3), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high)
        self.action_space = gym.spaces.MultiDiscrete([3] * n_agents)

        self.target_value = 1  # reward for collecting a target
        self.living_penalty = -0.1  # reward for each step

        self.viewer = None
        self.agents_pos = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        # action 0=turn_left,1=turn_right,2=move
        actions = np.array(actions)
        headings = np.vstack((np.cos(self.agents_angle), np.sin(self.agents_angle))) * self.agent_speed
        self.agents_pos[:, actions == 2] = np.clip(  # Constrain to the environment
            self.agents_pos[:, actions == 2] + headings[:, actions == 2],
            a_min=np.array([[0], [0]]),
            a_max=np.array([[self.size], [self.size]])
        )
        self.agents_angle[actions == 1] -= self.agent_turn
        self.agents_angle[actions == 0] += self.agent_turn

        targets = self.targets[:, self.alive]

        # compute matrix of distances between agents and targets
        # done with einstein summation convention, seems nice, no idea how it works
        # agents on rows, targets on columns
        ag2 = np.einsum('ij,ij->j', self.agents_pos, self.agents_pos)[:, np.newaxis]
        tg2 = np.einsum('ij,ij->j', targets, targets)[np.newaxis, :]
        agtg = self.agents_pos.T @ targets
        ag_tg_dist = np.sqrt(ag2 + tg2 - 2 * agtg)

        # different (slower) method for distances between agents, to avoid
        # rounding errors when distances are small
        diffs = self.agents_pos - self.agents_pos.T[:, :, np.newaxis]
        ag_ag_dist = np.sqrt(np.einsum('ijk,ijk->ik', diffs, diffs))
        self.ag_ag_dist = ag_ag_dist

        # has target been collected?
        collected = np.any(ag_tg_dist < 1, axis=0)
        # how many targets is agent collecting?
        collecting = np.sum(ag_tg_dist < 1, axis=1)

        rewards = collecting.T * self.target_value + self.living_penalty
        # Balance personal reward with average group reward
        rewards = list((1 - self.cooperation_level) * rewards + self.cooperation_level * np.mean(rewards))

        self.agents_rewards += rewards

        done = np.all(collected)

        # Remove collected targets
        self.alive[self.alive] = np.logical_not(collected)
        targets = self.targets[:, self.alive]
        ag_tg_dist = ag_tg_dist[:, np.logical_not(collected)]

        # Compute observations
        observations = list()
        infos = list()

        sector_width = 2 * np.pi / self.agent_obs_sectors
        # vectorized operations shape everything like ([2],n_agents,n_targets)
        tpos = targets[:, np.newaxis, :] - self.agents_pos[:, :, np.newaxis]
        tangle = self.agents_angle[:, np.newaxis] - np.arctan2(tpos[1, :, :], tpos[0, :, :])
        tangle += sector_width / 2  # make the first sector centered in front
        targets_sector = np.floor(tangle / sector_width % self.agent_obs_sectors)

        apos = self.agents_pos[:, np.newaxis, :] - self.agents_pos[:, :, np.newaxis]
        aangle = self.agents_angle[:, np.newaxis] - np.arctan2(apos[1, :, :], apos[0, :, :])
        aangle += sector_width / 2  # make the first sector centered in front
        agents_sector = np.floor(aangle / sector_width % self.agent_obs_sectors)
        np.fill_diagonal(agents_sector,-1) # an agent should not give a sector to itself

        # Get distance from closest wall in sector's direction
        # done thanks to the magic of trigonometry
        sectors = sector_width * np.arange(self.agent_obs_sectors)[np.newaxis, :]
        sector_angle = self.agents_angle[:, np.newaxis] - sectors  # shape [n_agents,n_sectors]
        cos, sin = np.cos(sector_angle), np.sin(sector_angle)
        xdist = ((cos > 0) * self.size - self.agents_pos[0, :, np.newaxis]) / cos  # TODO: what happens if cos is 0?
        ydist = ((sin > 0) * self.size - self.agents_pos[1, :, np.newaxis]) / sin  # TODO: what happens if sin is 0?
        walldist = np.minimum(xdist, ydist)
        walls_obs = walldist / self.agent_dist_norm

        targets_obs = np.ones((self.n_agents, self.agent_obs_sectors))
        agents_obs = np.ones((self.n_agents, self.agent_obs_sectors))
        targets_info = -np.ones((self.n_agents, self.agent_obs_sectors), dtype=int)
        agents_info = -np.ones((self.n_agents, self.agent_obs_sectors), dtype=int)

        for s in range(self.agent_obs_sectors):
            if not done:  # don't try to find closest target when there are none
                ag_tg_secdist = ag_tg_dist.copy()
                ag_tg_secdist[targets_sector != s] = np.inf
                closest_targets = ag_tg_secdist.argmin(axis=1)
                targets_obs[:, s] = ag_tg_secdist[np.arange(self.n_agents), closest_targets] / self.agent_dist_norm
                targets_info[:, s] = closest_targets

            ag_ag_secdist = ag_ag_dist.copy()
            ag_ag_secdist[agents_sector != s] = np.inf
            closest_agents = ag_ag_secdist.argmin(axis=1)
            agents_obs[:, s] = ag_ag_secdist[np.arange(self.n_agents), closest_agents] / self.agent_dist_norm
            agents_info[:, s] = closest_agents

        # set infinite distances to 1 and relative infos to -1
        targets_not_found = targets_obs == np.inf
        targets_obs[targets_not_found] = 1
        targets_info[targets_not_found] = -1
        agents_not_found = agents_obs == np.inf
        agents_obs[agents_not_found] = 1
        agents_info[agents_not_found] = -1

        for agent in range(self.n_agents):
            observations.append(np.hstack((targets_obs[agent, :], agents_obs[agent, :], walls_obs[agent, :])))
            infos.append({"targets_seen": targets_info[agent, :], "agents_seen": agents_info[agent, :]})

        self.agents_info = infos
        self.agents_obs = observations
        # infos needs to be a dict for compatibility with wrappers (e.g. timelimit)
        return observations, rewards, done, {'agents_info': infos} # TODO: sum(rewards) makes it work for centralized versions

    def _get_state(self):
        targets = self.targets[:, self.alive]

        # compute matrix of distances between agents and targets
        # done with einstein summation convention, seems nice, no idea how it works
        # agents on rows, targets on columns
        ag2 = np.einsum('ij,ij->j', self.agents_pos, self.agents_pos)[:, np.newaxis]
        tg2 = np.einsum('ij,ij->j', targets, targets)[np.newaxis, :]
        agtg = self.agents_pos.T @ targets
        ag_tg_dist = np.sqrt(ag2 + tg2 - 2 * agtg)

        # different (slower) method for distances between agents, to avoid
        # rounding errors when distances are small
        diffs = self.agents_pos - self.agents_pos.T[:, :, np.newaxis]
        ag_ag_dist = np.sqrt(np.einsum('ijk,ijk->ik', diffs, diffs))
        self.ag_ag_dist = ag_ag_dist

        # Compute observations
        observations = list()
        # TODO: can this be vectorized?
        for agent in range(self.n_agents):
            agent_pos = self.agents_pos[:, agent:agent + 1]  # Column vector
            sector_width = 2 * np.pi / self.agent_obs_sectors

            tpos = self.targets - agent_pos
            tangle = self.agents_angle[agent] - np.arctan2(tpos[1, :], tpos[0, :])
            tangle += sector_width / 2  # make the first sector centered in front
            targets_sector = np.floor(tangle / sector_width % self.agent_obs_sectors)

            apos = self.agents_pos - agent_pos
            aangle = self.agents_angle[agent] - np.arctan2(apos[1, :], apos[0, :])
            aangle += sector_width / 2  # make the first sector centered in front
            agents_sector = np.floor(aangle / sector_width % self.agent_obs_sectors)
            agents_sector[agent] = -1 # an agent should not give a sector to itself

            targets_dist = ag_tg_dist[agent, :]
            agents_dist = ag_ag_dist[agent, :]

            targets_obs = np.ones(self.agent_obs_sectors)
            agents_obs = np.ones(self.agent_obs_sectors)
            walls_obs = np.ones(self.agent_obs_sectors)
            for s in range(self.agent_obs_sectors):
                targets_in_sector, = np.where(targets_sector == s)
                if targets_in_sector.size > 0:
                    closest_target = targets_dist[targets_in_sector].argmin()
                    targets_obs[s] = targets_dist[closest_target] / self.agent_dist_norm

                agents_in_sector, = np.where(agents_sector == s)
                if agents_in_sector.size > 0:
                    closest_agent = agents_dist[agents_in_sector].argmin()
                    agents_obs[s] = agents_dist[closest_agent] / self.agent_dist_norm

                # Get distance from closest wall in sector's direction
                # done thanks to the magic of trigonometry
                # TODO: check this works in practice
                sector_angle = self.agents_angle[agent] - sector_width * s
                cos, sin = np.cos(sector_angle), np.sin(sector_angle)
                if cos > 0:
                    xdist = (self.size - agent_pos[0]) * (1 / cos)
                elif cos < 0:
                    xdist = agent_pos[0] * (-1 / cos)
                else:
                    xdist = np.inf
                if sin > 0:
                    ydist = (self.size - agent_pos[1]) * (1 / sin)
                elif sin < 0:
                    ydist = agent_pos[1] * (1 / sin)
                else:
                    ydist = np.inf
                walldist = min(xdist, ydist)
                walls_obs[s] = walldist / self.agent_dist_norm

            observations.append(np.hstack((targets_obs, agents_obs, walls_obs)))

        return observations

    def reset(self):
        # x,y for each agent
        self.agents_pos = self.np_random.rand(2, self.n_agents) * self.size * self.agents_area + self.size * (
                1 - self.agents_area) / 2
        # angle for each agent
        self.agents_angle = self.np_random.rand(self.n_agents) * 2 * np.pi
        # x,y for each target
        self.targets = self.np_random.rand(2, self.n_targets) * self.size
        # move them outside the agents area
        if self.targets_outside:
            central_targets = np.all(abs(self.targets/self.size-0.5)<self.agents_area/2,0)
            while np.any(central_targets):
                self.targets[:,central_targets] = self.np_random.rand(2, sum(central_targets)) * self.size
                central_targets = np.all(abs(self.targets/self.size-0.5)<self.agents_area/2,0)

        # is the target still to be collected?
        self.alive = np.full(self.n_targets, True)

        self.agents_info = [None] * self.n_agents
        self.agents_obs = self._get_state()

        self.agents_rewards = np.zeros(self.n_agents)

        self.viewer = None

        return self.agents_obs

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        class Sprite(rendering.Geom):
            def __init__(self, image, size):
                rendering.Geom.__init__(self)
                self.sprite = pyglet.sprite.Sprite(image)
                self.sprite.scale = size / self.sprite.width

            def render1(self):
                self.sprite.draw()

        class Text(rendering.Geom):
            def __init__(self, text, *args, **kwargs):
                rendering.Geom.__init__(self)
                self.label = pyglet.text.Label(text, *args, **kwargs)

            def render1(self):
                self.label.draw()

        if self.agents_pos is None:
            return None

        screen_width = screen_height = 700

        scale = screen_width / self.size
        agent_size = 1
        target_size = 1

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-scale, screen_width + scale, -scale, screen_height + scale)
            boundary = rendering.make_polygon(
                [[0, 0], [0, screen_height], [screen_width, screen_height], [screen_width, 0]], filled=False)
            self.viewer.add_geom(boundary)
            self.agents_transform = list()
            self.targets_geom = list()
            path = os.path.dirname(__file__)
            agent_img = pyglet.image.load(path + "/img/robot.png")
            agent_img.anchor_x = agent_img.width // 2
            agent_img.anchor_y = agent_img.height // 2
            target_img = pyglet.image.load(path + "/img/pineapple.png")
            target_img.anchor_x = target_img.width // 2
            target_img.anchor_y = target_img.height // 2
            for _ in range(self.n_agents):
                agent_sprite = Sprite(agent_img, scale)
                agent_transform = rendering.Transform()
                agent_sprite.add_attr(agent_transform)
                self.viewer.add_geom(agent_sprite)
                self.agents_transform.append(agent_transform)
            for t in range(self.n_targets):
                target_sprite = Sprite(target_img, scale)
                target_transform = rendering.Transform(translation=self.targets[:, t] * scale)
                target_sprite.add_attr(target_transform)
                self.viewer.add_geom(target_sprite)
                self.targets_geom.append(target_sprite)

        for n, (angle, pos, transform, info, obs, reward) in enumerate(zip(
                self.agents_angle,
                self.agents_pos.T,
                self.agents_transform,
                self.agents_info,
                self.agents_obs,
                self.agents_rewards)):
            transform.set_translation(*pos * scale)
            transform.set_rotation(angle)
            # reward_text = Text(str(round(reward, 1)), x=(pos[0] + 1) * scale, y=(pos[1] + 1) * scale,
            #                    color=(0, 0, 255, 255))
            # self.viewer.add_onetime(reward_text)
            # id_text = Text(str(n), x=(pos[0] - 1) * scale, y=(pos[1] - 1) * scale,
            #                     color=(0, 0, 0, 255))
            # self.viewer.add_onetime(id_text)

            if n==0 and info is not None:
                for sector, target in enumerate(info["targets_seen"]):
                    if target >= 0:
                        target_pos = self.targets[:, self.alive][:, target] * scale
                        self.viewer.draw_line(pos * scale, target_pos, color=(1, 0, 0))
                        sect_text = Text(str(sector), x=target_pos[0], y=target_pos[1], color=(255, 0, 0, 255))
                        self.viewer.add_onetime(sect_text)

                for sector,other in enumerate(info["agents_seen"]):
                    if other>=0:
                        other_pos = self.agents_pos[:,other]*scale
                        self.viewer.draw_line(pos*scale,other_pos,color=(0,0,1))
                        sect_text = Text(str(sector),x=other_pos[0],y=other_pos[1],color=(0,0,255,255))
                        self.viewer.add_onetime(sect_text)

                sector_width = 2*np.pi/self.agent_obs_sectors
                for sector,dist in enumerate(obs[self.agent_obs_sectors*2:]):
                    dist = int(round(dist*self.agent_dist_norm))
                    direction = angle-sector_width*sector
                    point = (pos+np.array([np.cos(direction),np.sin(direction)])*10)*scale
                    self.viewer.draw_line(pos*scale,point,color=(1,0,1))
                    sect_text = Text(str(sector),x=point[0],y=point[1],color=(255,0,255,255))
                    self.viewer.add_onetime(sect_text)

        for alive, geom in zip(self.alive, self.targets_geom):
            if not alive:
                geom.render = do_not_render

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class FoodCollectingCoop(FoodCollecting):
    def __init__(self):
        super().__init__(cooperation_level=1)


class FoodCollectingSingle(FoodCollecting):
    def __init__(self):
        super().__init__(n_agents=1, n_targets=33, size=30)


class FoodCollectingFewTargets(FoodCollecting):
    def __init__(self):
        super().__init__(n_targets=3)


class FoodCollectingFewClose(FoodCollecting):
    def __init__(self):
        super().__init__(n_targets=3, agents_area=0.2)


class FoodCollectingClose(FoodCollecting):
    def __init__(self):
        super().__init__(agents_area=0.2)

class FoodCollectingFour(FoodCollecting):
     def __init__(self):
        super().__init__(n_agents=4)

class FoodCollectingTen(FoodCollecting):
    def __init__(self):
        super().__init__(n_agents=10)

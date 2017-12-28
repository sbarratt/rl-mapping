import argparse
import os
import random

from envs import MappingEnvironment, LocalISM

import numpy as np

parser = argparse.ArgumentParser()

# General Stuff
parser.add_argument('--experiment', default='runs/myopic', help='folder to put results of experiment in')

# Environment
parser.add_argument('--N', type=int, default=25, help='size of grid')
parser.add_argument('--map_p', type=float, default=.1, help='probability map location is occupied')
parser.add_argument('--prims', action='store_true', help='prims algorithm for filling in map')
parser.add_argument('--episode_length', type=int, default=300, help='length of episode')

# Sensor
parser.add_argument('--sensor_type', default='local', help='local | range')
parser.add_argument('--sensor_span', type=int, default=1, help='span of sensor')
parser.add_argument('--sensor_p', type=float, default=.8, help='probability sensor reading is correct')

parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='random seed')

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

# make experiment path
os.makedirs(opt.experiment, exist_ok=True)
with open(os.path.join(opt.experiment, 'config.txt'), 'w') as f:
    f.write(str(opt))

# Initialize sensor
if opt.sensor_type == 'local':
    ism_proto = lambda x: LocalISM(x, span=opt.sensor_span, p_correct=opt.sensor_p)
else:
    raise Exception('sensor type not supported.')

# Initialize environment
env = MappingEnvironment(ism_proto, N=opt.N, p=opt.map_p, episode_length=opt.episode_length, prims=opt.prims)

# Test
rewards = []
for k in range(1000):
    obs = env.reset()

    done = False
    R = 0
    while not done:
        # Perform a_t according to actor_criticb
        best_ent = 0
        best_action = 0
        for i, (x, y) in enumerate([[1, 0], [-1, 0], [0, 1], [0, -1]]):
            p = (obs[opt.N-1+x, opt.N-1+y, 0]+1)/2
            mask = np.ones((3, 3))
            mask[1,1] = 0
            ent = obs[opt.N-1-1+x:opt.N-1+2+x, opt.N-1-1+y:opt.N-1+2+y, 1]
            expected_ent = (1-p) * np.sum(mask * (ent+1)/2)
            if expected_ent > best_ent:
                best_ent = expected_ent
                best_action = i
        if random.random() < 0:
            a = random.randint(0, 3)
        else:
            a = best_action

        # Receive reward r_t and new state s_t+1
        obs, reward, done, info = env.step(a)

        R += reward
    print (R)
    rewards.append(R)

np.save(os.path.join(opt.experiment, 'rewards_test'), rewards)

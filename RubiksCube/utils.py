from collections import Counter
from random import choice
import tqdm
import time
import multiprocessing

import numpy as np
import pycuber as pc

action_map = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11,
              'F2': 12, 'B2': 13, 'U2': 14, 'D2': 15, 'L2': 16, 'R2': 17, "F2'": 18, "B2'": 19, "U2'": 20, "D2'": 21,
              "L2'": 22, "R2'": 23}

action_map_small = {
    'F': 0,
    'B': 1,
    'U': 2,
    'D': 3,
    'L': 4,
    'R': 5,
    "F'": 6,
    "B'": 7,
    "U'": 8,
    "D'": 9,
    "L'": 10,
    "R'": 11
}

inv_action_map = {v: k for k, v in action_map.items()}

color_map = {
    'green': 0,
    'blue': 1,
    'yellow': 2,
    'red': 3,
    'orange': 4,
    'white': 5}

color_list_map = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
                  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}


def flatten(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.append(x[i][j].colour)
    return flat


def flatted_1d(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.extend(color_list_map[x[i][j].colour])
    return flat


def order(data):
    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    probs = [float(c) / len(data) for c in counts.values()]

    return max(probs)


def get_perc_solved(cube):
    flat = flatten(cube)
    perc_side = [order(flat[i:(i + 9)]) for i in range(0, 9 * 6, 9)]
    return np.mean(perc_side)


def gen_sample(n_steps=6):
    cube = pc.Cube()

    transformation = [choice(list(action_map.keys())) for _ in range(n_steps)]

    my_formula = pc.Formula(transformation)

    cube(my_formula)

    my_formula.reverse()

    sample_X = []
    sample_Y = []
    cubes = []

    for s in my_formula:
        sample_X.append(flatted_1d(cube))
        sample_Y.append(action_map[s.name])
        cubes.append(cube.copy())
        cube(s.name)

    return sample_X, sample_Y, cubes


def gen_sample_basic(n_steps=6):
    cube = pc.Cube()

    transformation = [choice(list(action_map_small.keys()))
                      for _ in range(n_steps)]

    my_formula = pc.Formula(transformation)

    cube(my_formula)

    my_formula.reverse()

    sample_X = []
    sample_Y = []
    cubes = []

    for s in my_formula:
        sample_X.append(flatted_1d(cube))
        sample_Y.append(action_map[s.name])
        cubes.append(cube.copy())
        cube(s.name)

    return sample_X, sample_Y, cubes


def gen_seq(n_steps=6):
    cube = pc.Cube()

    transformation = [choice(list(action_map_small.keys()))
                      for _ in range(n_steps)]

    my_formula = pc.Formula(transformation)

    cube(my_formula)

    my_formula.reverse()

    cubes = []
    distance_to_solved = []

    for i, s in enumerate(my_formula):
        cubes.append(cube.copy())
        cube(s.name)
        distance_to_solved.append(n_steps - i)

    return cubes, distance_to_solved


def possible_actions_basic(cube):

    flat_cubes = []
    rewards = []

    for a in action_map_small:

        cube_copy = cube.copy()
        cube_copy = cube_copy(a)

        flat_cubes.append(flatted_1d(cube_copy))
        rewards.append(2 * int(get_perc_solved(cube_copy) > 0.99) - 1)

    return flat_cubes, rewards


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

"""
Generate a sequence with 25 turns
"""
def gen_25(num):
    return gen_seq(25)

def generate_data(N_SAMPLES):
    cubes = []
    dist_solved = []
    c_next_reward = []
    flat_next_states = []
    cube_flat = []

    with multiprocessing.Pool() as pool:
        for _cubes, _dist_solved, in tqdm(
                pool.imap(gen_25, range(N_SAMPLES))):
            cubes.extend(_cubes)
            dist_solved.extend(_dist_solved)

        for a, b, c in tqdm(pool.imap(get_all_possible, cubes)):
            c_next_reward.append(a)
            flat_next_states.extend(b)
            cube_flat.append(c)

    return cubes, dist_solved, c_next_reward, flat_next_states, cube_flat



"""
Get all possible cubes from all basic/simple actions
"""
def get_all_possible(c):
    flat_cubes, rewards = possible_actions_basic(c)
    return rewards, flat_cubes, flatted_1d(c)

"""
Generate a sequence with 25 turns
"""
def generate_25(num):
    return gen_seq(25)


"""
Get all possible cubes from all basic/simple actions
"""
def get_all_possible(c):
    flat_cubes, rewards = possible_actions_basic(c)
    return rewards, flat_cubes, flatted_1d(c)


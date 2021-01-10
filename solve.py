#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from RubiksCube import SolveCubeNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve a rubik\'s cube')
    parser.add_argument(
        '--model-path',
        '-m',
        help="path where the model will be stored",
        type=str,
        required=True)
    parser.add_argument(
        '--num-shuffles',
        '-n',
        help="number of actions to shuffle",
        type=int,
        default=10)
    args = parser.parse_args()

    solver = SolveCubeNN(args.model_path, args.num_shuffles)
    solver.solve_random()

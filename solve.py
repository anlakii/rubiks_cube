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
    args = parser.parse_args()
    
    solver = SolveCubeNN(args.model_path)
    solver.solve_random()

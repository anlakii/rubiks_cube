#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from RubiksCube import TrainCubeNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a rubik\'s cube model')
    parser.add_argument(
        '--epochs',
        '-e',
        help="num of epochs",
        type=int, required=True)
    parser.add_argument(
        '--model-path',
        '-m',
        help="path where the model will be stored",
        type=str,
        required=True)
    parser.add_argument(
        '--samples',
        '-s',
        help="num of samples",
        type=int,
        default=100)
    parser.add_argument(
        '--train-dir',
        '-t',
        help="dir to train data",
        type=str)
    parser.add_argument(
        '--generate-data',
        '-g',
        help="generate training data",
        default=False,
        action='store_true')

    args = parser.parse_args()

    trainer = TrainCubeNN(
        model_path=args.model_path,
        n_samples=args.samples,
        n_epoch=args.epochs,
        gen_data=args.generate_data)
    trainer.run()

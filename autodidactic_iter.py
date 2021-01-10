import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib
from keras import backend as K
import multiprocessing
import numpy as np
import os
import time
from tqdm import tqdm
import argparse

from utils import action_map_small, gen_sequence, get_all_possible_actions_cube_small, chunker, \
    flatted_1d

def generate_25(num):
    return gen_sequence(25)

def get_all_possible(c):
    flat_cubes, rewards = get_all_possible_actions_cube_small(c)
    return rewards, flat_cubes, flatted_1d(c)


def get_all_possible_loop(cubes, start, end, c_next_reward, flat_next_states, cube_flat):
    for i in range(start, end):
        c = cubes[i]
        flat_cubes, rewards = get_all_possible_actions_cube_small(c)
        c_next_reward.append(rewards)
        flat_next_states.extend(flat_cubes)
        cube_flat.append(flatted_1d(c))


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

def get_model(lr=0.0001):
    input1 = Input((324,))

    # Densly connected NN layer
    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(1024)

    d4 = Dense(50)

    x = d1(input1)
    x = LeakyReLU()(x)
    x = d2(x)
    x = LeakyReLU()(x)
    x = d3(x)
    x = LeakyReLU()(x)
    x = d4(x)
    x = LeakyReLU()(x)

    out_value = Dense(1, activation="linear", name="value")(x)
    out_pol = Dense(len(action_map_small), activation="softmax", name="policy")(x)

    model = Model(input1, [out_value, out_pol])

    model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                  metrics={"policy": acc})
    model.summary()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train a rubik\'s cube model')
    parser.add_argument('--epochs', '-e', help="num of epochs", type=int, required=True)
    parser.add_argument('--model-dir', '-m', help="dir to store model to", type=str)
    parser.add_argument('--samples', '-s', help="num of samples", type=int, default=100)
    args = parser.parse_args()
    manager = multiprocessing.Manager()

    tf.compat.v1.enable_eager_execution()
    print(device_lib.list_local_devices())
    print(K.tensorflow_backend._get_available_gpus())

    N_SAMPLES = args.samples
    N_EPOCH = args.epochs

    file_path = 'auto.h5'
    final_path = os.path.join(args.model_dir, 'auto.h5')

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)
    callbacks_list = [checkpoint, early, reduce_on_plateau]

    lr = 0.0001
    model = get_model(lr=lr)
    print(f'Learning rate: {lr}')

    if tf.io.gfile.exists(final_path):
        print("loading weights")
        with tf.gfile.Open(final_path, 'rb') as infile:
            with file_io.FileIO(file_path, mode='wb+') as outfile:
                outfile.write(infile.read())
        model.load_weights(file_path)


    for i in range(N_EPOCH):
        cubes = []
        dist_solved = []

        c_next_reward = []
        flat_next_states = []
        cube_flat = []

        with multiprocessing.Pool() as pool:
            for _cubes, _dist_solved, in tqdm(pool.imap(generate_25, range(N_SAMPLES))):
                cubes.extend(_cubes)
                dist_solved.extend(_dist_solved)

            for a, b, c in tqdm(pool.imap(get_all_possible, cubes)):
                c_next_reward.append(a)
                flat_next_states.extend(b)
                cube_flat.append(c)

        for _ in range(20):

            target_val = []
            target_pol = []

            next_val, _ = model.predict(np.array(flat_next_states), batch_size=1024, use_multiprocessing=True, workers=20)
            next_val = next_val.ravel().tolist()
            next_val = list(chunker(next_val, size=len(action_map_small)))

            for c, rewards, values in tqdm(zip(cubes, c_next_reward, next_val)):
                reward_values = 0.4*np.array(rewards) + np.array(values)
                target_v = np.max(reward_values)
                target_p = np.argmax(reward_values)
                target_val.append(target_v)
                target_pol.append(target_p)

            target_val = (target_val-np.mean(target_val))/(np.std(target_val)+0.01)

            sample_weights = 1. / np.array(dist_solved)
            sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)

            x = np.array(cube_flat)
            y = [np.array(target_val), np.array(target_pol)[..., np.newaxis]]

            model.fit(x, y, epochs=1, batch_size=128, sample_weight=[sample_weights, sample_weights])

        print(f'Epoch: {i}')

        model.save_weights(file_path)

    with tf.gfile.Open(final_path, 'wb+') as outfile:
        with file_io.FileIO(file_path, mode='rb') as infile:
            outfile.write(infile.read())


if __name__ == "__main__":
    main()

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
    flatten_1d_b


def get_all_possible_loop(cubes, start, end, cube_next_reward, flat_next_states, cube_flat):
    for i in range(start, end):
        c = cubes[i]
        flat_cubes, rewards = get_all_possible_actions_cube_small(c)
        cube_next_reward.append(rewards)
        flat_next_states.extend(flat_cubes)
        cube_flat.append(flatten_1d_b(c))


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

def get_model(lr=0.0001):
    input1 = Input((324,))

    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(1024)

    d4 = Dense(50)

    x1 = d1(input1)
    x1 = LeakyReLU()(x1)
    x1 = d2(x1)
    x1 = LeakyReLU()(x1)
    x1 = d3(x1)
    x1 = LeakyReLU()(x1)
    x1 = d4(x1)
    x1 = LeakyReLU()(x1)

    out_value = Dense(1, activation="linear", name="value")(x1)
    out_policy = Dense(len(action_map_small), activation="softmax", name="policy")(x1)

    model = Model(input1, [out_value, out_policy])

    model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                  metrics={"policy": acc})
    model.summary()

    return model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a rubik\'s cube model')
    parser.add_argument('--epochs', '-e', help="num of epochs", type=int, required=True)
    parser.add_argument('--model-dir', '-m', help="dir to store model to", type=str)

    args = parser.parse_args()
    manager = multiprocessing.Manager()

    tf.compat.v1.enable_eager_execution()

    print(device_lib.list_local_devices())
    print(K.tensorflow_backend._get_available_gpus())

    N_SAMPLES = 100
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
        cubes = manager.list()
        distance_to_solved = []
        for j in tqdm(range(N_SAMPLES)):
            _cubes, _distance_to_solved = gen_sequence(25)
            cubes.extend(_cubes)
            distance_to_solved.extend(_distance_to_solved)

        
        cube_next_reward = manager.list()
        flat_next_states = manager.list()
        cube_flat = manager.list


        procs = []
        print("STARTING PROCESSES")

        cpus = multiprocessing.cpu_count()

        total = len(cubes)
        for i in range(cpus):
            proc = None
            if i + 1 == cpus:
                proc = multiprocessing.Process(target=get_all_possible_loop, args=(cubes, i * total // cpus, total, cube_next_reward, flat_next_states, cube_flat))
            else:
                proc = multiprocessing.Process(target=get_all_possible_loop, args=(cubes, i * total // cpus, (i + 1) * total // cpus, cube_next_reward, flat_next_states, cube_flat))
            proc.start()
            procs.append(proc)

        for p in procs:
            p.join()

            

        print("JOINING THREADS")

        for t in threads:
            t.join()

        print("DONE")



        for _ in range(20):

            cube_target_value = []
            cube_target_policy = []

            next_state_value, _ = model.predict(np.array(flat_next_states), batch_size=1024, use_multiprocessing=True, workers=20)
            next_state_value = next_state_value.ravel().tolist()
            next_state_value = list(chunker(next_state_value, size=len(action_map_small)))

            print("This takes a lot of time...")
            for c, rewards, values in tqdm(zip(cubes, cube_next_reward, next_state_value)):
                r_plus_v = 0.4*np.array(rewards) + np.array(values)
                target_v = np.max(r_plus_v)
                target_p = np.argmax(r_plus_v)
                cube_target_value.append(target_v)
                cube_target_policy.append(target_p)


            cube_target_value = (cube_target_value-np.mean(cube_target_value))/(np.std(cube_target_value)+0.01)

            #print(cube_target_policy[-30:])
            #print(cube_target_value[-30:])

            sample_weights = 1. / np.array(distance_to_solved)
            sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)

            x = np.array(cube_flat)
            y = [np.array(cube_target_value), np.array(cube_target_policy)[..., np.newaxis]]
            print(f"size x: {x.size}")
            print(f"size y: {y[0].size}, {y[1].size}")
            model.fit(x, y, epochs=1, batch_size=128, sample_weight=[sample_weights, sample_weights])
            # sample_weight=[sample_weights, sample_weights],

        print(f'Epoch: {i}')

        model.save_weights(file_path)

    with tf.gfile.Open(final_path, 'wb+') as outfile:
        with file_io.FileIO(file_path, mode='rb') as infile:
            outfile.write(infile.read())

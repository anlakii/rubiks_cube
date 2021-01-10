import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib
from keras import backend as K
from tqdm import tqdm
import numpy as np
import multiprocessing
import argparse
import os

from .utils import action_map_small, gen_seq, possible_actions_basic, chunker, \
    flatted_1d, generate_25, get_all_possible

class TrainCubeNN:
    def __init__(self, **kwargs):
        self.model_path = kwargs.get('model_path')
        if not self.model_path:
            raise ValueError('Model path must be set!')

        self.n_samples = 100
        self.n_epoch = 10
        self.gen_data = False
        self.train_path = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    """
    Creates a Keras model with the specified learning rate
    """
    def get_model(self, lr=0.0001):
        input1 = Input((324,))

        x = Dense(2048)(input1)
        x = LeakyReLU()(x)
        x = Dense(2048)(x)
        x = LeakyReLU()(x)
        x = Dense(2048)(x)
        x = LeakyReLU()(x)
        x = Dense(100)(x)
        x = LeakyReLU()(x)

        out_value = Dense(1, activation="linear", name="value")(x)
        out_pol = Dense(
            len(action_map_small),
            activation="softmax",
            name="policy")(x)

        model = Model(input1, [out_value, out_pol])

        model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                      metrics={"policy": acc})
        model.summary()

        return model

    def run(self):
        file_path = 'auto.h5'

        train_file = None
        if self.train_path:
            train_file = open(self.train_path, "rb")

        checkpoint = ModelCheckpoint(
            file_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min')

        early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)
        reduce_on_plateau = ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.1,
            patience=50,
            min_lr=1e-8)
        callbacks_list = [checkpoint, early, reduce_on_plateau]

        lr = 0.0001
        model = self.get_model(lr=lr)
        print(f'Learning rate: {lr}')

        if tf.io.gfile.exists(self.model_path):
            print("loading weights")
            with tf.gfile.Open(self.model_path, 'rb') as infile:
                with file_io.FileIO(file_path, mode='wb+') as outfile:
                    outfile.write(infile.read())
            model.load_weights(file_path)

        for i in range(self.n_epoch):
            cubes = []
            dist_solved = []

            c_next_reward = []
            flat_next_states = []
            cube_flat = []

            import pickle
            if not train_file:
                with multiprocessing.Pool() as pool:
                    for _cubes, _dist_solved, in tqdm(
                            pool.imap(generate_25, range(self.n_samples))):
                        cubes.extend(_cubes)
                        dist_solved.extend(_dist_solved)

                    for a, b, c in tqdm(pool.imap(get_all_possible, cubes)):
                        c_next_reward.append(a)
                        flat_next_states.extend(b)
                        cube_flat.append(c)

                if self.gen_data:
                    with open("train.dat", "ab+") as f:
                        pickle.dump((cubes, dist_solved, c_next_reward, flat_next_states, cube_flat), f)
                    break
            else:
                try:
                    while True:
                        print(train_file)
                        a = pickle.load(train_file)
                        cubes = a[0]
                        dist_solved = a[1]
                        c_next_reward = a[2]
                        flat_next_states = a[3]
                        cube_flat = a[4]
                except EOFError:
                    print("End of file!")
                    return

            for _ in range(20):

                target_val = []
                target_pol = []

                next_val, _ = model.predict(
                    np.array(flat_next_states), batch_size=1024, use_multiprocessing=True, workers=20)
                next_val = next_val.ravel().tolist()
                next_val = list(chunker(next_val, size=len(action_map_small)))

                for c, rewards, values in tqdm(
                        zip(cubes, c_next_reward, next_val)):
                    reward_values = 0.4 * np.array(rewards) + np.array(values)
                    val = np.max(reward_values)
                    pol = np.argmax(reward_values)
                    target_val.append(val)
                    target_pol.append(pol)

                target_val = (target_val - np.mean(target_val)) / \
                    (np.std(target_val) + 0.01)

                sample_weights = 1. / np.array(dist_solved)
                sample_weights = sample_weights * \
                    sample_weights.size / np.sum(sample_weights)

                x = np.array(cube_flat)
                y = [np.array(target_val), np.array(target_pol)[..., np.newaxis]]

                model.fit(
                    x,
                    y,
                    epochs=1,
                    batch_size=128,
                    sample_weight=[
                        sample_weights,
                        sample_weights])

            print(f'Epoch: {i}')

            model.save_weights(file_path)

        if self.gen_data:
            with tf.gfile.Open('train.dat', 'rb') as infile:
                with file_io.FileIO(self.train_path, mode='wb+') as outfile:
                    outfile.write(infile.read())
            return

        with tf.gfile.Open(self.model_path, 'wb+') as outfile:
            with file_io.FileIO(file_path, mode='rb') as infile:
                outfile.write(infile.read())


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())




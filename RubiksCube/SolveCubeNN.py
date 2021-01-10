from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import numpy as np
from .utils import *


class SolveCubeNN:
    def __init__(self, model_path, num_shuffles):
        self.model_path = model_path
        self.num_shuffles = num_shuffles

    def get_model(self, lr=0.0001):
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
        out_policy = Dense(
            len(action_map_small),
            activation="softmax",
            name="policy")(x1)

        model = Model(input1, [out_value, out_policy])

        model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                      metrics={"policy": acc})
        model.summary()

        return model

    def solve_random(self):

        model = self.get_model()

        model.load_weights(self.model_path)

        sample_X, sample_Y, cubes = gen_sample(self.num_shuffles)
        cube = cubes[0]
        cube.score = 0

        chosen = []
        actions = []

        seq_list = [[cube]]
        print("--- Generated cube ---")
        print(seq_list[0])
        print("--- Generated cube ---")
        print(get_perc_solved((seq_list[0][-1])))

        existing_cubes = set()

        for j in range(1000):

            X = [flatted_1d(x[-1]) for x in seq_list]

            value, policy = model.predict(np.array(X), batch_size=1024)

            new_seq = []

            for x, policy in zip(seq_list, policy):

                pred = np.argsort(policy)

                action = list(action_map.keys())[pred[-1]]
                first = x[-1].copy()(list(action_map.keys())[pred[-1]])
                actions.append((first, action))
                
                
                action = list(action_map.keys())[pred[-2]]
                second = x[-1].copy()(list(action_map.keys())[pred[-2]])
                actions.append((second, action))

                new_seq.append(x + [first])
                new_seq.append(x + [second])

            last_states_flat = [flatted_1d(x[-1]) for x in new_seq]
            value, _ = model.predict(
                np.array(last_states_flat), batch_size=1024)
            value = value.ravel().tolist()

            for x, v in zip(new_seq, value):
                x[-1].score = v if str(x[-1]) not in existing_cubes else -1

            new_seq.sort(key=lambda x: x[-1].score, reverse=True)

            new_seq = new_seq[:100]

            existing_cubes.update(set([str(x[-1])
                                       for x in new_seq]))

            seq_list = new_seq

            seq_list.sort(
                key=lambda x: get_perc_solved(x[-1]), reverse=True)
            new_seq.sort(
                key=lambda x: get_perc_solved(x[-1]), reverse=True)

            prec = get_perc_solved((new_seq[0][-1]))
            chosen.append(new_seq[0])
            print(prec)

            if prec == 1:
                break

        print(chosen[-1])

def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

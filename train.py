import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import sys



if len(sys.argv) != 4:
    print("Usage: python3 train.py [project name] [epochs] [# of trials]")
    quit()
dataset = pd.read_pickle("data/MHP.pkl")



BATCH_SIZE =32 
BUFFER_SIZE = 10000
history = 1117
target = 180
EPOCHS = int(sys.argv[2])
TRIALS = int(sys.argv[3])
PROJECT_NAME = sys.argv[1]

features = dataset[['DEPTH', 'SWC', 'TEMP']]
features.index = dataset['DATE TIME']
TRAIN_SPLIT = len(features)*3//5

data = features.values
data_mean = data[:TRAIN_SPLIT].mean(axis=0)
data_std = data[:TRAIN_SPLIT].std(axis=0)

data = (data-data_mean)/data_std


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def create_time_steps(length):
  return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(13, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/1, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/1, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
                hp.Int('input_unit', min_value=128,max_value=384,step=32),
                return_sequences=True,
                input_shape=(x_train.shape[-2:])))
    
    model.add(tf.keras.layers.Dropout(hp.Float('1st dropout_rate',min_value=.3, max_value=.6,step=.1)))
    for i in range(hp.Int('n_layers', 2, 4)):
        model.add(tf.keras.layers.LSTM(
                        hp.Int(f'lstm_{i}_units',min_value=96,max_value=256,step=32),
                        return_sequences=True))
    model.add(tf.keras.layers.LSTM(
        hp.Int('layer_2_neurons',min_value=32,max_value=128,step=32)))
    model.add(tf.keras.layers.Dropout(hp.Float('2nd dropout_rate',min_value=.2, max_value=.5,step=.1)))
    model.add(tf.keras.layers.Dense(180, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

    



x_train, y_train = multivariate_data(data, data[:, 1], 0, TRAIN_SPLIT, history, target, 1)
x_test, y_test = multivariate_data(data, data[:, 1], TRAIN_SPLIT, None, history, target, 1)
val_data_multi = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_data_multi = val_data_multi.batch(BATCH_SIZE)
print(x_test.shape)
tuner = RandomSearch(
        build_model,
        objective='mse',
        max_trials=TRIALS,
        executions_per_trial=4)

tuner.search(
        x=x_train,
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test,y_test),
        callbacks=[tf.keras.callbacks.TensorBoard("./" + PROJECT_NAME + "/log")]
        )
#save the 10 best models in the project directory
best_models = tuner.get_best_models(num_models=10)
for count, model in enumerate(best_models):
    tf.keras.models.save_model(model, PROJECT_NAME + '/LSTM_'+str(count)+'.h5')

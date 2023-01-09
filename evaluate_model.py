import train
import sys
import numpy as np

import tensorflow as tf

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 evalueate_model.py [path to model] [number of images to generate]")
        quit()
    num_images = int(sys.argv[2])
    path = sys.argv[1]
    
    model = tf.keras.models.load_model(path)
    
    for x, y in train.val_data_multi.take(num_images):
        print(x[0].shape)
        new_y = np.concatenate((y[0], y[0], y[0])).reshape((180, 3), order='F')
        pred = model.predict(x)[0]
        print(pred.shape)
        new_pred = np.concatenate((pred, pred, pred)).reshape((180, 3), order='F')
        train.multi_step_plot(train.sc.inverse_transform(x[0]), train.sc.inverse_transform(new_y)[:,1], train.sc.inverse_transform(new_pred)[:,1])

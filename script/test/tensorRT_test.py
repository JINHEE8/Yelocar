import tensorflow as tf
import h5py
import numpy as np

model_version = 'model_01'

with h5py.File('script/Model/dataset_yelo_truck.hdf5', 'r') as hf:
    image_train = np.array(hf['x_train'][:, :, :, :])
    action_train = np.array(hf['y_train'][:, :])
    image_val = np.array(hf['x_val'][:, :, :, :])
    action_val = np.array(hf['y_val'][:, :])

(x_train, y_train), (x_test, y_test) = (image_train, action_train), (image_val, action_val)

x_train /= 255.
x_test /= 255.

model = tf.saved_model.load('./script/Model/'+model_version+'_optimized')
outputs = model(x_test[0:10])
print(outputs)
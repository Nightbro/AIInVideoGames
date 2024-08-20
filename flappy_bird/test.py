import tensorflow as tf


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Ensure TensorFlow uses the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
print(tf.config.list_physical_devices('GPU'))

if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import tensorflow as tf
print(tf.__version__)


tensor = tf.constant([])
print(tensor.device)
import tensorflow as tf

# Prints TensorFlow version (to ensure there are no issues with the TensorFlow installation)
print("TensorFlow version:", tf.__version__)

# Checks and prints out whether TensorFlow can access the GPU
print("Is GPU available:", tf.test.is_gpu_available())

# Lists and prints details about the GPUs if available
print("List of GPU(s):", tf.config.list_physical_devices('GPU'))

# Simple TensorFlow operations to test GPU utilization
tf.random.set_seed(0)
with tf.device('/GPU:0'): # Specifies that the operations are to be run on the GPU
    a = tf.random.normal((100, 100))
    b = tf.random.normal((100, 100))
    c = tf.matmul(a, b)

print("TensorFlow ran a matrix multiplication on GPU:", c.device.endswith('GPU:0'))

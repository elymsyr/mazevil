import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("\n\n\nResults: \n")

# Create a simple matrix multiplication to check if it runs on GPU
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])

with tf.device('/GPU:0'):
    c = tf.matmul(a, b)

print(c)
import tensorflow as tf

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 打印数据集文件的本地路径
print("MNIST dataset is loaded from:", tf.keras.utils.get_file('mnist.npz', 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'))
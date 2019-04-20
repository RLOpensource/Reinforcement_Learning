import tensorflow as tf

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def mlp(x, hidden, activation, output_size):
    for h in hidden:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    x = tf.layers.dense(inputs=x, units=output_size, activation=None)
    return x

def dueling(x, hidden, activation, output_size):
    for h in hidden:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    a = tf.layers.dense(inputs=x, units=1, activation=None)
    v = tf.layers.dense(inputs=x, units=output_size, activation=None)
    return a + v

def cnn_dueling(x, hidden, activation, output_size):
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    x = tf.reshape(x, [-1, 7 * 7 * 64])
    x = dueling(x, hidden, activation, output_size)
    return x
    

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
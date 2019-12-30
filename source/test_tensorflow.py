import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
'''
# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output.decode())
'''
'''
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
'''

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

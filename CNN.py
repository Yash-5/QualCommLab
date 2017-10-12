import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                       )

def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME'
                         )

def CNN(x):
    """
    """
    #Reshaping the Input w.r.t the image size
    x_image = tf.reshape(x, [-1,28,28,1])
    #This is the for the First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])  #Weight for first layer
    b_conv1 = bias_variable([32])#Bias
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #Convolution
    #Pooling
    h_pool1 = max_pool_2x2(h_conv1)

    #This is the Densely Connected Layer
    W_fc1 = weight_variable([14 * 14 * 32, 10])
    b_fc1 = bias_variable([10])
    #Reshaping for dense layer
    h_pool1_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])
    pred = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

    return pred

def main():
    batch_size = None
    image_width = None
    image_height = None
    n_class = None
    learning_rate = None
    num_iter = None
    print_step = None

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, [None, image_width * image_width])
    y = tf.placeholder(tf.float32, [None, n_class])
    
    pred = CNN(x)
    
    cross_entropy = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
                                )
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(num_iter):
        batch = mnist.train.next_batch(batch_size)
        if i%print_step == 0:
            train_accuracy = accuracy.eval(
                                           feed_dict={
                                                      x: batch[0],
                                                      y: batch[1]
                                                     }
                                          )
            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(
                       feed_dict={
                                  x: batch[0],
                                  y: batch[1]
                                 }
                      )

    print("test accuracy %g" % accuracy.eval(
                                             feed_dict={
                                                        x: mnist.test.images,
                                                        y: mnist.test.labels
                                                       }
                                            ))

if __name__ == '__main__':
    main()

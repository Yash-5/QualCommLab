import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

# Functions in which we can use RNN LSMT or GRU
def RNN(Cel, x, weights, biases, nHidden):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(axis=0, num_or_size_splits=nSteps, value=x) 

    if Cel == 'rnn':
        Cell = rnn.BasicRNNCell(nHidden)
    
    elif Cel == 'lstm':
        Cell = rnn.BasicLSTMCell(nHidden)
    
    elif Cel == 'gru':
        Cell = rnn.GRUCell(nHidden)

    outputs, states = rnn.static_rnn(Cell, x, dtype=tf.float32) 
    return tf.matmul(outputs[-1], weights) + biases

def main():   
    Cel = None
    learningRate = None
    trainingIters = None 
    batchSize = None
    nHidden = None
    displayStep = None
    nInput = None
    nSteps = None
    nClasses = None

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)	# Input the Data
    x = tf.placeholder('float', [None, nSteps, nInput])	
    y = tf.placeholder('float', [None, nClasses])

    weights = 

    biases = 
    
    pred = 
    
    cost = 

    optimizer = 

    correctPred = 

    accuracy = 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        
        while step < trainingIters:
            batchX, batchY = dataset.train.next_batch(batchSize)
            batchX = batchX.reshape((batchSize, nSteps, nInput))

            sess.run(optimizer, feed_dict={
                                            x: batchX,
                                            y: batchY
                                          })

            if step % displayStep == 0:
                acc = 
                loss = 

                print("Iter " + str(step) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))
            step +=1
        print('Optimization finished')
        
        testData = dataset.test.images.reshape((-1, nSteps, nInput))
        testLabel = dataset.test.labels
        print("%g final accuracy" %) 

if __name__ == "__main__":
    main()

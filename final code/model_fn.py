##########################################################################################################
##  This file is a initial exploration function design for CNN network, it generated the base model for the 
##  deep learning project. It also have similar function as the tools.py file, the loss, the evaluation
##  and the optimizer. 
##  Auther: Kevin Guo
##  Github: https://github.com/UWMonkey
##  Project Github: https://github.com/bathurstAi/deep_learning
##  Date of creation: 2018-12-28
##  Date updated:2019-01-12
##########################################################################################################

#################################        Coding Start         ############################################
#import packages
import tensorflow as tf 
import pipeline_clean 
import numpy as np


#inference design of the simple 3 conv, 3 pooling and 3 fc CNN
def inference(images,batch_size,unique_class):
    # '''
    # inference: This function it step up a simple CNN archiecture for CNN that uses the 3 convolution layer 
    #            with kernel size of [3,3], and a [2,2] max pooling, with 128 neurons full connected layers
    # Arg: images (Tensor),batch_size (Int),unique_class (Int)
    # images: list of images that are decoded in tensor
    # batch_size: batch size for the training
    # unique_class: number of classification for output
    # Return: logits
    # '''

    with tf.variable_scope('conv1_1', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,3,16],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',
                                    shape = [16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')  
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1_1 = tf.nn.relu(pre_activation,name = scope.name)

    ####  1st pooling layer name = 'pooling1_lrn' ###############
    with tf.variable_scope('pooling1_lrn',reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1_1, ksize=[1,2,2,1],strides=[1,1,1,1],
                                padding='SAME',name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4, bias=1.0,alpha=0.001/9.0,
                            beta=0.75,name="norm1") #cifar10 setting...need to explore

    ####  2st convolution layer name = 'conv2' ###############
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,16,16],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',
                                    shape = [16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding='SAME')  
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name = 'conv2')

    ####  2st pooling layer name = 'pooling2_lrn' ###############
    with tf.variable_scope('pooling2_lrn',reuse=tf.AUTO_REUSE) as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius=4, bias=1.0,alpha=0.001/9.0,
                            beta=0.75,name="norm2") #cifar10 setting...need to explore
        pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1],strides=[1,1,1,1],
                                padding='SAME',name='pooling2')

    ####  3rd convolution layer name = 'conv3' ###############
    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,16,16],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',
                                    shape = [16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding='SAME')  
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation,name = 'conv3')

    ####  3st pooling layer name = 'pooling2_lrn' ###############
    with tf.variable_scope('pooling3_lrn',reuse=tf.AUTO_REUSE) as scope:
        norm3 = tf.nn.lrn(conv3,depth_radius=4, bias=1.0,alpha=0.001/9.0,
                            beta=0.75,name="norm3") 
        pool3 = tf.nn.max_pool(norm3, ksize=[1,2,2,1],strides=[1,1,1,1],
                                padding='SAME',name='pooling3')

    #### 1st fully connected layer #############################

    #for simplicity lets use 128 neurons
    with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
        #reshape = tf.reshape(pool2, shape = [batch_size,-1]) 
        #dim = reshape.get_shape()[1].value #readjsut this to get shape = [dim,128]
        shape = pool3.get_shape().as_list()
        dim = np.prod(shape[1:])
        reshape = tf.reshape(pool3, [-1, dim])
        weights = tf.get_variable('weights',
                                    shape=[dim,128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer()) 
        biases = tf.get_variable('biases',
                                    shape = [128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases, name = scope.name)
       
    #### 2st fully connected layer #############################
    #for simplicity lets use 128 neurons
    with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',shape=[128,128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer()) 
        biases = tf.get_variable('biases',
                                    shape = [128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases, name = 'local4')


    #### output layer softmax ##################################
    with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('softmax_linear',
                                    shape=[128,unique_class],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer()) 
        biases = tf.get_variable('biases',
                                    shape = [unique_class],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name = 'softmax_linear')
    return softmax_linear

# loss function for CNN
def losses(logits, labels):
    # '''
    # loss: implement the loss function that will be used to evaluate the CNN output and the actual value
    #       it can be sparse or non sparse.
    # arg: logits(tnesor), labels(one-hot encoded or not one-hot enchoded)
    # logits: logits that generated from the CNN network
    # labels: actual lables
    # '''
    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name = 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

# optimize function use for CNN
def training(loss,learning_rate):
    # '''
    # optimize: adding the optimization function into the cnn network
    # arg: loss(tensor), learning_rate(float),global_step(momentum speed of learning rate)
    # return: optimizer(tensor)
    # '''   
    with tf.name_scope('optimizer'):
        my_global_step = tf.Variable(0, name = 'global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = my_global_step)
    return train_op

def evaluation(logits,labels):
    # '''
    # evaluation: evaluate the correct prediction of the logits (accuracy)
    # arg: logits(tnesor), labels(one-hot encoded or not one-hot enchoded)
    # logits: logits that generated from the CNN network
    # labels: actual lables
    # return: accuracy in decimal 
    # '''
    with tf.variable_scope('accuracy', reuse = tf.AUTO_REUSE) as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy 

    
# out = images
# # Define the number of channels of each convolution
# # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
# num_channels=3
# momentum = 0.99
# channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
# training =True
# use_batch_norm = True

# IMAGE_PIXEL = 64*64

# for i, c in enumerate(channels):
#     #print(i)
#     #print(c)
    
#     with tf.variable_scope('block_{}'.format(i+1)):
#         print(c)
        
#         out = tf.layers.conv2d(out, c, 3, padding='same')
#         print(out.get_shape().as_list())
#         if use_batch_norm:
#             out = tf.layers.batch_normalization(out, momentum=momentum, training=training)
#         out = tf.nn.relu(out)
#         out = tf.layers.max_pooling2d(out, 2, 2)

# assert out.get_shape().as_list() == [None, 4,4,num_channels*8]

#reshape 
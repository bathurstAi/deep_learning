##########################################################################################################
##  This is the convolution variable, pooling variable, batch normaliztion, loss, accuracy and optimizer functions
##  Purpose: Tools construct the convolution variables, pooling variables, etc. for streamline of the CNN architecture 
##  Auther: Kevin Guo
##  Github: https://github.com/UWMonkey
##  Project Github: https://github.com/bathurstAi/deep_learning
##  Date of creation: 2018-01-08
##  Date updated:2019-01-12
##########################################################################################################

#################################        Coding Start         ############################################

#import packages
import tensorflow as tf
import numpy as np

# conv is the convolution kernel variable 
def conv(layer_name, layer, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    # '''
    # conv: convolution layer of a CNN, that have a kernel default of [3*3], retraining weight = true and stride of 1. 
    #       in return a convolution layer variable.
    # Arg: Layer_name (string), layer (Previous tensor layer), out_channels (int), kernel_size ([Int,Int])
    #      stride ([int,int,int,int]),is_pretrain [True/False]
    # Layer_name: Tensor variable name, not repeateable make sure is unique
    # layer: previous layer in tenor objectives
    # out_channels: Filter size used, during the convolution
    # kernel_size: Reception kernel size, default is [3,3]
    # stride: stride pace of the kernel 
    # is_pretrain: True/False, True = retain weight (training), False = use defined (Test/Validation)
    # Return: tensor layer value
    # '''

    in_channels = layer.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        layer = tf.nn.conv2d(layer, w, stride, padding='SAME', name='conv')
        layer = tf.nn.bias_add(layer, b, name='bias_add')
        layer = tf.nn.relu(layer, name='relu')
        return layer

# pool is the pooling variable
def pool(layer_name, layer, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    # '''
    # pool: pooling layer of a CNN, that perform either max pooling or average pooling and return tensor objective
    # Arg : Layer_name (string), layer (Previous tensor layer), kernel_size ([Int,Int,Int,Int])
    #       stride ([int,int,int,int]),is_max_pool [True/False]
    # Layer_name: Tensor variable name, not repeateable make sure is unique
    # layer: previous layer in tenor objectives
    # kernel = kernel used for pooling 
    # stride = pace of moving the kernel for pooling
    # is_max_pooling = True/False, True = max pooling, False = average pooling
    # '''

    if is_max_pool:
        layer = tf.nn.max_pool(layer, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        layer = tf.nn.avg_pool(layer, kernel, strides=stride, padding='SAME', name=layer_name)
    return layer

# batch_norm is the batch normal function after pooling layer to prevent over fitting
def batch_norm(layer):
    # '''
    # batch_norm: perform batch normalization, currently uses epsilon of CIFRA10, and its setup
    # Arg : Layer (Tensor layer)
    # Return: Layer (Tensor)
    # '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(layer, [0])
    layer = tf.nn.batch_normalization(layer,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return layer

# FC_layer is the fully connected layer for CNN
def FC_layer(layer_name, layer, out_nodes):
    # '''
    # FC_layer: fully connected layer that intake previous layer, and number of neurons used (out side)
    # Layer_name: Tensor variable name, not repeateable make sure is unique
    # layer: previous layer in tenor objectives
    # out_nodes: Number of neurons 
    # '''
    shape = layer.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_layer = tf.reshape(layer, [-1, size]) # flatten into 1D
        
        layer = tf.nn.bias_add(tf.matmul(flat_layer, w), b)
        layer = tf.nn.relu(layer)
        return layer

# loss function for CNN
def loss(logits, labels):
    # '''
    # loss: implement the loss function that will be used to evaluate the CNN output and the actual value
    #       it can be sparse or non sparse.
    # arg: logits(tnesor), labels(one-hot encoded or not one-hot enchoded)
    # logits: logits that generated from the CNN network
    # labels: actual lables
    # '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy') #need 1 hot
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name = 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss

# accuracy function for CNN
def accuracy(logits, labels):
    # '''
    # accuracy: evaluate the correct prediction of the logits (accuracy)
    # arg: logits(tnesor), labels(one-hot encoded or not one-hot enchoded)
    # logits: logits that generated from the CNN network
    # labels: actual lables
    # return: accuracy in decimal 
    # '''
  with tf.name_scope('accuracy') as scope:
      correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)*100.0
      tf.summary.scalar(scope+'/accuracy', accuracy)
  return accuracy

# optimize function use for CNN
def optimize(loss, learning_rate, global_step):
    # '''
    # optimize: adding the optimization function into the cnn network
    # arg: loss(tensor), learning_rate(float),global_step(momentum speed of learning rate)
    # return: optimizer(tensor)
    # '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    



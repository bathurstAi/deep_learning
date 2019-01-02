import tensorflow as tf 
import pipeline_clean 
import numpy as np

def inference(images,batch_size,unique_class):
    # tf.reset_default_graph()
    # image_placeholder =tf.placeholder(tf.float32,shape = (batch_size,IMAGE_PIXEL))
    # shape = [kernal size, kernal size, channels, kernal numbers]
    ####  1st convolution layer name = 'conv1a' ###############
    with tf.variable_scope('conv1a', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,3,16],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                    shape = [16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')  
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name = scope.name)
    ####  1st pooling layer name = 'pooling1_lrn' ###############
    with tf.variable_scope('pooling1_lrn',reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                padding='SAME',name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4, bias=1.0,alpha=0.001/9.0,
                            beta=0.75,name="norm1") #cifar10 setting...need to explore

    ####  2st convolution layer name = 'conv2' ###############
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,16,16],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
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
                            beta=0.75,name="norm1") #cifar10 setting...need to explore
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],strides=[1,1,1,1],
                                padding='SAME',name='pooling2')

    #### 1st fully connected layer #############################
    #for simplicity lets use 128 neurons
    with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
        #reshape = tf.reshape(pool2, shape = [batch_size,-1]) 
        #dim = reshape.get_shape()[1].value #readjsut this to get shape = [dim,128]
        shape = pool2.get_shape().as_list()
        dim = np.prod(shape[1:])
        reshape = tf.reshape(pool2, [-1, dim])
        weights = tf.get_variable('weights',
                                    shape=[dim,128],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)) 
        biases = tf.get_variable('biases',
                                    shape = [128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases, name = scope.name)
        #[64,4096]*[64,128] <----matmul
    #### 2st fully connected layer #############################
    #for simplicity lets use 128 neurons
    with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',shape=[128,128],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)) 
        biases = tf.get_variable('biases',
                                    shape = [128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases, name = 'local4')


    #### output layer softmax ##################################
    with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('softmax_linear',
                                    shape=[128,unique_class],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)) 
        biases = tf.get_variable('biases',
                                    shape = [unique_class],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name = 'softmax_linear')
    return softmax_linear

def losses(logits, labels):
    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name = 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def training(loss,learning_rate):
    with tf.name_scope('optimizer'):
        my_global_step = tf.Variable(0, name = 'global_step', trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = my_global_step)
    return train_op

def evaluation(logits,labels):
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
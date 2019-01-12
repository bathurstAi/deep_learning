##########################################################################################################
##  This file CNN setup that uses the tools.py to setup the architect of the model
##  There are varies version of this file, for different architecture setup, but the general
##  presentation is illustrated in this file. 
##  Auther: Kevin Guo
##  Github: https://github.com/UWMonkey
##  Project Github: https://github.com/bathurstAi/deep_learning
##  Date of creation: 2018-01-09
##  Date updated:2019-01-12
##########################################################################################################
import tensorflow as tf
import tools

# for deeplearning project, this is "first finetune model"
# model_finetune is the setup for 5 convolution - 5 Pooling and batch normlization
def Model_finetune(layer, n_classes, is_pretrain=True):
    # '''
    # Model_finetune: that uses varies tools.py function to setup CNN and returns the logits
    # Arg: layer(tensor), n_classes(int), is_pretrain(Booleen)
    # layer: list of images that are decoded in tensor
    # n_classes: batch size for the training
    # is_pretrain: number of classification for output
    # Return: layer(logits)
    # '''
    with tf.name_scope('Model_finetune'):
        # first conv + pool
        layer = tools.conv('conv1_1', layer, 64, kernel_size=[7,7], stride=[1,1,1,1], is_pretrain=is_pretrain)   
        with tf.name_scope('pool1'):    
            layer = tools.pool('pool1', layer, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        # second conv + pool
        layer = tools.conv('conv2_1', layer, 128, kernel_size=[7,7], stride=[1,1,1,1], is_pretrain=is_pretrain)    
        with tf.name_scope('pool2'):    
            layer = tools.pool('pool2', layer, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
                    
        # thrid conv + pool
        layer = tools.conv('conv3_1', layer, 256, kernel_size=[7,7], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            layer = tools.pool('pool3', layer, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        # fourth conv + pool
        layer = tools.conv('conv4_1', layer, 512, kernel_size=[7,7], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            layer = tools.pool('pool4', layer, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
        # fifth conv + pool
        layer = tools.conv('conv5_1', layer, 512, kernel_size=[7,7], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            layer = tools.pool('pool5', layer, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            
        
        # 3 fully connected layers, last one is softmax
        layer = tools.FC_layer('fc6', layer, out_nodes=2048)        
        with tf.name_scope('batch_norm1'):
            layer = tools.batch_norm(layer)           
        layer = tools.FC_layer('fc7', layer, out_nodes=2048)        
        with tf.name_scope('batch_norm2'):
            layer = tools.batch_norm(layer)            
        layer = tools.FC_layer('fc8', layer, out_nodes=n_classes)
    
        return layer
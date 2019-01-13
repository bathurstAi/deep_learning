# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:36:52 2019

@author: Muhammad Shahbaz
"""

from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import os

# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
DATASET_PATH = 'indoorCVPR_09/Images' # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 67 # CHANGE HERE, total number of classes
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale
learning_rate = 0.0001
num_steps = 100
batch_size = 128
display_step = 5
dropout = 0.75
save_dir='model_results/Mobile_Net/results'
model_dir = 'model_results/Mobile_Net/model'


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size,model):
    if model == "mobilenet":
        imageheight=224
        imagewidth = 224
    else:
        imageheight=299
        imagewidth = 299
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  
            classes = sorted(os.walk(dataset_path).__next__()[1])
        except Exception:  
            pass
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  
                walk = os.walk(c_dir).__next__()
            except Exception:  
                pass
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [imageheight, imagewidth])

    # Normalize
    image = tf.image.per_image_standardization(image)

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y

def getFeatures(X,model):
    #X,Y = read_images(DATASET_PATH, 'folder', 128)
    #init=tr_data['iterator_init_op']
    if model == "mobilenet":
        
        #MobileNet API
        module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2")
        IMG_HEIGHT = 224 # CHANGE HERE, the image height to be resized to
        IMG_WIDTH = 224
    else:
        #NASNet API
        module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
        IMG_HEIGHT = 299 # CHANGE HERE, the image height to be resized to
        IMG_WIDTH = 299
    
    features = module(X)  
    
    return features

# Create model
def conv_net(FEATURES,n_classes, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        fc1 = FEATURES

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 2048)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        #Fully connected layer (in contrib folder for now)
        fc2 = tf.layers.dense(fc1, 1048)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)

        #fc3 = tf.layers.dense(fc2, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        #fc3 = tf.layers.dropout(fc2, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out




        
#Reading Images
X, Y = read_images(DATASET_PATH, MODE, batch_size,"mobilenet")

#Getting Feature vector from the API
features = getFeatures(X,"mobilenet")
logits = conv_net(features,N_CLASSES, dropout, reuse=False, is_training=True)
logits_test = conv_net(features, N_CLASSES, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#Tests
loss_op_test = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_test, labels=Y))

# Evaluate Training of model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Evaluate test model (with test logits, for dropout to be disabled)
correct_pred_test = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))


init = tf.global_variables_initializer()
summary_op= tf.summary.merge_all()


# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    #Start Queue
    tf.train.start_queue_runners()
    summary_writer = tf.summary.FileWriter(save_dir,sess.graph)
    # Training cycle
    for step in range(1, num_steps+1):
        

        if step % display_step == 0:
            
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)
    loss_test, acc_test = sess.run([loss_op_test, accuracy_test])
    print("Test Loss = " + "{:.4f}".format(loss_test) + ", Test Accuracy= " + "{:.3f}".format(acc_test))
    saver.save(sess, 'MobileNet')
import tensorflow as tf 

import numpy as np 
import os
import matplotlib.pyplot as plt 
import skimage.io as io 
import matplotlib.pyplot as plt

def get_file(file_dir):
    images = []
    temp = []
    labels = []
    label_recorder =[]

    for root, sub_folders, files in os.walk(test_dir):
        for name in files:
            images.append(os.path.join(root,name)) #get image path
            label_name = root.split('/')[-1]  #split and find label name
            labels = np.append(labels, label_name) #append label name to a list
            label_name = ""
            #print(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root,name))
            label_recorder=np.append(label_recorder,name.split('/')[-1])
    ############ Lets create a dictionary list with label:int ################
    count = 0
    label_dic = dict()
    for i in label_recorder:
        #print(i)
        label_dic[i]=count
        count +=1  
    ############ Lets change all the label to numeric based on dictionary ################
    labels_copy = np.array(labels)#copy a label version for testing purpose

    count = 0
    for i in labels_copy:
        #print(i)
        labels_copy[count] = label_dic[i]
        count+=1

    #combine images + labels
    temp = np.array([images,labels_copy])
    temp = temp.transpose()
    np.random.shuffle(temp) #randomize the images lists

    #split into image and label list
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = list(map(int,label_list))
    return image_list, label_list


################ in the processing of fixing #########################################33
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    #input:
        #image : list type
        #label : list type
        #image_w: image_width
        #image_H: image_height
        #batch: batch size
        #capacity: the max element in queue
    #return:
        #image_batch: 4D tensor
        #label_batch: 1D tensor

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    #make input queue
    input_queue = tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_content, channels = 3)
    
    ########## Add augment if needed?? decide with team (goes here)################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size, num_threads = 64, capacity = capacity)

    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch, label_batch


batch_size = 2
capacity = 256
img_w = 208
img_h = 208

#test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images//" #kevin's laptop
test_dir = "C://Users//KG//Desktop//MMAI 894//Project//Images//" #kevin's desktop
image_list, label_list = get_file(test_dir)

image_batch, label_batch = get_batch(image_list,label_list, img_w,img_h, batch_size, capacity)

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    try:
        while not coord.should_stop():
            img, label = sess.run([image_batch,label_batch])

            for j in np.array(batch_size):
                print("label: %d" %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print("Done")
    finally:
        coord.request_stop()
    coord.join(threads)




# ######################################################### CODE FOR TESTING ######################

# ####################################################
image_list_a = image_list[0:5]
label_list_a = label_list[0:5]

#image = tf.cast(image_list_a, tf.string)
#label = tf.cast(label_list_a, tf.int32)
#make input queue
# input_queue = tf.train.slice_input_producer([image,label]) #function will be removed soon
input_queue  = tf.data.Dataset.from_tensor_slices((image_list_a,label_list_a)) #new function replace slice_input_producer

# create TensorFlow Iterator object
iterator = tf.data.Iterator.from_structure(input_queue.output_types,
                                   input_queue.output_shapes)
next_element = iterator.get_next()

# create initialization ops to switch between the datasets
init_op = iterator.make_initializer(input_queue)

##start TF session
with tf.Session() as sess:
    # initialize the iterator on the training data
    sess.run(init_op)
    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

##################### Read Image #################################3
number_class = len(label_list_a)
batch_size = 2
capacity = 256
img_w = 208
img_h = 208


def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, number_class)

    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded, img_w, img_h)
    #img_centered = tf.subtract(img_resize, IMAGENET_MEAN) #using imageNet mean to standardarize?
    img_decoded= tf.image.per_image_standardization(img_resize)
    return img_decoded, one_hot

sess = tf.Session()
tr_data = input_queue.map(input_parser)
dataset = tr_data.batch(batch_size)
# Prefetch data for faster consumption
dataset = dataset.prefetch(batch_size)
# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()
# Initialize the iterator
sess.run(iterator.initializer)

# Neural Net Input (images, labels)
X, Y = iterator.get_next()


num_steps = 1000
display_step = 10
learning_rate = 0.001
dropout = 0.75

# THIS IS A CLASSIC CNN 
# Note that a few elements have changed (usage of sess run).

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[2, 208, 208, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, number_class, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = conv_net(X, number_class, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits_train, labels=Y))
    
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)


# Training cycle
for step in range(1, num_steps + 1):

    # Run optimization
    sess.run(train_op)

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        # (note that this consume a new batch of data)
        loss, acc = sess.run([loss_op, accuracy])
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

print("Optimization Finished!")
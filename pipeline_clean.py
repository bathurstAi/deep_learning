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
def get_queue(image_list_a, label_list_a):
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
                print(elem) #for testing 
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
    return input_queue

def input_parser(img_path, label):
    # convert the label to one-hot encoding
    #one_hot = tf.one_hot(label, number_class)
    one_hot = label
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.convert_image_dtype(img_decoded, tf.float32)
    img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded, img_w, img_h)
    #img_centered = tf.subtract(img_resize, IMAGENET_MEAN) #using imageNet mean to standardarize?
    img_decoded= tf.image.per_image_standardization(img_resize)
    return img_decoded, one_hot

##################### Read Image #################################3
number_class = len(label_list_a)
batch_size = 5
img_w = 208
img_h = 208

#test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images//" #kevin's laptop
test_dir = "C://Users//KG//Desktop//MMAI 894//Project//Images//" #kevin's desktop
image_list, label_list = get_file(test_dir)

#sess = tf.Session()
input_queue = get_queue(image_list,label_list)
tr_data = input_queue.map(input_parser)
dataset = tr_data.batch(batch_size)
# Prefetch data for faster consumption
dataset = dataset.prefetch(batch_size)
# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()
# Neural Net Input (images, labels)
X, Y = iterator.get_next()
# Initialize the iterator
init_op = iterator.initializer

#testing
with tf.Session() as sess:
    # Initialize the iterator
    sess.run(init_op)
    #print(sess.run(Y))
    img,label = sess.run([X,Y])
    plt.subplot(2, 1, 1)
    plt.imshow(img[4].astype(np.uint8))
    plt.title(f'label {label[4]}')
    plt.subplot(2, 1, 1)
    plt.imshow(img[3].astype(np.uint8))
    plt.title(f'label {label[3]}')
    plt.show()

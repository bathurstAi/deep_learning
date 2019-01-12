##########################################################################################################
##  This is the image read preparation pipeline used for CNN 
##  Purpose: File loading for MMAI_894 class deep learning project. The pipeline reads a folder location and
##           uses the sub folder name as the label, and perform image reading (JPG), transformation, augmentation.  
##  arg: Folder dir with sub folder name as label, and images within subfolder
##  Return: tensorfor iterator object in a class having, {"image":image_tensor, "labels":label_tenor,"iterator_init_op":iterator}
##  Auther: Kevin Guo
##  Github: https://github.com/UWMonkey
##  Project Github: https://github.com/bathurstAi/deep_learning
##  Date of creation: 2018-12-28
##  Date updated:2019-01-12
##########################################################################################################

#################################        Coding Start         ############################################

## import useful package
import tensorflow as tf 
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt 
import skimage.io as io 
import matplotlib.pyplot as plt

## get_file = file path reading
def get_file(file_dir):
    # '''
    # get_file: Function reading stored image path, and split into train & test, atm uses 20/80 split
    #           The function also convert string label name into numerics, based on sequence of reading
    #           the function also random sort the data pre split
    # Arg     : file dir path: C:/Project/Images
    # Return  : image_tr, label_tr,image_test,label_test --> list of string and int
    # image_tr: list of training image paths
    # label_tr: list of training label ID (Int) 
    # image_test: list of testing image paths
    # label_test: list of testing label ID(int)
    # '''
    images = []
    temp = []
    labels = []
    label_recorder =[]

    for root, sub_folders, files in os.walk(file_dir):
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
    #train and test split
    data = []
    trainData =[]
    testData =[]
    temp_pd = pd.DataFrame(temp)
    unique_class = len(set(temp[:,1]))
    for i in range(unique_class):
        #print(i)
        data = temp[temp_pd.loc[:,1] == str(i)]
        for i in range(0,int(.8*len(data))):
            trainData.append(data[i]) 
        for j in range(int(.8*len(data)),len(data)):
            testData.append(data[j])
    trainData = np.array(trainData)
    testData = np.array(testData)
    # len(temp)
    # len(trainData)
    # len(testData)
    np.random.shuffle(trainData) #randomize the train images lists
    np.random.shuffle(testData) #randomize the test images lists

    #split into image and label list
    #train split
    image_tr = list(trainData[:,0])
    label_tr = list(trainData[:,1])
    label_tr = list(map(int,label_tr))
    #test split
    image_test = list(testData[:,0])
    label_test = list(testData[:,1])
    label_test = list(map(int,label_test))
    return image_tr, label_tr,image_test,label_test

################ in the processing of fixing #########################################33
def get_queue(image_list_a, label_list_a):
    # '''
    # get_queue: matching image path and label into the tensor iterator
    # Arg     : list of image (string),list of labels (string/numeric)
    # Return  : input_queue --> tensor iterator for matched image path and label
    # '''
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
                #print(elem) #for testing 
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
    return input_queue

# input_parse decodes the images, and transformating the images
def input_parser(img_path, label):
    # '''
    # input_parser: the function reads image path then, decode, convert to float32, crop with padding
    #               and then standardize
    # Arg     : list of image (string),list of labels (string/numeric)
    # Return  : img_decoded, labels
    # img_decoded: transformed image data in tensor
    # labels: associated labels in tensor
    # '''
    # convert the label to one-hot encoding
    #labels = tf.one_hot(label, number_class)
    labels = label
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.convert_image_dtype(img_decoded, tf.float32)#changed to 64
    img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded, 227, 227) #need to reset resize
    #img_centered = tf.subtract(img_resize, IMAGENET_MEAN) #using imageNet mean to standardarize?
    img_decoded= tf.image.per_image_standardization(img_resize)
    return img_decoded, labels

# train_preprocess does image augentation
def train_preprocess(image, label):
    # '''
    # train_preprocess: the function reads image data in tensor decoded format, and does image augmentation
    #                   image left/right flip, change the brightness, and random saturation
    # Arg     : list of image (tensor),list of labels (tensor)
    # Return  : image, label
    # img_decoded: augmentated image data in tensor
    # labels: augmentated labels in tensor
    # '''
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

# input_fn, process get_queue,input_parser,and train_preprocess
def input_fn(filenames, labels, batch_size):
    # '''
    # input_fn: reads list of image path, and labels, and perform, process get_queue,input_parser,and train_preprocess 
    #           and batch it together. Returns a class of tensor objects.
    # Arg     : filenames(list),labels (list), batch_size(int)
    # Return  : inputs(class)
    # '''
    input_queue = get_queue(filenames,labels)
    tr_data = input_queue.map(input_parser)
    tr_data = tr_data.map(train_preprocess, num_parallel_calls=4)
    dataset = tr_data.batch(batch_size)
    dataset = dataset.prefetch(1) # make sure you always have one batch ready to serve
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer
    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs



# ##################### testing area #################################
# #number_class = len(label_list_a)
# batch_size = 64

# #test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images//" #kevin's laptop
# test_dir = "C://Users//KG//Desktop//MMAI 894//Project//Images//" #kevin's desktop
# image_list, label_list = get_file(test_dir)
# image_tr,image_test = create_train_test(image_list)
# label_tr,label_test = create_train_test(label_list)
# #len(image_tr) == len(label_tr)
# #len(image_test) == len(label_test)
# # print(image_tr[500])
# # print(label_tr[500])
# tr_data = input_fn(image_tr,label_tr,batch_size)
# test_data = input_fn(image_test,label_test,batch_size)



# #sess = tf.Session()
# input_queue = get_queue(image_list,label_list,img_w, img_h)
# tr_data = input_queue.map(input_parser)
# dataset = tr_data.batch(batch_size)
# # Prefetch data for faster consumption
# dataset = dataset.prefetch(batch_size)
# # Create an iterator over the dataset
# iterator = dataset.make_initializable_iterator()
# # Neural Net Input (images, labels)
# X, Y = iterator.get_next()
# # Initialize the iterator
# init_op = iterator.initializer

#testing
# X = tr_data["images"]
# Y = tr_data["labels"]
# init_op = tr_data["iterator_init_op"]
# with tf.Session() as sess:
#     # Initialize the iterator
#     sess.run(init_op)
#     #print(sess.run(Y))
#     img,label = sess.run([X,Y])
#     print(img[4].astype(np.uint8))
#     plt.subplot(2, 1, 1)
#     plt.imshow(img[4].astype(np.uint8))
#     plt.title(f'label {label[4]}')
#     plt.show()
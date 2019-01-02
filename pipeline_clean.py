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
    np.random.shuffle(temp) #randomize the images lists

    #split into image and label list
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = list(map(int,label_list))
    return image_list, label_list

def create_train_test(imgData):
    trainData = []
    testData = []
    for i in range(0,int(.8*len(imgData))):
        trainData.append(imgData[i]) 
    for j in range(int(.8*len(imgData)),len(imgData)):
        testData.append(imgData[j])
    
    return trainData, testData
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
                #print(elem) #for testing 
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
    return input_queue

def input_parser(img_path, label):
    # convert the label to one-hot encoding
    #labels = tf.one_hot(label, number_class)
    labels = label
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.convert_image_dtype(img_decoded, tf.float32)
    img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded, 227, 227) #need to reset resize
    #img_centered = tf.subtract(img_resize, IMAGENET_MEAN) #using imageNet mean to standardarize?
    img_decoded= tf.image.per_image_standardization(img_resize)
    return img_decoded, labels

def input_fn(filenames, labels, batch_size):
    input_queue = get_queue(filenames,labels)
    tr_data = input_queue.map(input_parser)
    dataset = tr_data.batch(batch_size)
    dataset = dataset.prefetch(1) # make sure you always have one batch ready to serve
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer
    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs


# ##################### Read Image #################################3
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

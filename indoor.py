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

    for root, sub_folders, files in os.walk(test_dir):
        for name in files:
            images.append(os.path.join(root,name)) #get image path
            label_name = root.split('/')[-1]  #split and find label name
            labels = np.append(labels, label_name) #append label name to a list
            label_name = ""
            #print(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root,name))
            
    #combine images + labels
    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp) #randomize the images lists

    #split into image and label list
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])

    return image_list, label_list



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
capacity = 512
img_w = 208
img_h = 208

test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images//"
image_list, label_list = get_file(test_dir)
image_batch, label_batch = get_batch(image_list,label_list, img_w,img_h, batch_size, capacity)



with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    try:
        while not coord.should_stop() and i<1:
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

########################## Testing space for file name and path ######################
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
        
        #print(os.path.join(root, name))
        #print(name.split('/')[-1])

temp = np.array([images,labels])
temp = temp.transpose()
#np.random.shuffle(temp) #randomize the images lists

image_list = list(temp[:,0])
label_list = list(temp[:,1])


# ########################## Testing space for batch ######################
# image = tf.cast(image_list, tf.string)
# label = tf.cast(label_list, tf.int32)

# #make input queue
# input_queue = tf.train.slice_input_producer([image,label])

# label=input_queue[1]
# image_content = tf.read_file(input_queue[0])
# image = tf.image.decode_jpeg(image_content, channels=3)
# image = tf.image.resize_image_with_crop_or_pad(image, img_w, img_h)
# image = tf.image.per_image_standardization(image)
# image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size, num_threads = 64, capacity = capacity)
# label_batch = tf.reshape(label_batch,[batch_size])
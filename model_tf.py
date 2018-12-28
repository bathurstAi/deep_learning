import tensorflow as tf 
import pipeline_clean




#number_class = len(label_list_a)
batch_size = 64

#test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images//" #kevin's laptop
test_dir = "C://Users//KG//Desktop//MMAI 894//Project//Images//" #kevin's desktop
image_list, label_list = get_file(test_dir)
image_tr,image_test = create_train_test(image_list)
label_tr,label_test = create_train_test(label_list)
#len(image_tr) == len(label_tr)
#len(image_test) == len(label_test)
# print(image_tr[500])
# print(label_tr[500])
tr_data = input_fn(image_tr,label_tr,batch_size)
test_data = input_fn(image_test,label_test,batch_size)


images = tr_data['images']
#shape = [None, 208,208,3]
out = images
# Define the number of channels of each convolution
# For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
num_channels=3
momentum = 0.99
channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
training =True
use_batch_norm = True

tf.reset_default_graph() 
for i, c in enumerate(channels):
    print(i)
    print(c)
    with tf.variable_scope('block_{}'.format(i+1)):
        out = tf.layers.conv2d(out, c, 3, padding='same')
        if use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=momentum, training=training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)
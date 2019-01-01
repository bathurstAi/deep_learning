import tensorflow as tf 
import pipeline_clean 

test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images3//" #kevin's laptop
    #test_dir = "C://Users//KG//Desktop//MMAI 894//Project//Images//" #kevin's desktop

save_dir = "C://Users//kevin//Desktop//MMAI_894//Project//deep_learning//save_workspace"


image_list, label_list = get_file(test_dir)
image_tr,image_test = create_train_test(image_list)
label_tr,label_test = create_train_test(label_list)
    #len(image_tr) == len(label_tr)
    #len(image_test) == len(label_test)
    # print(image_tr[500])
    # print(label_tr[500])
batch_size = 64
tr_data = input_fn(image_tr,label_tr,batch_size)
test_data = input_fn(image_test,label_test,batch_size)

training =True
use_batch_norm = True
momentum = 0.99
num_channels = 3
unique_class = len(set(label_list))

learning_rate = 0.05
MAX_STEP = 1000

def build_model(is_training, inputs, momentum,num_channels,use_batch_norm,unique_class):
    images = inputs['images']
    # labels = tr_data['labels']
    # init=tr_data['iterator_init_op']

    assert images.get_shape().as_list() == [None, 64, 64, 3]

    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
       
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
    # IMAGE_PIXEL = 64*64

    for i, c in enumerate(channels):
        #print(i)
        #print(c)
        
        with tf.variable_scope('block_{}'.format(i+1),reuse=tf.AUTO_REUSE):
            print(c)
            
            out = tf.layers.conv2d(out, c, 3, padding='same')
            print(out.get_shape().as_list())
            if use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=momentum, training=training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.get_shape().as_list() == [None, 4,4,num_channels*8]

    #reshape
    out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
    #1st fully connected
    with tf.variable_scope('fc_1a', reuse=tf.AUTO_REUSE): 
        out = tf.layers.dense(out, num_channels * 8) 
        if use_batch_norm: 
            out = tf.layers.batch_normalization(out, momentum=momentum, training=training) 
        out = tf.nn.relu(out) 
    #2nd fully connected
    with tf.variable_scope('fc_2a', reuse=tf.AUTO_REUSE): 
        out = tf.layers.dense(out, num_channels * 8) 
        if use_batch_norm: 
            out = tf.layers.batch_normalization(out, momentum=momentum, training=training) 
        out = tf.nn.relu(out) 
    #output layer
    with tf.variable_scope('fc_2', reuse=tf.AUTO_REUSE): 
        logits = tf.layers.dense(out, unique_class) 
    return logits 


def model_fn(is_training, inputs, momentum,num_channels,use_batch_norm,unique_class):
    # inputs = tr_data['labels']
    # labels = inputs
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)


    # MODEL: define the layers of the model 
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE): 
    # Compute the output distribution of the model and the predictions 
        logits = build_model(training,tr_data,momentum,num_channels,use_batch_norm,unique_class) 
        predictions = tf.argmax(logits, 1) 

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    
    # Define training step that minimizes the loss with the Adam optimizer 
    if training: 
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
        init=tf.global_variables_initializer()
        if use_batch_norm: 
    # Add a dependency to update the moving mean and variance for batch normalization 
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
                train_op = optimizer.minimize(loss, global_step=global_step) 
        else: 
            train_op = optimizer.minimize(loss, global_step=global_step) 

    # ----------------------------------------------------------- 
    # METRICS AND SUMMARIES 
    # Metrics for evaluation using tf.metrics (average over whole dataset) 
    with tf.variable_scope("metrics"): 
        metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)), 
                    'loss': tf.metrics.mean(loss) } 

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified 
    for label in range(0, unique_class): 
        mask_label = tf.logical_and(mask, tf.equal(predictions, label)) 
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label) 
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label) 

    model_spec = inputs 
    model_spec['variable_init_op'] = tf.global_variables_initializer() 
    model_spec["predictions"] = predictions 
    model_spec['loss'] = loss 
    model_spec['accuracy'] = accuracy 
    model_spec['metrics_init_op'] = metrics_init_op 
    model_spec['metrics'] = metrics 
    model_spec['update_metrics'] = update_metrics_op 
    model_spec['summary_op'] = tf.summary.merge_all() 

    
    if training: 
        model_spec['train_op'] = train_op 
    
    return model_spec 


model_spec = model_fn(training,tr_data,momentum,num_channels,use_batch_norm,unique_class)


# num_batch = round(len(label_tr)/batch_size)
# init_g = tf.global_variables_initializer()
# init_l = tf.local_variables_initializer()
# init=tr_data['iterator_init_op']

# with tf.Session() as sess:
#     sess.run(init_g)
#     sess.run(init_l)
#     sess.run(init)
#     for i in range(num_batch):
#         print(i)
#         _, loss_val = sess.run([train_op,loss]) #sess.run([train_op,loss])
#         print(loss_val)



# update_metrics = model_spec['update_metrics']
# eval_metrics = model_spec['metrics']
# global_step = tf.train.get_global_step()
# sess.run(model_spec['iterator_init_op'])
# sess.run(model_spec['metrics_init_op'])
##########################################################################################################
##  This file is a initial exploration function design for CNN network, it generated the base model for the 
##  deep learning project. It also have similar function as the tools.py file, the loss, the evaluation
##  and the optimizer. 
##  Auther: Kevin Guo
##  Github: https://github.com/UWMonkey
##  Project Github: https://github.com/bathurstAi/deep_learning
##  Date of creation: 2018-01-09
##  Date updated:2019-01-12
##########################################################################################################

#################################        Coding Start         ############################################
import tensorflow as tf
import pandas as pd
import model_fn
import pipeline_clean
import numpy as np
import os
import model_setup3 as model_setup
import matplotlib.pyplot as plt 

#evaluate testing or validation of the model
def evaluate():
    
    # set tensorgraph
    with tf.Graph().as_default():
        #saved tensor checkpoint
        log_dir ='D://MMAI//MMAI894//Project//Save_workspace4//'
        #image folder, can use seperate folders or original
        test_dir = 'C://Users//KG//Desktop//MMAI_894//Project//Images//'
        n_test = 160
        batch_size=64
        #read image from test
        image_tr, label_tr,image_test,label_test = pipeline_clean.get_file(test_dir)
        test_data = pipeline_clean.input_fn(image_test,label_test,batch_size)  
        images = test_data['images']
        labels = test_data['labels']
        #Requirement for inference
        unique_class = len(set(label_tr))

        #transform testing images to tensor and the model. Use top k to get the highest guess
        #shape = [None, 227,227,3] <---this is what it is working with, width and height need change in pipeline
        test_logits = model_fn.inference(images,batch_size,unique_class)
        top_k_op = tf.nn.in_top_k(test_logits,labels,1)
        saver = tf.train.Saver(tf.global_variables())
        
        #active the tensor session
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            #There is a problem with windows loading saved point, the file using tensor function to read path and returns
            #with double backward slash "//", however tf.restore requires 1 -backward slashed. Therefore currently
            # saver.restore used a constant path, not the ckpt variable.
            #if os or unix submit ckpt i.e saver.restore(sess, ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, "D:/MMAI/MMAI894/Project/Save_workspace3/model.ckpt-150")
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            #get the precision by using correct/batch_size
            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
#call evaluatation function
evaluate()
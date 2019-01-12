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
import pipeline_clean #depend on which version, import according it and name as pipeline_clean
import numpy as np
import os
import model_setup3 as model_setup #depend on which version, import according it and name as model_setup
learning_rate = 0.05
batch_size = 64 #change around to test
MAX_STEP = 500 #change accord to steps
IS_PRETRAIN = True

#Parameter setup to be used in the functions
def run_training3():
    # 1st define image path, and model saving path

    #test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images3//" #kevin's laptop
    test_dir = "C://Users//KG//Desktop//MMAI_894//Project//Images//" #kevin's desktop
    #save_dir = "C://Users//kevin//Desktop//MMAI_894//Project//deep_learning//save_workspace"#kevin's laptop
    save_dir = "D://MMAI//MMAI894//Project//Save_workspace4//" #kevin's desktop
   
    #2nd call pipeline function to get tensor objects
    image_tr, label_tr,image_test,label_test = pipeline_clean.get_file(test_dir)
    # image_tr,image_test = pipeline_clean.create_train_test(image_list)
    # label_tr,label_test = pipeline_clean.create_train_test(label_list)
    tr_data = pipeline_clean.input_fn(image_tr,label_tr,batch_size)
    test_data = pipeline_clean.input_fn(image_test,label_test,batch_size)
    
    
    #call pipeline return class 
    images = tr_data['images']
    labels = tr_data['labels']
    #get number of classes
    unique_class = len(set(label_tr))
    
    
            #shape = [None, 227,227,3] <---this is what it is working with, width and height need change in pipeline
    #call model_setup (currently use model_fn for loss, optimizer and accuracy. But same function is in tools)
    #to pass image and label into logits, loss, optimazer and evaluation
    train_logits = model_setup.Model_finetune(images, unique_class, IS_PRETRAIN)
    train_loss = model_fn.losses(train_logits,labels)
    train_op = model_fn.training(train_loss,learning_rate)
    train_acc = model_fn.evaluation(train_logits, labels)

    #define saving path and active saver
    summary_op= tf.summary.merge_all()
    sess=tf.Session()
    train_writer = tf.summary.FileWriter(save_dir,sess.graph)
    saver = tf.train.Saver()
    
    #initialize operators, will need improvement, currently too many variables to initialized
    init_var=tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    init=tr_data['iterator_init_op']
    
    sess.run(init)
    sess.run(init_var)
    sess.run(init_l)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
   
    #active tensor session and save every defined step
   
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss,train_acc])

            if step % 10==0:
                print("step: %d, train loss: %.2f, train accuarcy = %.2f%%" %(step,tra_loss,tra_acc*100))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 50==0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(save_dir,'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)    
    except tf.errors.OutOfRangeError:
        print("Done Training -- epoch limit reached")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
#run training2
run_training3()
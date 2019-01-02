import tensorflow as tf
import model_fn
import pipeline_clean
import numpy as np
import os

learning_rate = 0.005
batch_size = 16
MAX_STEP = 1000

def run_training():
    #test_dir = "C://Users//Kevin//Desktop//MMAI_894//Images3//" #kevin's laptop
    test_dir = "C://Users//KG//Desktop//MMAI_894//Project//Images//" #kevin's desktop
    #save_dir = "C://Users//kevin//Desktop//MMAI_894//Project//deep_learning//save_workspace"#kevin's laptop
    save_dir = "C://Users//KG//Desktop//MMAI_894//Project//save_workspace//" #kevin's desktop

    image_list, label_list = pipeline_clean.get_file(test_dir)
    image_tr,image_test = pipeline_clean.create_train_test(image_list)
    label_tr,label_test = pipeline_clean.create_train_test(label_list)
    tr_data = pipeline_clean.input_fn(image_tr,label_tr,batch_size)
    test_data = pipeline_clean.input_fn(image_test,label_test,batch_size)

    #def train():

    images = tr_data['images']
    labels = tr_data['labels']
    init=tr_data['iterator_init_op']
        #Requirement for inference
    unique_class = len(set(label_list))
    
    
            #shape = [None, 227,227,3] <---this is what it is working with, width and height need change in pipeline
    logits = model_fn.inference(images,batch_size,unique_class)
    loss = model_fn.losses(logits,labels)

    my_global_step=tf.Variable(0,name = 'global_step',trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step = my_global_step)
    saver = tf.train.Saver(tf.global_variables())
    summary_op= tf.summary.merge_all()


    init_var=tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    init=tr_data['iterator_init_op']
    sess=tf.Session()
    sess.run(init)
    sess.run(init_var)
    sess.run(init_l)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    summary_writer = tf.summary.FileWriter(save_dir,sess.graph)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, loss_value = sess.run([train_op, loss])
            if step % 50==0:
                print("step: %d, loss: %4f" % (step,loss_value))
            if step % 100==0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 2000==0 or (step + 1) ==MAX_STEP:
                checkpoint_path = os.path.join(save_dir,'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)    
    except tf.errors.OutOfRangeError:
        print("Done Training -- epoch limit reached")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

run_training()
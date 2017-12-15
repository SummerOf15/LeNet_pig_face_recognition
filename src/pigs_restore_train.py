#coding:utf-8
'''
Created on 2017年11月25日

@author: Administrator
'''
import tensorflow as tf
import pigs_inference
import pigs_input_data
import numpy as np
import os

#network parameters
BATCH_SIZE=100
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.9 #loss decay rates
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
#path for saving model
MODEL_SAVE_PATH='model/'
MODEL_NAME='model.ckpt'

def train(pigs_img):
    x=tf.placeholder(tf.float32, [BATCH_SIZE,pigs_inference.IMAGE_SIZE,pigs_inference.IMAGE_SIZE,
                                pigs_inference.NUM_CHANNELS], name='x-input')
    y_=tf.placeholder(tf.int32,[BATCH_SIZE,],name='y-input')
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=pigs_inference.inference(x, True, regularizer)
#     tf.summary.image('weighted_image',y,3)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #tf.trainable_variables() get all variables in which train=true
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    #y=tf.clip_by_value(y, 1e-10, 10.0)
    #argmax(y,1) get index of the max value of the row
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
    #cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=tf.arg_max(y_,1), logits=tf.arg_max(y,1))
    #y1=tf.nn.softmax(y)
    #cross_entropy=y_*tf.log(y1)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy_mean',cross_entropy_mean)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, pigs_img.train.num_examples/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)
    tf.summary.scalar('learning_rate',learning_rate)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op('train')
    #initialize tf class
    saver=tf.train.Saver()
    with tf.Session() as sess:
        merged=tf.summary.merge_all()
        writer=tf.summary.FileWriter('des',sess.graph)
        tf.initialize_all_variables().run()
        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            load_path=saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(TRAINING_STEPS):
            xs,ys=pigs_img.train.next_batch(BATCH_SIZE)
            reshaped_xs=np.reshape(xs,(BATCH_SIZE,pigs_inference.IMAGE_SIZE,pigs_inference.IMAGE_SIZE,pigs_inference.NUM_CHANNELS))
            reshaped_ys=np.reshape(ys,(BATCH_SIZE,))
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:reshaped_ys})
            if i%50==0 or i==5:
                disres=sess.run(merged,feed_dict={x:reshaped_xs,y_:reshaped_ys})
                writer.add_summary(disres, i)
                print("After %d training steps, loss on training batch is %g."%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
    
def main(argv=None):
    pigsImg=pigs_input_data.read_data_set('images/','validation/')
    train(pigsImg)
if __name__=='__main__':
    tf.app.run()